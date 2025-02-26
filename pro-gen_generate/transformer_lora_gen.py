# 将lora加入ProGen使用的transformer模型之中
# 参考：https://zhuanlan.zhihu.com/p/654897296
from __future__ import print_function
import torch
import numpy as np
import loralib as lora

def angle_defn(pos, i, d_model_size):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model_size))
  return pos * angle_rates

def positional_encoding(position, d_model_size):
  # create the sinusoidal pattern for the positional encoding
  angle_rads = angle_defn(np.arange(position)[:, np.newaxis], np.arange(d_model_size)[np.newaxis, :], d_model_size)
  
  sines = np.sin(angle_rads[:, 0::2])
  cosines = np.cos(angle_rads[:, 1::2])
  
  pos_encoding = torch.tensor(np.concatenate([sines, cosines], axis=-1)[np.newaxis, ...], dtype=torch.float)
  return pos_encoding

def scaled_dot_product_attention(q, k, v, mask):
  # calculate attention，即注意力计算
  matmul_qk = torch.matmul(q, k.permute(0,1,3,2)) # 计算q和k的乘积
  
  dk = k.shape[-1]  # k的最后一个维度的大小，表示嵌入维度大小
  scaled_attention_logits = matmul_qk / np.sqrt(dk) # 对乘积缩放

  if mask is not None:
    # 如果掩码 mask 不为空，则将其乘以一个非常大的负数 -1e9 并加到缩放的注意力权重 scaled_attention_logits 上。
    # 这样做是为了在注意力权重中屏蔽掉无效位置，使得这些位置的注意力权重接近于零
    scaled_attention_logits += (mask * -1e9)
    
  attention_weights = torch.softmax(scaled_attention_logits, dim=-1) 
  output = torch.matmul(attention_weights, v)
  
  return output


class MultiHeadAttention(torch.nn.Module):
  def __init__(self, d_model_size, num_heads, config):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model_size = d_model_size
    self.depth = int(d_model_size / self.num_heads)

    # ------------------------------------------------
    # 使用lora.MergedLinear改写attention层
    # 改写过后的attention层包括预训练部分和低秩适配器部分
    # ------------------------------------------------
    # qkv矩阵
    self.Wq = lora.Linear(d_model_size, d_model_size, 
                          r=config.lora_attn_dim, # r值，即lora中的秩
                          lora_alpha=config.lora_attn_alpha, # alpha值，用于表示微调过程中对新知识的侧重程度
                          lora_dropout=config.lora_dropout, # dropout值
                          fan_in_fan_out=False,  # 表示在计算时是否要做矩阵转置
                          merge_weights=False)  # 表示是否想将低秩适配权重合并到预训练权重中
    self.Wk = torch.nn.Linear(d_model_size, d_model_size)
    self.Wv = lora.Linear(d_model_size, d_model_size, 
                          r=config.lora_attn_dim, # r值，即lora中的秩
                          lora_alpha=config.lora_attn_alpha, # alpha值，用于表示微调过程中对新知识的侧重程度
                          lora_dropout=config.lora_dropout, # dropout值
                          fan_in_fan_out=False,  # 表示在计算时是否要做矩阵转置
                          merge_weights=False)  # 表示是否想将低秩适配权重合并到预训练权重中

    self.dense = torch.nn.Linear(d_model_size, d_model_size)
        
  # 将输入x按照头的数量进行切分
  def split_into_heads(self, x, batch_size):
    x = x.reshape(batch_size, -1, self.num_heads, self.depth)
    return x.permute([0, 2, 1, 3])  # (batch_size, head, seq_length, head_features)


  def forward(self, x, mask, past=None):
    batch_size = x.shape[0]

    # 计算qkv
    # x = self.c_attn(x)
    # query, key, value = x.split(self.split_size, dim=2)
    query = self.Wq(x)
    key = self.Wk(x)
    value = self.Wv(x)

    if not (past is None):  # past不是None时，执行
        key = torch.cat((past[0],key),1)
        value = torch.cat((past[1],value),1)
        mask = torch.cat((torch.zeros(query.shape[1],key.shape[1]-query.shape[1]).to(x.device),mask),1)

    past = [key,value]

    # 将qkv切分为多个头，query=(batch_size, num_heads, seq_length, head_features)
    query = self.split_into_heads(query, batch_size)
    key = self.split_into_heads(key, batch_size)
    value = self.split_into_heads(value, batch_size)
    
    # 计算注意力，scaled_attention = (batch_size, seq_length, num_heads, head_features)
    scaled_attention = scaled_dot_product_attention(query, key, value, mask).permute([0, 2, 1, 3])
    # 拼接所有的结果，original_size_attention = (batch_size, seq_length, d_model_size)
    original_size_attention = scaled_attention.reshape(batch_size, -1, self.d_model_size)
    # 使用线性层进行线性变换
    output = self.dense(original_size_attention)
        
    return output, past



def point_wise_feed_forward_network(d_model_size, dff):
  return torch.nn.Sequential(torch.nn.Linear(d_model_size, dff), torch.nn.ReLU(), torch.nn.Linear(dff, d_model_size))


class EncoderLayer(torch.nn.Module):
  def __init__(self, loraconfig, d_model_size, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    # multi_head_attention 是一个多头注意力机制模块，通过调用 MultiHeadAttention 类创建。
    # ffn 是一个点式前馈神经网络模块，通过调用 point_wise_feed_forward_network 函数创建
    self.multi_head_attention = MultiHeadAttention(d_model_size, num_heads, loraconfig)
    self.ffn = point_wise_feed_forward_network(d_model_size, dff)

    # 创建了两个归一化层模块：layernorm1 和 layernorm2。
    # 它们使用 torch.nn.LayerNorm 类创建，用于在每个编码器层中对输入进行归一化
    self.layernorm1 = torch.nn.LayerNorm(d_model_size, eps=1e-6)
    self.layernorm2 = torch.nn.LayerNorm(d_model_size, eps=1e-6)
    
    self.dropout1 = torch.nn.Dropout(rate)
    self.dropout2 = torch.nn.Dropout(rate)
    

  # 这是 EncoderLayer 类的前向传播方法 forward。在前向传播中，输入 x 经过归一化操作 layernorm1 后得到 normed。
  # 然后，将 normed 传入多头注意力机制 multi_head_attention，并传入 mask 进行注意力计算，得到 attn_output。attn_output 经过 Dropout 操作 dropout1 后与输入 x 相加，得到 out1。
  # 接下来，out1 经过归一化操作 layernorm2 得到 out2。out2 经过点式前馈神经网络 ffn 计算得到 ffn_output，再经过 Dropout 操作 dropout2。
  # 最后，将 out1 和 ffn_output 相加得到 out2，作为该编码器层的输出。
  # 最终，函数返回 out2，即编码器层的输出。
  def forward(self, x, mask, past=None):
    normed = self.layernorm1(x)
    attn_output , past = self.multi_head_attention(normed, mask, past)
    attn_output = self.dropout1(attn_output)
    out1 = x + attn_output

    out2 = self.layernorm2(out1)
    ffn_output = self.ffn(out2)
    ffn_output = self.dropout2(ffn_output)
    out2 = out1 + ffn_output
    
    return out2, past


# for oct28 and nov07 ckpt-> num_layers=24, d_model_size=1280, num_heads=16, dff=5120,
# for ctrl_36 -> num_layers=36, d_model_size=1280, num_heads=16, dff=8192, input_vocab_size=50000,rate=0.1,
class Encoder(torch.nn.Module):
  def __init__(self, loraconfig, num_layers=36, d_model_size=1280, num_heads=16, dff=8192, input_vocab_size=50000,
               rate=0.1, **kwargs):
    super(Encoder, self).__init__()

    self.d_model_size = d_model_size
    self.num_layers = num_layers
    
    self.pos_encoding = positional_encoding(input_vocab_size, self.d_model_size).to('cuda')

    # 构建神经网络，每个层共享相同的参数和权重
    for i in range(num_layers):
      # setattr：设置对象的属性值
      setattr(self, "layer%i" % i, EncoderLayer(loraconfig, d_model_size, num_heads, dff, rate))
    
    self.layernorm = torch.nn.LayerNorm(d_model_size, eps=1e-6)  
    self.dropout = torch.nn.Dropout(rate)

  def forward(self, x, past=None):
    # 获取输入序列的长度
    seq_len = x.shape[1]
    
    # 创建一个全1上三角矩阵作为掩码矩阵
    mask = torch.triu(torch.ones(seq_len, seq_len), 1).to('cuda')
    x *= np.sqrt(self.d_model_size)
     # 进行位置编码
    if past is None:
        x += self.pos_encoding[:, :seq_len, :]
    else:
        past_len=past[0][0].shape[1]
        x += self.pos_encoding[:, past_len:past_len+seq_len, :]

    x = self.dropout(x)

    if past is None:
        past = [None]*self.num_layers

    # 创建一个长度为num_layers的空列表new_past，用于存储每一层的过去序列信息
    new_past = [None]*self.num_layers

    # 对每一层进行编码操作
    for i in range(self.num_layers):
      # getattr函数用于获取编码器的第i层，然后调用这一层的forward函数，传入x、掩码mask和past[i]
      # x就是经过编码器当前层处理后的输出，past_i是更新后的过去序列信息
      x, past_i = getattr(self, "layer%i" % i)(x, mask, past[i])
      new_past[i] = past_i
    return self.layernorm(x), new_past
  

# 新增参数配置类
class lora_config(object):
  def __init__(
      self,
      lora_attn_dim=0,
      lora_attn_alpha=128,
      lora_dropout=0.0,
      lora_r_dropout=0.0,
  ):
      self.lora_attn_dim = lora_attn_dim
      self.lora_attn_alpha = lora_attn_alpha
      self.lora_dropout = lora_dropout
      self.lora_r_dropout = lora_r_dropout