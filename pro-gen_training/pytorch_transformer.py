from __future__ import print_function
import torch
import os
import tqdm
import pdb
import numpy as np
import platform
import re
import argparse


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
  def __init__(self, d_model_size, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model_size = d_model_size
    
    self.depth = int(d_model_size / self.num_heads)
    
    self.Wq = torch.nn.Linear(d_model_size, d_model_size)
    self.Wk = torch.nn.Linear(d_model_size, d_model_size)
    self.Wv = torch.nn.Linear(d_model_size, d_model_size)
    
    self.dense = torch.nn.Linear(d_model_size, d_model_size)
        
  def split_into_heads(self, x, batch_size):
    x = x.reshape(batch_size, -1, self.num_heads, self.depth)
    return x.permute([0, 2, 1, 3])  # (batch, head, seq_length, head_features)
    
  def forward(self, v, k, q, mask):
    batch_size = q.shape[0]
    
    q = self.Wq(q)
    k = self.Wk(k)
    v = self.Wv(v)
    
    q = self.split_into_heads(q, batch_size)
    k = self.split_into_heads(k, batch_size)
    v = self.split_into_heads(v, batch_size)
    
    scaled_attention = scaled_dot_product_attention(q, k, v, mask).permute([0, 2, 1, 3])
    original_size_attention = scaled_attention.reshape(batch_size, -1, self.d_model_size)
    output = self.dense(original_size_attention)
        
    return output



def point_wise_feed_forward_network(d_model_size, dff):
  return torch.nn.Sequential(torch.nn.Linear(d_model_size, dff), torch.nn.ReLU(), torch.nn.Linear(dff, d_model_size))


class EncoderLayer(torch.nn.Module):
  def __init__(self, d_model_size, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    # multi_head_attention 是一个多头注意力机制模块，通过调用 MultiHeadAttention 类创建。
    # ffn 是一个点式前馈神经网络模块，通过调用 point_wise_feed_forward_network 函数创建
    self.multi_head_attention = MultiHeadAttention(d_model_size, num_heads)
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
  def forward(self, x, mask):
    normed = self.layernorm1(x)
    attn_output  = self.multi_head_attention(normed, normed, normed, mask)
    attn_output = self.dropout1(attn_output)
    out1 = x + attn_output

    out2 = self.layernorm2(out1)
    ffn_output = self.ffn(out2)
    ffn_output = self.dropout2(ffn_output)
    out2 = out1 + ffn_output
    
    return out2


# for oct28 and nov07 ckpt-> num_layers=24, d_model_size=1280, num_heads=16, dff=5120,
# for ctrl_36 -> num_layers=36, d_model_size=1280, num_heads=16, dff=8192, input_vocab_size=50000,rate=0.1,
class Encoder(torch.nn.Module):
  def __init__(self, num_layers=36, d_model_size=1280, num_heads=16, dff=8192, input_vocab_size=50000,
               rate=0.1, **kwargs):
    super(Encoder, self).__init__()

    self.d_model_size = d_model_size
    self.num_layers = num_layers
    
    self.pos_encoding = positional_encoding(input_vocab_size, self.d_model_size).to('cuda')

    # 构建神经网络，每个层共享相同的参数和权重
    for i in range(num_layers):
      # setattr：设置对象的属性值
      setattr(self, "layer%i" % i, EncoderLayer(d_model_size, num_heads, dff, rate))
    
    self.layernorm = torch.nn.LayerNorm(d_model_size, eps=1e-6)  
    self.dropout = torch.nn.Dropout(rate)

  def forward(self, x):
    # 获取输入序列的长度
    seq_len = x.shape[1]
    
    # 创建一个全1上三角矩阵作为掩码矩阵
    mask = torch.triu(torch.ones(seq_len, seq_len), 1).to('cuda')
    x *= np.sqrt(self.d_model_size)
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x)
    
    for i in range(self.num_layers):
      x = getattr(self, "layer%i" % i)(x, mask)
    return self.layernorm(x)
