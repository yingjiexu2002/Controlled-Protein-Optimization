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
  # calculate attention
  matmul_qk = torch.matmul(q, k.permute(0,1,3,2))

  dk = k.shape[-1]
  scaled_attention_logits = matmul_qk / np.sqrt(dk)

  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
  output = torch.matmul(attention_weights, v)

  return output


class MultiHeadAttention(torch.nn.Module):
  def __init__(self, d_model_size, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model_size = d_model_size

    self.depth = int(d_model_size / self.num_heads) # 在下面的参数中，这个数字是80

    self.Wq = torch.nn.Linear(d_model_size, d_model_size)
    self.Wk = torch.nn.Linear(d_model_size, d_model_size)
    self.Wv = torch.nn.Linear(d_model_size, d_model_size)

    self.dense = torch.nn.Linear(d_model_size, d_model_size)

  # 将输入x进行维度划分为各个头的形式
  # 这里使用了reshape和permute函数对张量进行重新排列，以实现拆分操作。
  def split_into_heads(self, x, batch_size):
    x = x.reshape(batch_size, -1, self.num_heads, self.depth)
    return x.permute([0, 2, 1, 3])  # 按照指定的维度顺序重新排列张量的维度

  def forward(self, x, mask, past=None):
    batch_size = x.shape[0]

    q = self.Wq(x)
    k = self.Wk(x)
    v = self.Wv(x)

    if not (past is None):  # past不是None时，执行

        k = torch.cat((past[0],k),1)
        v = torch.cat((past[1],v),1)

        mask = torch.cat((torch.zeros(q.shape[1],k.shape[1]-q.shape[1]).to(x.device),mask),1)

    past = [k,v]

    q = self.split_into_heads(q, batch_size)
    k = self.split_into_heads(k, batch_size)
    v = self.split_into_heads(v, batch_size)


    scaled_attention = scaled_dot_product_attention(q, k, v, mask).permute([0, 2, 1, 3])
    original_size_attention = scaled_attention.reshape(batch_size, -1, self.d_model_size)
    # 最后，通过线性层self.dense对original_size_attention进行线性变换，得到最终的输出
    output = self.dense(original_size_attention)

    return output, past



def point_wise_feed_forward_network(d_model_size, dff):
  return torch.nn.Sequential(torch.nn.Linear(d_model_size, dff), torch.nn.ReLU(), torch.nn.Linear(dff, d_model_size))


class EncoderLayer(torch.nn.Module):
  def __init__(self, d_model_size, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.multi_head_attention = MultiHeadAttention(d_model_size, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model_size, dff)

    self.layernorm1 = torch.nn.LayerNorm(d_model_size, eps=1e-6)
    self.layernorm2 = torch.nn.LayerNorm(d_model_size, eps=1e-6)

    self.dropout1 = torch.nn.Dropout(rate)
    self.dropout2 = torch.nn.Dropout(rate)

  def forward(self, x, mask, past=None):
    normed = self.layernorm1(x)



    attn_output, past  = self.multi_head_attention(normed, mask, past)
    attn_output = self.dropout1(attn_output)
    out1 = x + attn_output

    out2 = self.layernorm2(out1)
    ffn_output = self.ffn(out2)
    ffn_output = self.dropout2(ffn_output)
    out2 = out1 + ffn_output



    return out2, past


# for oct28 and nov07 ckpt-> num_layers=24, d_model_size=1280, num_heads=16, dff=5120,
# for ctrl_36 -> num_layers=36, d_model_size=1280, num_heads=16, dff=8192, input_vocab_size=50000,rate=0.1,
# d_model_size表示模型的维度大小，dff表示隐藏层的大小，rate表示dropout的比例
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

  def forward(self, x, past=None):

    seq_len = x.shape[1]

    # 创建一个上三角矩阵作为mask矩阵
    mask = torch.triu(torch.ones(seq_len, seq_len), 1).to('cuda')
    # 进行特征缩放，用于在注意力计算中提高数值稳定性
    x *= np.sqrt(self.d_model_size)
    
    # 进行位置编码
    if past is None:
        x += self.pos_encoding[:, :seq_len, :]
    else:
        past_len=past[0][0].shape[1]
        x += self.pos_encoding[:, past_len:past_len+seq_len, :]

    # 对x进行dropout操作，将一部分元素置零，以减少模型的过拟合。
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
