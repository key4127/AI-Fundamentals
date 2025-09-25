# MQA: 每个头共享k、v参数
# kv缓存占用变为原来的1/h

from math import sqrt
import torch
import torch.nn as nn


class MultiQueryAttention(nn.Module):

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k // num_heads, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v // num_heads, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        batch, seq_len, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh
        dv = self.dim_v // nh
        
        # after linear: batch, seq_len, dim_k
        q = self.linear_q(x).reshape(batch, seq_len, nh, dk).transpose(1, 2)
        k = self.linear_k(x).reshape(batch, seq_len, 1, dk).transpose(1, 2)
        v = self.linear_v(x).reshape(batch, seq_len, 1, dv).transpose(1, 2)

        k = k.expand(batch, nh, seq_len, dk)
        v = v.expand(batch, nh, seq_len, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact
        dist = torch.softmax(dist, -1)

        att = torch.matmul(dist, v)
        att = att.transpose(1, 2).reshape(batch, seq_len, self.dim_v)
        return att