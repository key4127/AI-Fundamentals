from math import sqrt
import torch
import torch.nn as nn


class GroupedQueryAttention(nn.Module):

    def __init__(self, dim_in, dim_k, dim_v, num_heads=8, group_num=4):
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.group_num =  group_num

        self.linear_q = nn.Linear(dim_in, dim_k)
        self.linear_k = nn.Linear(dim_in, dim_k // num_heads * group_num)
        self.linear_v = nn.Linear(dim_in, dim_v // num_heads * group_num)

        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        batch, seq_len, dim_in = x.shape
        assert dim_in == self.dim_in

        ng = self.group_num
        nh = self.num_heads
        dk = self.dim_k // nh
        dv = self.dim_v // nh

        q = self.linear_q(x).reshape(batch, seq_len, nh, dk).transpose(1, 2)
        k = self.linear_k(x).reshape(batch, seq_len, ng, dk).transpose(1, 2)
        v = self.linear_v(x).reshape(batch, seq_len, ng, dv).transpose(1, 2)

        # k: [batch, ng(4), seq_len, dk]

        k = k.repeat_interleave(nh // ng, dim=1)
        v = v.repeat_interleave(nh // ng, dim=1)

        # q: [batch, num_heads, seq_len, dq]
        # k: [batch, num_heads, seq_len, dk] -> [batch, num_heads, dk, seq_len]

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)

        # dist: [batch, num_heads, seq_len, seq_len]
        # v: [batch, num_heads, seq_len, dv]
        # mat: [batch, num_heads, seq_len, dv]

        att = torch.matmul(dist, v)
        att = att.transpose(1, 2).reshape(batch, seq_len, self.dim_v)
        return att