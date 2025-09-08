from math import sqrt

import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    """
    Basic scaled dot-product attention.
    """

    def __init__(self, dim_in, dim_k, dim_v):
        """
        dim_q equals dim_k.
        """
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, False)
        self.linear_k = nn.Linear(dim_in, dim_k, False)
        self.linear_v = nn.Linear(dim_in, dim_v, False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        """
        x is [batch, seq_length, dim_in]
        output is [batch, seq_length, dim_v]
        """
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)

        att = torch.bmm(dist, v)
        return att