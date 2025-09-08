from math import sqrt

import torch
import torch.nn as nn
import numpy as np

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

    def forward(self, x, causal_mask=False):
        """
        x is [batch, seq_length, dim_in]
        output is [batch, seq_length, dim_v]
        """
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x) # q is [batch, seq_length, dim_k]
        k = self.linear_k(x)
        v = self.linear_v(x)

        # transpose: transform k from [batch, seq_length, dim_k] to [batch, dim_k, seq_length]
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
    
        if causal_mask:
            mask = torch.tril(torch.ones(n, n))
            dist = dist.masked_fill(mask == 0, -np.Inf)

        dist = torch.softmax(dist, dim=-1)

        att = torch.bmm(dist, v)
        return att