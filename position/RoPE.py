import torch
import torch.nn as nn


class RoPE(nn.Module):

    def __init__(self, dim, seq_len, theta=10000.0):
        super().__init__()
        range_vec = torch.arange(0, dim, 2).float()
        freqs = 1.0 / (theta ** (range_vec / dim))
        t = torch.arange(seq_len).float()
        freqs = torch.outer(t, freqs)
        self.freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        # freqs_cis: [seq, dim // 2]

    def forward(self, xq, xk):
        batch, seq_len, nh, dim = xq.shape

        xq_ = xq.float().reshape(batch, seq_len, nh, dim // 2, 2)
        xk_ = xk.float().reshape(batch, seq_len, nh, dim // 2, 2)

        xq_ = torch.view_as_complex(xq_)
        xk_ = torch.view_as_complex(xk_)

        xq_out = torch.view_as_real(xq_ * self.freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * self.freqs_cis).flatten(3)

        return xq_out.type_as(xq), xk_out.type_as(xk)