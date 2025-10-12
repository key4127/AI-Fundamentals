import torch
import torch.nn as nn


class APE(nn.Module):
    
    def __init__(self, dim_in, seq_len):
        super().__init__()
        pos = torch.arange(seq_len).unsqueeze(1)
        dim_pos = torch.arange(0, dim_in, 2).unsqueeze(0)
        div_term = 1 / torch.pow(10000, dim_pos / dim_in)
        angle = pos * div_term
        pe = torch.zeros(seq_len, dim_in)
        pe[:, 0::2] = torch.sin(angle)
        pe[:, 1::2] = torch.cos(angle)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x is [batch, seq, dim_in]
        """
        return self.pe[:x.size(1)]