import torch
import torch.nn as nn


class RPE(nn.Module):

    def __init__(self, dim_in, k):
        super().__init__()
        self.dim_in = dim_in
        self.k = k

        vocab_size = 2 * k + 1

        self.relative_position_k = nn.Embedding(vocab_size, dim_in)
        self.relative_position_v = nn.Embedding(vocab_size, dim_in)

    def get_relative_position(self, seq_len):
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)

        distance_mat_clipped = torch.clamp(
            distance_mat, - self.k, self.k
        )

        final_mat = distance_mat_clipped + self.k
        return final_mat

    def forward(self, seq_len):
        relative_position = self.get_relative_position(seq_len)

        relative_k = self.relative_position_k(relative_position)
        relative_v = self.relative_position_v(relative_position)

        return relative_k, relative_v