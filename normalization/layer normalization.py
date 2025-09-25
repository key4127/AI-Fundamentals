import torch
import torch.nn as nn


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-5, affine=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(hidden_size))
            self.beta = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            output = self.gamma * norm + self.beta
        else:
            output = norm

        return output


batch_size = 2
seq_len = 3
hidden_size = 4
tensor = torch.randn(batch_size, seq_len, hidden_size)
my_LN = LayerNorm(hidden_size, affine=True)
torch_LN = nn.LayerNorm(hidden_size, elementwise_affine=True)

print(f"Input:\n{tensor}\n")
print(f"My LN:\n{my_LN(tensor)}\n")
print(f"Pytorch LN\n: {torch_LN(tensor)}")