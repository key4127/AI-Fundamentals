import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-5, affine=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("gamma", None)

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        norm = x / rms
        
        if self.affine:
            output = norm * self.gamma
        else:
            output = norm

        return output


batch_size = 2
seq_len = 3
hidden_size = 4
tensor = torch.randn(batch_size, seq_len, hidden_size)
my_LN = RMSNorm(hidden_size)
torch_LN = nn.RMSNorm(hidden_size, elementwise_affine=True)

print(f"Input:\n{tensor}\n")
print(f"My LN:\n{my_LN(tensor)}\n")
print(f"Pytorch LN\n: {torch_LN(tensor)}")