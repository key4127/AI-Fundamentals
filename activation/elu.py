import torch
import torch.nn as nn


class ELU(nn.Module):

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))


x = torch.tensor([2.5, -2.5])
alpha = 1.0
torch_elu = nn.ELU(alpha)
my_elu = ELU(alpha)

print(f"Input: \t\t{x}")
print(f"Pytorch ELU: \t{torch_elu(x)}")
print(f"My ELU: \t{my_elu(x)}")