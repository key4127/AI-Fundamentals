import torch
import torch.nn as nn


class LeakyReLU(nn.Module):

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * x)
    

x = torch.tensor([2.5, -2.5])
alpha = 0.01
torch_leakyrelu = nn.LeakyReLU(alpha)
my_leakyrelu = LeakyReLU(alpha)

print(f"Input: \t\t\t{x}")
print(f"Pytorch LeakyReLU: \t{torch_leakyrelu(x)}")
print(f"My LeakyReLU: \t\t{my_leakyrelu(x)}")