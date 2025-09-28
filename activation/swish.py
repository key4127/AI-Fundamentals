import torch
import torch.nn as nn


class Swish(nn.Module):

    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x / (1 + torch.exp(- self.beta * x))
    

x = torch.randn(1)
beta = 1.0
torch_swish = nn.SiLU() # beta = 1
my_swish = Swish(beta)

print(f"Input: \t\t{x}")
print(f"Pytorch Swish: \t{torch_swish(x)}")
print(f"My Swish: \t{my_swish(x)}")