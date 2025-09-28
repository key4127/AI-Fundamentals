import torch
import torch.nn as nn


class Sigmoid(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1 / (torch.exp(-x) + 1)
    

x = torch.randn(1)
torch_sigmoid = nn.Sigmoid()
my_sigmoid = Sigmoid()

print(f"Input: \t\t{x}")
print(f"Pytorch Sigmoid:{torch_sigmoid(x)}")
print(f"My Sigmoid:\t{my_sigmoid(x)}")