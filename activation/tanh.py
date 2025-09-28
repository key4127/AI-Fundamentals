import torch
import torch.nn as nn


class Tanh(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        exp_x = torch.exp(x)
        exp_x_ne = torch.exp(-x)
        return (exp_x - exp_x_ne) / (exp_x + exp_x_ne)
    

x = torch.randn(1)
torch_tanh = nn.Tanh()
my_tanh = Tanh()

print(f"Input: \t\t{x}")
print(f"Pytorch Tanh: \t{torch_tanh(x)}")
print(f"My Tanh: \t{my_tanh(x)}")