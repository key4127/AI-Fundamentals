import math
import torch
import torch.nn as nn

# x * phi(x)
# 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))

class Tanh(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        exp_x = torch.exp(x)
        exp_x_ne = torch.exp(-x)
        return (exp_x - exp_x_ne) / (exp_x + exp_x_ne)


class GeLU(nn.Module):

    def __init__(self):
        super().__init__()
        self.tanh = Tanh()

    def forward(self, x):
        tanh = self.tanh(math.sqrt(2 / torch.pi) * (x + 0.044715 * x ** 3))
        return 0.5 * x * (1 + tanh)
    

x = torch.randn(1)
torch_gelu = nn.GELU()
my_gelu = GeLU()

print(f"Input: \t\t{x}")
print(f"Pytorch GELU: \t{torch_gelu(x)}")
print(f"My GELU: \t{my_gelu(x)}")