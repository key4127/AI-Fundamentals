import torch
import torch.nn as nn


class Swish(nn.Module):

    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x / (torch.exp(-x * self.beta) + 1)
    

class SwiGLU(nn.Module):

    def __init__(self, dim_in, hidden_size):
        super().__init__()
        self.swish = Swish()
        self.W = nn.Linear(dim_in, hidden_size, bias=True)
        self.V = nn.Linear(dim_in, hidden_size, bias=False)

    def forward(self, x):
        gate = self.swish(self.W(x))
        value = self.V(x)
        return gate * value
    

input_size = 4
hidden_size = 8
batch_size = 2
x = torch.randn(batch_size, input_size)
my_swiglu = SwiGLU(input_size, hidden_size)
print(my_swiglu(x))