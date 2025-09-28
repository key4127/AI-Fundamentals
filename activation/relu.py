import torch
import torch.nn as nn


class ReLU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(torch.tensor(0, dtype=torch.float32), x)


x = torch.tensor([2.5, -2.5])
torch_relu = nn.ReLU()
my_relu = ReLU()

print(f"Input1: \t{x}")
print(f"Pytorch ReLU: \t{torch_relu(x)}")
print(f"My ReLU: \t{my_relu(x)}")