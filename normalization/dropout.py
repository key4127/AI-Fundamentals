import torch
import torch.nn as nn


class Dropout(nn.Module):

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (torch.rand(x.shape) > self.p).float()
            output = mask * x / (1 - self.p) # future does not change
        else:
            output = x

        return output


batch_size = 2
hidden_size = 6
dropout_ratio = 0.4
x = torch.ones(batch_size, hidden_size)
my_dropout = Dropout(p=dropout_ratio)
torch_dropout = nn.Dropout(p=dropout_ratio)

my_dropout.train()
torch_dropout.train()
print(f"Input:\n{x}\n")
print(f"My Dropout (training):\n{my_dropout(x)}\n")
print(f"Pytorch Dropout (training):\n{torch_dropout(x)}\n")

my_dropout.eval()
torch_dropout.eval()
my_dropout.eval()
torch_dropout.eval()
print(f"My Dropout (eval):\n{my_dropout(x)}\n")
print(f"Pytorch Dropout (eval):\n{torch_dropout(x)}")