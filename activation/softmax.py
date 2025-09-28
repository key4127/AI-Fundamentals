import torch
import torch.nn as nn


class Softmax(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # 0 represents value, 1 represents indice
        x_max = torch.max(x, self.dim, keepdim=True)[0]
        exp_x = torch.exp(x - x_max)
        exp_sum = torch.sum(exp_x, dim=self.dim, keepdim=True)
        return exp_x / exp_sum


x = torch.tensor([[-4, 5, 8], [2, 6, 10]], dtype=torch.float32)
torch_softmax = nn.Softmax(dim=1)
my_softmax = Softmax(dim=1)
print(f"Pytorch Softmax:\n{torch_softmax(x)}")
print(f"My Softmax:\n{my_softmax(x)}")