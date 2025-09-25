import torch
import torch.nn as nn


class BatchNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.momentum=momentum
        self.affine=affine

        self.register_buffer('running_mean', torch.zeros(self.hidden_size))
        self.register_buffer('running_var', torch.ones(self.hidden_size))

        if affine:
            self.gamma = nn.Parameter(torch.ones(hidden_size))
            self.beta = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x):
        if self.training:
            mean = torch.mean(x, dim=0)
            var = torch.var(x, dim=0, unbiased=False)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            output = self.gamma * x_norm + self.beta
        else:
            output = x_norm

        return output


batch_size = 4
num_features = 3
x = torch.randn(batch_size, num_features)
my_BN = BatchNorm(num_features)
torch_BN = nn.BatchNorm1d(num_features)

print(f"Input:\n{x}\n")
print(f"My BN:\n{my_BN(x)}\n")
print(f"Pytorch BN:\n{torch_BN(x)}")