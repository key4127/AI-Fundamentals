import torch
import torch.nn as nn


class MSELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.pow(y_pred - y_true, 2))
    

y_true = torch.FloatTensor([1, 2, 3, 4, 5])
y_pred = torch.FloatTensor([1.1, 2.5, 3.5, 4.5, 5.5])

torch_mse = nn.MSELoss()
my_mse = MSELoss()

print(f"Pytorch MSE: {torch_mse(y_pred, y_true)}")
print(f"My MSE: {my_mse(y_pred, y_true)}")