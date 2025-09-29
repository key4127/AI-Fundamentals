import torch
import torch.nn as nn


class MAELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_true - y_pred))


y_true = torch.FloatTensor([1, 2, 3, 4, 5])
y_pred = torch.FloatTensor([1.1, 2.5, 3.5, 4.5, 5.5])

torch_mae = nn.L1Loss()
my_mae = MAELoss()

print(f"Pytorch MAE: {torch_mae(y_pred, y_true)}")
print(f"My MAE: {my_mae(y_pred, y_true)}")