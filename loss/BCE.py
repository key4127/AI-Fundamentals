import torch
import torch.nn as nn


class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def forward(self, x, y_true):
        y_sig = self.sigmoid(x)
        return - torch.mean(y_true * torch.log(y_sig) + (1 - y_true) * torch.log(1 - y_sig)) 
    

y_pred = torch.FloatTensor(
    [[1.2, -0.5, 2.0, 0.1],
    [0.1, 1.5, -0.5, 2.0],
    [2.0, 0.5, 0.9, -1.0]]
)
y_true = torch.FloatTensor(
    [[1, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 1, 0]]
)

torch_bce = nn.BCEWithLogitsLoss()
my_bce = BCELoss()
print(f"Pytorch BCE: {torch_bce(y_pred, y_true)}")
print(f"My BCE: {my_bce(y_pred, y_true)}")