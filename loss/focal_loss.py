import torch
import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def forward(self, x_pred, y_true):
        y_pred = self.sigmoid(x_pred)
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        a_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        return - torch.mean(a_t * ((1 - p_t) ** self.gamma) * torch.log(p_t))
    

y_pred = torch.FloatTensor([
    [2.0],
    [-3.0],
    [0.1],
    [-0.1]
])

y_true = torch.FloatTensor([
    [1.0],
    [0.0],
    [1.0],
    [0.0]
])

focal_loss = FocalLoss(alpha=0.25, gamma=2)
print(f"Focal Loss: {focal_loss(y_pred, y_true)}")