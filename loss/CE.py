import torch
import torch.nn as nn


class CELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def softmax(self, x):
        x_max = torch.max(x, dim=-1, keepdim=True)[0]
        x_exp = torch.exp(x - x_max)
        exp_sum = torch.sum(x_exp, dim=-1, keepdim=True)
        return x_exp / exp_sum

    def forward(self, x, y_true):
        y_pred = self.softmax(x)
        log_y_pred = torch.log(y_pred)
        return - torch.mean(torch.sum(y_true * log_y_pred, dim=-1))


y_pred = torch.FloatTensor([[2.0, 1.0, 0.1], 
                           [0.1, 2.0, 1.0],
                           [0.1, 0.1, 2.0],
                           [0.2, 0.5, 3.0]])
y_true = torch.FloatTensor([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 1]])

torch_ce = nn.CrossEntropyLoss()
my_ce = CELoss()
y_true_indices = torch.argmax(y_true, dim=1)

print(f"Pytorch CE: {torch_ce(y_pred, y_true_indices)}")
print(f"My CE: {my_ce(y_pred, y_true)}")