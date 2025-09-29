import torch
import torch.nn as nn
import torch.nn.functional as F


class KLLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def softmax(self, x):
        x_max = torch.max(x, dim=-1, keepdim=True)[0]
        x_exp = torch.exp(x - x_max)
        x_exp_sum = torch.sum(x_exp, dim=-1, keepdim=True)
        return x_exp / x_exp_sum

    def forward(self, x_pred, x_true):
        y_pred = self.softmax(x_pred)
        y_true = self.softmax(x_true)
        y_pred_log = torch.log(y_pred)
        y_true_log = torch.log(y_true)
        return torch.mean(torch.sum(y_true * (y_true_log - y_pred_log), dim=-1))


y_pred = torch.FloatTensor(
    [[1.0, 2.0, 1.5, 0.5],
    [2.0, 1.0, 0.5, 1.5],
    [0.5, 1.5, 2.0, 1.0]]
)

y_true = torch.FloatTensor(
    [[2.0, 1.0, 0.5, 1.5],
    [1.5, 2.0, 1.0, 0.5],
    [1.0, 0.5, 1.5, 2.0]]
)

torch_kl = nn.KLDivLoss(reduction='batchmean')
my_kl = KLLoss()

print(f"Pytorch KL: {torch_kl(F.log_softmax(y_pred, dim=1), F.softmax(y_true, dim=1))}")
print(f"My KL: {my_kl(y_pred, y_true)}")