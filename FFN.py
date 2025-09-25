import torch.nn as nn


class FFN(nn.Module):

    def __init__(self, dim_in, dim_hid, dim_out):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hid)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim_hid, dim_out)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out