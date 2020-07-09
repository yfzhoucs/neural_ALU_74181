import torch.nn as nn
import torch.nn.functional as F
import torch

class GateIn2Out1(nn.Module):
    def __init__(self):
        super(GateIn2Out1, self).__init__()
        self.linear1 = nn.Linear(2, 6)
        self.out = nn.Linear(6, 1)

    def forward(self, x1, x2):
        x1 = x1.float()
        x2 = x2.float()
        x = torch.cat([x1, x2], dim=1)

        x = F.sigmoid(self.linear1(x))
        y = F.sigmoid(self.out(x))
        
        return y


if __name__ == '__main__':
    gate = GateIn2Out1()
    x1 = torch.randint(0, 2, (1, 1))
    x2 = torch.randint(0, 2, (1, 1))
    y = gate(x1, x2)
    print(x1, x2)
    print(y)