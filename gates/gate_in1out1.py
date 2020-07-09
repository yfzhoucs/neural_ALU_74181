import torch.nn as nn
import torch.nn.functional as F

class GateIn1Out1(nn.Module):
    def __init__(self):
        super(GateIn1Out1, self).__init__()
        self.linear1 = nn.Linear(1, 3)
        self.out = nn.Linear(3, 1)

    def forward(self, x):
        x = x.float()

        x = F.sigmoid(self.linear1(x))
        y = F.sigmoid(self.out(x))
        
        return y


if __name__ == '__main__':
    import torch
    gate = GateIn1Out1()
    x = torch.randint(0, 2, (1, 1))
    y = gate(x)
    print(x)
    print(y)