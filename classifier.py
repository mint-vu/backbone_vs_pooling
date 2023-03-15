from torch import nn

class Classifier(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.din = d_in
        self.dout = d_out

        self.linear1 = nn.Linear(self.din, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, self.dout)
        self.act = nn.ReLU()
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.dp1(x)
        x = self.act(self.linear2(x))
        x = self.dp2(x)
        x = self.linear3(x)

        return x
