import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self,d_in,d_out) -> None:
        super(Classifier, self).__init__()

        self.din=d_in
        self.dout=d_out
        self.linear1=nn.Linear(self.din, 64)
        self.linear2=nn.Linear(64,64)
        self.Linear3=nn.Linear(64,self.dout)
        self.act=nn.ReLU()
        self.dp1=nn.Dropout(0.5)
        self.dp2=nn.Dropout(0.5)

    def forward(self,x):
        x = self.act(self.linear1(x))
        x=self.dp1(x)
        x = self.act(self.linear2(x))
        x=self.dp2(x)
        x = self.Linear3(x)

        return x
