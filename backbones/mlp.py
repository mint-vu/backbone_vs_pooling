from torch import nn


class MLP(nn.Module):
    def __init__(self, d_in=None, d_out=None, num_layers=2, hidden_layer_size=512, act_f=nn.LeakyReLU()):
        super().__init__()
        
        layers = nn.ModuleList()

        layers.append(nn.Sequential(nn.Linear(d_in, hidden_layer_size), act_f))
        for i in range(num_layers - 1):
            layers.append(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), act_f))
            
        layers.append(nn.Sequential(nn.Linear(hidden_layer_size, d_out)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
