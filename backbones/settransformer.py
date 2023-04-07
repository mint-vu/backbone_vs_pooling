from torch import nn
from attention_layers import SAB, ISAB

class SetTransformer(nn.Module):
    def __init__(self, d_in=None, d_out=None, type_="isab", num_hidden_layers=4, hidden_layer_size=512, num_heads=4, num_inds=16):
        super().__init__()
        
        layers = nn.ModuleList()

        layers.append(self.get_layer(type_, d_in, hidden_layer_size, num_heads, num_inds))
        for i in range(num_hidden_layers - 1):
            layers.append(self.get_layer(type_, hidden_layer_size, hidden_layer_size, num_heads, num_inds))
        layers.append(self.get_layer(type_, hidden_layer_size, d_out, num_heads, num_inds))

        self.net = nn.Sequential(*layers)

    def get_layer(self, type_, dim_in, dim_out, num_heads, num_inds=None):
        if type_ == "sab":
            return SAB(dim_in, dim_out, num_heads)
        elif type_ == "isab":
            return ISAB(dim_in, dim_out, num_heads, num_inds)
        else:
            raise ValueError(f'Layer type must be either "sab" or "isab", but got {type}!')

    def forward(self, x):
        return self.net(x)