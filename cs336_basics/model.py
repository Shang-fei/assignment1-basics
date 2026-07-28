import torch 
import torch.nn as nn
from einops import rearrange, einsum

class Linear(nn.Module):

    def __init__(self, in_features, out_feature, device=None, dtype=None):
        super().__init__()
        kwargs = {"device":device, "dtype":dtype}
        self.in_features = in_features
        self.out_features = out_feature
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, **kwargs))

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        return einsum(x, self.weight, "... din, dout din -> ... dout")
