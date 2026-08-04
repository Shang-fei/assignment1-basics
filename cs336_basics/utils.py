import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int
from einops import einsum

def softmax(x:torch.Tensor, dim:int):
    max_val = x.max(dim=dim, keepdim=True).values
    x = x - max_val
    return x.exp() / x.exp().sum(dim=dim, keepdim=True)
