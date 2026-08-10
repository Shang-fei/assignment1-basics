import torch
import math
import numpy as np
import random

from torch import Tensor
from jaxtyping import Bool, Float, Int
from einops import einsum

def softmax(x:torch.Tensor, dim:int):
    max_val = x.max(dim=dim, keepdim=True).values
    x = x - max_val
    return x.exp() / x.exp().sum(dim=dim, keepdim=True)

def cross_entropy(inputs:torch.Tensor, targets:torch.Tensor):
    # result = -torch.log(torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1))).mean()

    inputs = inputs - inputs.max(dim=-1, keepdim=True).values
    gt_scores = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1))
    return -(gt_scores - torch.logsumexp(inputs, dim=-1, keepdim=True)).mean()

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it >= warmup_iters and it < cosine_cycle_iters:
        return min_learning_rate+0.5*(1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))*(max_learning_rate-min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    l2_norm = 0
    for parameter in parameters:
        if parameter.grad is not None:
            l2_norm += torch.sum(parameter.grad.data**2)
    l2_norm = l2_norm**0.5

    if l2_norm > max_l2_norm:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.data *= (max_l2_norm / (l2_norm + eps))
    return l2_norm

def get_batch(dataset, batch_size, context_length, device):
    inputs = np.zeros((batch_size, context_length))
    targets = np.zeros((batch_size, context_length))

    max_start = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    for i, start in enumerate(starts):
        chunk = dataset[start:start+context_length+1]
        inputs[i] = chunk[:-1]
        targets[i] = chunk[1:]

    inputs = torch.from_numpy(inputs).long().to(device)
    targets = torch.from_numpy(targets).long().to(device)
    return inputs, targets

def save_checkpoint(model, optimizer, iteration, out):
    output = {}
    output.update({"model":model.state_dict()})
    output.update({"optimizer":optimizer.state_dict()})
    output.update({"iteration": iteration})
    torch.save(output, out)

def load_checkpoint(src, model, optimizer):
    state = torch.load(src)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]








