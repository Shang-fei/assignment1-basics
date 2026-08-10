import torch 
import math
import torch.nn as nn
from torch import Tensor
from einops import rearrange, einsum
from jaxtyping import Bool, Float, Int
from collections.abc import Callable, Iterable
from typing import Optional

from cs336_basics.utils import softmax

class Linear(nn.Module):

    def __init__(self, in_features, out_feature, device=None, dtype=None):
        super().__init__()
        kwargs = {"device":device, "dtype":dtype}
        self.in_features = in_features
        self.out_features = out_feature
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features, **kwargs))

        std = 2/(in_features+out_feature)
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x:Tensor)-> Tensor:
        return einsum(x, self.weight, "... din, dout din -> ... dout")

class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        kwargs = {"device":device, "dtype":dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), **kwargs))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids:Tensor) -> Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):

    def __init__(self, d_model:int, eps:float=1e-5, device=None, dtype=None):
        super().__init__()
        kwargs = {"device":device, "dtype":dtype}
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.ones(d_model, **kwargs))

    def forward(self, x:Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        result = x / (((torch.sum(x**2, dim=-1, keepdim=True) / self.d_model) + self.eps) ** 0.5) * self.weight
        return result.to(in_dtype)

class SwiGLu(nn.Module):

    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        kwargs = {"device":device, "dtype":dtype}
        self.d_model = d_model
        if d_ff is None:
            self.d_ff = ((self.d_model * 8 / 3 + 32) // 64) * 64
        else:
            self.d_ff = d_ff

        self.linear1 = Linear(d_model, d_ff, **kwargs)
        self.linear2 = Linear(d_ff, d_model, **kwargs)
        self.linear3 = Linear(d_model, d_ff, **kwargs)

    def silu(self, x:Tensor) -> Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x:Tensor) -> Tensor:
        return self.linear2(self.silu(self.linear1(x)) * self.linear3(x))
        
class RoPE(nn.Module):

    def __init__(self, d_k:int, theta:float, max_seq_len:int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32)/d_k))
        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)


    def forward(self, x:Tensor, token_positions:Tensor):
        sin = self.sin_cache[token_positions]
        cos = self.cos_cache[token_positions]

        x_even = x[...,::2]
        x_odd = x[...,1::2]

        out = torch.zeros_like(x)

        out[..., ::2] = cos * x_even - sin * x_odd
        out[..., 1::2] = sin * x_even + cos * x_odd

        return out


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:

    d_k = Q.shape[-1]
    attn = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / (d_k**0.5)

    if mask is not None:
        attn = attn.masked_fill(~mask, float("-inf"))
        
    attn = softmax(attn, dim=-1)
    output = einsum(attn, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return output

    
class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len=None,
        theta=None,
        device=None,
        dtype=None
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        super().__init__()
        kwargs = {"device":device, "dtype":dtype}
        self.num_heads = num_heads
        d_h = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, **kwargs)
        self.k_proj = Linear(d_model, d_model, **kwargs)
        self.v_proj = Linear(d_model, d_model, **kwargs)
        self.o_proj = Linear(d_model, d_model, **kwargs)

        if max_seq_len is not None:
            self.rope = RoPE(d_h, theta, max_seq_len, device=device)

    def forward(self, x:Tensor, token_positions=None, use_rope=True):
        seq_len = x.shape[-2]

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, "... seq (h d_h) -> ... h seq d_h", h=self.num_heads)
        K = rearrange(K, "... seq (h d_h) -> ... h seq d_h", h=self.num_heads)
        V = rearrange(V, "... seq (h d_h) -> ... h seq d_h", h=self.num_heads)

        if use_rope and token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_output = rearrange(attn_output, "... h seq d_h -> ... seq (h d_h)")

        output = self.o_proj(attn_output)
        return output

class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        super().__init__()
        kwargs = {"device": device, "dtype": dtype}
        self.mha = MultiHeadAttention(d_model, num_heads, max_seq_len, theta, **kwargs)
        self.ffn = SwiGLu(d_model, d_ff, **kwargs)
        self.rmsnorm1 = RMSNorm(d_model, **kwargs)
        self.rmsnorm2 = RMSNorm(d_model, **kwargs)

    def forward(self, x:Tensor):
        token_position = torch.arange(x.shape[-2])
        residual = x
        x = self.rmsnorm1(x)
        x = self.mha(x, token_position, use_rope=True)
        x = residual + x

        residual = x
        x = self.rmsnorm2(x)
        x = self.ffn(x)
        x = residual + x

        return x

class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        super().__init__()
        kwargs = {"device":device, "dtype":dtype}
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, **kwargs)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, **kwargs)
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(d_model, **kwargs)
        self.ffn = Linear(d_model, vocab_size, **kwargs)

    def forward(self, tokens:Tensor):
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.ffn(x)
        return x

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int = 256,
    ):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        outputs = x
        for _ in range(max_new_tokens):
            logits = self.forward(outputs)[:, -1]
            scores = softmax(logits/temperature, dim=-1)
            values, indices = scores.sort(dim=-1, descending=True)
            cum_values = torch.cumsum(values, dim=-1)
            idx = torch.nonzero(cum_values[0]>=top_k)[0].item()
            values[idx+1:] = float('-inf')
            sampled_token_id = indices[:, torch.multinomial(values, num_samples=1).item()].unsqueeze(0)
            outputs = torch.cat((outputs, sampled_token_id), dim=-1)
            if sampled_token_id == eos_token_id:
                break

        return  outputs

    @classmethod
    def from_file(cls, model_config, src):
        model = cls(**model_config)
        state_dict = torch.load(src, weights_only=True)['model']
        model.load_state_dict(state_dict)
        return model


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta1":betas[0], 'beta2':betas[1], "weight_decay":weight_decay, "eps":eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                m = state.get("m", 0)
                v = state.get('v', 0)

                grad = p.grad.data  # Get the gradient of loss with respect to p.

                state['m'] = beta1 * m + (1-beta1)*grad
                state['v'] = beta2 * v + (1-beta2)*grad**2
                state["t"] = t + 1  # Increment iteration number.

                p.data -= weight_decay * lr * p.data
                p.data -= lr * ((1 - beta2**state['t'])**0.5 / (1 - beta1**state['t'])) * (state['m'] / (state['v']**0.5 + eps))

        return loss