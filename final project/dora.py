# Implementation of LoRA, rsLoRA and DoRA.
# Note that W ∈ R^d×k, (d, k) = (out_dim, in_dim) in LoRA paper, but (in_dim, out_dim) in DoRA paper,
# so m ∈ R^1×k should be (1, out_dim).
# Inspired by https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch


import math
import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Integrates LoRA and DoRA, separating the routes with and without dropout.
    """
    def __init__(self, linear, rank=8, alpha=16, dropout_p=0.0, use_dora=False, use_rslora=False):
        super().__init__()
        
        self.linear = linear
        # freeze original weight and bias
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # random Gaussian initialization    
        self.A = nn.Parameter(torch.randn(rank, linear.in_features) / math.sqrt(rank))  # (r, in)
        # zero initialization, so ∆W = BA starts at 0
        self.B = nn.Parameter(torch.zeros(linear.out_features, rank))  # (out, r)
        self.dropout = nn.Dropout(dropout_p) if dropout_p != 0.0 else None
        # rsLoRA for rank >= 64, seems not necessary for DoRA?
        self.scaling = alpha / math.sqrt(rank) if use_rslora else alpha / rank
        self.use_dora = use_dora
        
        if use_dora:
            self.m = nn.Parameter(linear.weight.norm(p=2, dim=1, keepdim=True))  # (out, in) -> (out, 1)
        
    def forward(self, x):
        # W_dora = m * (W0 + ∆W) / ‖W0 + ∆W‖, let S = m / ‖W0 + ∆W‖
        
        # lora = ∆W in LoRA paper or ∆V in DoRA
        lora = (self.B @ self.A) * self.scaling  # (out, r) @ (r, in) -> (out, in)
        W = self.linear.weight + lora  # (out, in) + (out, in)
        
        if self.use_dora:
            # detach the norm from the backpropagation to reduce memory overhead
            S = self.m / W.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8).detach()  # W.norm: (out, 1)
        else:
            # S = 1.0
            S = torch.ones(1, device=W.device, dtype=W.dtype)
        
        if self.dropout:
            # apply dropout to x of the adapter portion only, S(W0)x + S(∆W)dropout(x)
            out_base = F.linear(x, S*self.linear.weight, self.linear.bias) 
            # out_adapter = F.linear(self.dropout(x), S*lora, bias=None)
            # use LoRA's low-rank multiplication: S * (B @ (A @ dropout(x)))
            out_adapter = (self.dropout(x) @ self.A.T @ self.B.T) * self.scaling * S.squeeze(-1)
            return out_base + out_adapter
        else:
            # S(W0 + ∆W)x
            return F.linear(x, S*W, self.linear.bias)
    
    @torch.no_grad()
    def merge_weights(self):
        # cast to float64 to avoid rounding errors occurred during merging
        W0 = self.linear.weight.to(torch.float64)
        A = self.A.to(torch.float64)
        B = self.B.to(torch.float64)
        m = self.m.to(torch.float64)
        
        lora = (B @ A) * self.scaling
        W = W0 + lora
        
        if self.use_dora:
            W_norm = W.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            W_final = m * (W / W_norm)
        else:
            W_final = W
        
        self.linear.weight.data.copy_(W_final.data.to(self.linear.weight.dtype))


def apply_dora_to_all(model, rank=8, alpha=16, dropout_p=0.0, use_dora=True):
    """Recursively replaces all nn.Linear layers with LoRA/DoRA adapters."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, rank, alpha, dropout_p, use_dora))
        else:
            # recursively apply to submodules
            apply_dora_to_all(module, rank, alpha, dropout_p, use_dora)


def apply_dora_to_layer(model, name_list, rank=8, alpha=16, dropout_p=0.0, use_dora=True):
    """
    Recursively replaces assigned nn.Linear layers with LoRA/DoRA adapters.
    It's simple implementation; it cannot handle cases like 'intermediate.dense', but works for this project.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in name_list:
            setattr(model, name, LoRALinear(module, rank, alpha, dropout_p, use_dora))
        else:
            # recursively apply to submodules
            apply_dora_to_layer(module, name_list, rank, alpha, dropout_p, use_dora)


def merge_and_unload_all(model):
    """
    Iterates through the model to merge DoRA/LoRA weights into the base linear layers 
    and replaces LoRALinear instances with standard nn.Linear.
    """
    for name, module in model.named_children():
        if isinstance(module, LoRALinear):
            module.merge_weights()
            setattr(model, name, module.linear)
        else:
            merge_and_unload_all(module)
    
