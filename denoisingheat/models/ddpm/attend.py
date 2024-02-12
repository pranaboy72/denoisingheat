from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

class Attend(nn.Module):
    def __init__(
        self,
        dropout=0.,
        flash = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        self.flash = flash
        
    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device
        
        q, k, v = map(lambda t: t.contiguous(), (q,k,v))
        
        config = self.cuda_config if is_cuda else self.cpu_config
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )
        
        return out
    
    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device
        
        if self.flash:
            return self.flash_attn(q, k, v)
        
        scale = q.shape[-1] ** -0.5
        
        # similarity
        
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale
        
        # attention
        
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # aggregate values
        
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)
        
        return out