import torch
from einops import rearrange
import torch.nn as nn
from utils import apply_angles_1d, generate_angles_1d, RMSNorm
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, context_length, emb_dim, n_heads=8):
        super().__init__()
        self.context_length = context_length
        self.n_heads = n_heads
        head_dim = emb_dim // n_heads
        self.qkv = nn.Linear(emb_dim, 3*emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.register_buffer("freq", generate_angles_1d(context_length, head_dim), persistent=False)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B N (h D) -> B h N D", h=self.n_heads)
        k = rearrange(k, "B N (h D) -> B h N D", h=self.n_heads)
        v = rearrange(v, "B N (h D) -> B h N D", h=self.n_heads)

        q = apply_angles_1d(q, self.freq)
        k = apply_angles_1d(k, self.freq)

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "B h N D -> B N (h D)")
        x = self.proj(x)
        return x

class MLPSwiGLU(nn.Module):
    def __init__(self, dim: int, upsample=2, transpose=False):
        """
        dim = embedding dimension
        tokens = number of tokens per embedding
        """
        super().__init__()
        self.transpose = transpose
        self.dim = dim
        self.linearIn = nn.Linear(dim, upsample*dim, bias=True)
        self.gate = nn.Linear(dim, upsample*dim, bias=True)
        self.linearOut = nn.Linear(upsample*dim, dim, bias=True)

    def forward(self, x: torch.Tensor):
        """
        Requires input to be B N D where N=tokens
        Outputs a singleton for x[-1] (z) of shape B 1 D
        Transposes by N, D axis to create a per-feature affine transform
        """
        x = rearrange(x, "B N D -> B D N") if self.transpose else x # batch of token vectors to batch of per-token feature vectors
        x = self.linearOut(F.silu(self.linearIn(x)) * self.gate(x))
        x = rearrange(x, "B D N -> B N D") if self.transpose else x # recover x,y,z
        return RMSNorm(x)