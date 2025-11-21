import torch
from transformers import AutoTokenizer
import torch.nn as nn
from backbone import TransformerBackbone
from utils import trunc_normal_, build_padding_mask

class trm_lm(nn.Module):
    def __init__(self, config):
        """
        'device':'cuda' or 'cpu'
        'vocab_size': int
        'n_heads': int
        'learning_rate': float
        'dim': int
        'output_classes': int
        'n': int
        'T': int
        'depth': int
        'clip_graph': bool
        'context': int
        """
        super().__init__()
        self.dim = config['dim']
        self.context = config['context']
        self.embedding = nn.Embedding(config['vocab_size'], config['dim'])
        self.backbone = TransformerBackbone(config)
        self.lm_head = nn.Linear(config['dim'], config['vocab_size'])
        self.device = config['device']

    def inner(self, x, y, z, L, mask, n=6):
        for _ in range(n):
            z = self.backbone(x + y + z, L) * mask
        y = self.backbone(y + z, L) * mask
        return y, z

    def outer(self, x, y, z, L, mask, n=6, T=3, clip_graph=False):
        if clip_graph:
            with torch.no_grad():
                for j in range(T-1):
                    y, z = self.inner(x, y, z, L, mask, n=n)
        else:
            for j in range(T-1):
                y, z = self.inner(x, y, z, L, mask, n=n)
        y, z = self.inner(x, y, z, L, mask, n=n)
        return y, z

    def forward(self, x, L, y=None, z=None, n=6, T=3, clip_graph=False):
        """
        Expects x of shape B, context
        """
        B, _ = x.shape
        x = self.embedding(x)
        if y is None and z is None:
            y, z = trunc_normal_((2, self.context, self.dim), mean=0, std=(1/self.context)**0.5, upper=2, lower=-2, device=self.device).chunk(2, dim=0)
            mask = build_padding_mask(y, L, self.context, mask_value=1).unsqueeze(-1)
            y = y * mask
            z = y * mask
        y, z = self.outer(x, y, z, L, mask, n=n, T=T, clip_graph=clip_graph)
        return self.lm_head(y[range(B), L-1, :]), y.detach(), z.detach()

if __name__ == "__main__":
    config = {
        "dim":192,
        "context":100,
        "vocab_size":50257,
        "n_heads":8,
        'attention':False,
        'depth':2,
        'device':'cpu'
        }

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
    padded_tokens = tokenizer("You currently have zero compute units available.")['input_ids']
    padded_tokens = padded_tokens + [0 for _ in range(100 - len(padded_tokens))]
    slm = trm_lm(config)
    print(slm(torch.tensor([padded_tokens for _ in range(16)]), torch.tensor([10 for _ in range(16)]))[0].shape)
    print(sum(param.numel() for _, param in slm.named_parameters()), "Total Parameters")