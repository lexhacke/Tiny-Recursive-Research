import torch
from einops import rearrange
import torch.nn as nn
from utils import ViT_Sequence
import torch.nn.functional as F
from utils import trunc_normal_
from backbone import Attention, MLPSwiGLU

class trm_vit(nn.Module):
    def __init__(self, config):
        """
        'device':'cuda' or 'cpu'
        'learning_rate': float
        'input_dim': int
        'dim': int
        'output_classes': int
        'n': int
        'T': int
        'residual_alpha': float
        'learnable_alpha': bool
        'vit_depth': int
        'depth': int
        'attention': bool
        'clip_graph': bool
        """
        super().__init__()
        self.dim = config['dim']
        self.HW = 224 # standard for ViT
        self.a = nn.Parameter(config['residual_alpha'] * torch.ones(self.dim), requires_grad=(config['learnable_alpha']))
        self.patchifier = nn.Sequential(ViT_Sequence(config['vit_depth']).eval(), nn.Linear(config['input_dim'], self.dim))
        self.seq_len = 197 + 1 # for cls token once again
        self.backbone = []
        for _ in range(config['depth']):
            if config['attention']:
                self.backbone.append(Attention(self.seq_len, self.dim))
        self.backbone.append(MLPSwiGLU(self.seq_len, transpose=True))
        self.backbone.append(MLPSwiGLU(self.dim, transpose=False))
        self.backbone = nn.Sequential(*self.backbone)
        self.cls_token = nn.Parameter(trunc_normal_((1, 1, config['dim']), mean=0, std=1))
        self.lm_head = nn.Linear(config['dim'], config['output_classes'])
        self.device = config['device']

    def inner(self, x, y, z, n=6):
        for _ in range(n):
            z = self.a * z + self.backbone(x + y + z)
        y = self.a * y + self.backbone(y + z)
        return y, z

    def outer(self, x, y, z, n=6, T=3, clip_graph=False):
        if clip_graph:
            with torch.no_grad():
                for j in range(T-1):
                    y, z = self.inner(x, y, z, n)
        else:
            for j in range(T-1):
                y, z = self.inner(x, y, z, n)
            y, z = self.inner(x, y, z, n)
        return y, z

    def forward(self, x, y=None, z=None, n=6, T=3, clip_graph=False):
        """
        Expects x of shape B, 3, H, W (H=W=HW)
        """
        x = self.patchifier(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], 1, self.dim), x), dim=1)

        if y is None and z is None:
            y, z = trunc_normal_((2, self.seq_len, self.dim), mean=0, std=(1/self.seq_len)**0.5, upper=2, lower=-2, device=self.device).chunk(2, dim=0)
        y, z = self.outer(x, y, z, n, T, clip_graph)
        return self.lm_head(y[:, 0, :]), y.detach(), z.detach()


if __name__ == "__main__":
    trm = trm_vit({"input_dim":192, "dim":384, "residual_alpha":0, 'learnable_alpha':True, 'attention':False, 'vit_depth':4, 'depth':2, 'device':'cpu', 'output_classes':100})
    print(trm(torch.randn(16, 3, 224, 224))[0].shape)
    print(sum(param.numel() for _, param in trm.named_parameters()), "Total Parameters")
    print(sum(param.numel() for _, param in trm.named_parameters()) - sum(param.numel() for _, param in trm.patchifier.named_parameters()), "Total Parameters Without ViT")