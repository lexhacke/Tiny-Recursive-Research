import torch
from einops import rearrange
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification

RMSNorm = lambda x : x / x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()

def build_padding_mask(x, L, context, mask_value=float('inf')):
    B, N, D = x.shape
    padding_mask = torch.full((len(x), context), mask_value)
    for i in range(B):
        padding_mask[i, :L[i]] = 0
        padding_mask[i, i:]
    return padding_mask


def trunc_normal_(shape, mean=0, std=1, upper=2, lower=-2, device="cpu"):
    x = torch.randn(shape, device=device)
    x.clamp_(lower, upper)
    x *= std / x.std(unbiased=False)
    return x

def preprocess(img):
    arr = np.array(img)
    arr = cv2.resize(arr, (512, 512))
    return torch.tensor(arr).permute(2, 0, 1).float() / 255.0

class ViT_Sequence(torch.nn.Module):
    def __init__(self, depth, preprocess=False):
        super().__init__()
        self.preprocess = preprocess
        self.depth = min(depth, 12)
        self.processor = AutoImageProcessor.from_pretrained("WinKawaks/vit-tiny-patch16-224", use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224").eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.processor(x, return_tensors="pt") if self.preprocess else x
            hidden = self.model.vit.embeddings(**x) if self.preprocess else self.model.vit.embeddings(x)
            for i in range(self.depth):
                hidden = self.model.vit.encoder.layer[i](hidden)
        return hidden

def generate_angles_2d(H,W,D, device='cpu', freq=None):
    """
    Generates a 3D frequency field for 2D Rotary Positional Embeddings.
    - H: Height of the feature map.
    - W: Width of the feature map.
    - D: Embedding Dimension (must be even).
    - freq: Optional precomputed frequency tensor for the embedding dimension.
    """
    assert D % 2 == 0, "Embedding Dimension must be even!"
    freq = torch.tensor([10000**(-2*i/D) for i in range(int(D/2))], device=device) if freq is None else freq
    pos = torch.outer(torch.linspace(-1, 1, steps=H, device=device),torch.linspace(-1, 1, steps=W, device=device))
    freq_tensor = torch.einsum("ij,k->ijk", pos, freq) # outer product
    return freq_tensor

def generate_angles_1d(N, D, device='cpu', freq=None):
    """
    1d variation of generate_angles_2d
    """
    assert D % 2 == 0, "Embedding Dimension must be even!"
    freq = torch.tensor([10000**(-2*i/D) for i in range(int(D/2))], device=device) if freq is None else freq
    pos = torch.linspace(-1, 1, steps=N, device=device)
    freq_tensor = torch.einsum("i,j->ij", pos, freq) # outer product
    return freq_tensor

def apply_angles_2d(x, f):
    """
    Applies the 2D Rotary Positional Embeddings to the input tensor.
    - x: Input tensor of shape (B, H, W, D)
    - f: Frequency tensor of shape (H, W, D/2)
    Rotates each pair of dimensions in the last dimension via orthogonal 2D matrix multiplication.
    """
    x_reshaped = rearrange(x, "B H W (D p) -> B H W D p", p=2)
    real = x_reshaped[..., 0]
    imag = x_reshaped[..., 1]
    cosines, sines = f.cos(), f.sin()
    # r , i -> rcos-isin , rsin icos
    rot_real = real * cosines - imag * sines
    rot_imag = real * sines + imag * cosines
    rot_full = torch.concat((rot_real.unsqueeze(-1), rot_imag.unsqueeze(-1)), dim=-1)
    return rearrange(rot_full, "B H W D p -> B H W (D p)", p=2)

def apply_angles_1d(x, f):
    """
    1d variation of apply_angles_2d
    """
    x_reshaped = rearrange(x, "... (D p) -> ... D p", p=2)
    real = x_reshaped[..., 0]
    imag = x_reshaped[..., 1]
    cosines, sines = f.cos(), f.sin()
    # r , i -> rcos-isin , rsin icos
    rot_real = real * cosines[:real.shape[-2], :] - imag * sines[:real.shape[-2], :]
    rot_imag = real * sines[:real.shape[-2], :] + imag * cosines[:real.shape[-2], :]
    rot_full = torch.concat((rot_real.unsqueeze(-1), rot_imag.unsqueeze(-1)), dim=-1)
    return rearrange(rot_full, "... D p -> ... (D p)", p=2)

# Sanity Check :)
if __name__ == "__main__":
    print(ViT_Sequence(12)(torch.randn(1,3,224,224)).shape)
    print(apply_angles_1d(torch.randn(1,4,43,768), generate_angles_1d(64,768)).shape)
    print(apply_angles_2d(torch.randn(1,64,64,768), generate_angles_2d(64,64,768)).shape)