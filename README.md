# Tiny Recursive Models (TRM)

Recursive transformer architectures that apply iterative refinement through nested loops over learned representations. Includes variants for both vision (TRM-ViT) and language modeling (TRM-LM).

## Architecture

### Core Recursive Structure
Both models share the same recursive computation pattern:
- **Inner loop (N iterations)**: `z = Backbone(x + y + z)`
- **Outer loop (T iterations)**: Repeatedly applies inner loop and updates `y`

### TRM-ViT (Vision)
The model combines a frozen ViT backbone with a lightweight recursive head:

1. **Patchifier**: Pretrained ViT (configurable depth) extracts patch embeddings from 224x224 images
2. **Backbone**: Attention layers + SwiGLU MLPs with RMSNorm
3. **Recursive Computation**: Nested inner/outer loops refine latent states `y` and `z`:
   - Inner loop (N iterations): `z = α*z + Backbone(x + y + z)`
   - Outer loop (T iterations): Repeatedly applies inner loop and updates `y`

### Key Components
- **RoPE Attention**: Rotary positional embeddings for sequence modeling
- **SwiGLU MLP**: Gated linear units with transposed variants for token/feature mixing
- **Learnable α**: Optional residual scaling parameter

## Usage

```python
config = {
    'device': 'cuda',
    'learning_rate': 3e-4,
    'input_dim': 192,
    'dim': 384,
    'output_classes': 100,
    'n': 6,              # inner loop iterations
    'T': 4,              # outer loop iterations
    'learnable_alpha': True,
    'attention': False,
    'residual_alpha': 0,
    'vit_depth': 12,
    'depth': 1
}

from trm_vit import trm_vit
model = trm_vit(config)
logits, y, z = model(images)  # images: B x 3 x 224 x 224
```

### TRM-LM (Language)

A recursive language model using the same iterative refinement:

1. **Embedding**: Token embeddings from vocabulary
2. **Backbone**: TransformerBackbone with attention + MLP
3. **Recursive Computation**: Same inner/outer loop structure with padding mask support
4. **LM Head**: Projects to vocabulary for next-token prediction

```python
config = {
    'device': 'cuda',
    'dim': 192,
    'context': 100,
    'vocab_size': 50257,  # GPT-2 tokenizer
    'n_heads': 8,
    'depth': 2,
    'n': 6,
    'T': 3,
}

from trm_lm import trm_lm
model = trm_lm(config)
logits, y, z = model(tokens, lengths)  # tokens: B x context, lengths: B
```

## Training

```bash
python train.py
```

Uses PyTorch Lightning with Mini-ImageNet (100 classes).

---

## Experimental Results (ImageNet-100)

### Best Result
| Metric | Value |
|--------|-------|
| Cross Entropy | **0.144** |
| Epochs | 5 |
| N, T | 6, 4 |
| Learning Rate | 3e-4 |

### Hyperparameter Search

**Base Model**: 50k images, 3125 gradient updates, n=6, T=3, lr=1e-3, depth=12, dim=512

| Experiment | Config Change | Train Loss (CE) |
|------------|---------------|-----------------|
| Base Model | - | 3.6 |
| Residual Backbone | `x + Backbone(x)` | 4.6 |
| **N=6, T=4** | - | **2.8** |
| N=8, T=4 | - | 4.3 |
| N=6, T=5 | - | 4.3 |
| dim=384 | - | 3.6 |
| dim=768 | - | 4.6 |

### Things to Explore
1. Embedding Dim (Base: 512)
2. Recursive Parameters (Base: n=6, T=3)
3. Learning Rate (Base: 1e-3)
4. ViT depth (Base: 12)

## Dependencies

- PyTorch
- PyTorch Lightning
- einops
- transformers
- datasets (HuggingFace)
