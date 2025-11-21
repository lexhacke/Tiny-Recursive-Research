import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vision_model.trm_vit import trm_vit
from transformers import AutoImageProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as L

class trm_lightning(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.lr = config['learning_rate']
        self.CE = nn.CrossEntropyLoss()
        self.trm = trm_vit(config).to(config['device'])
        self.n = config['n']
        self.T = config['T']

    def forward(self, x, y=None, z=None):
        return self.trm(x, y, z, self.n, self.T, clip_graph=False)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _, _ = self(x)
        loss = self.CE(y_hat, y.cuda())
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

class imagenet(Dataset):
    def __init__(self, HW=224):
        self.HW = HW
        self.processor = AutoImageProcessor.from_pretrained("WinKawaks/vit-small-patch16-224", use_fast=True)
        self.data = load_dataset("timm/mini-imagenet", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = np.array(self.data[idx]['image'])
        if x.shape == (x.shape[0], x.shape[1], 3):
            return self.processor(x)['pixel_values'][0], torch.tensor(self.data[idx]['label']).long()
        else:
            return self.__getitem__(idx+1 % len(self.data))

if __name__ == "__main__":
    config = {
        'device':'cuda',
        'learning_rate':3e-4,
        'input_dim':192,
        'dim':384,
        'output_classes':100,
        'n':6,
        'T':4,
        'learnable_alpha':True,
        'attention':False,
        'residual_alpha':0,
        'vit_depth':12,
        'depth':1
    }

    trm = trm_lightning(config)
    ds = imagenet()
    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    trainer = L.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=5,
    )
    trainer.fit(trm, loader)
    torch.save(trm.trm.state_dict(), "TinyRecursiveViT.pt")