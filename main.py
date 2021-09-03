## 필요 라이브러리 불러오기

import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader

from Patch import *
from Multihead import *
from FeedForward import *
##
# specify path to data
data_dir = './datasets'

# if not exists the path, make the directory
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

train_ds = datasets.STL10(data_dir, split='train', download=True, transform=transforms.ToTensor())
val_ds = datasets.STL10(data_dir, split='test', download=True, transform=transforms.ToTensor())

##이미지 확인하기
print(len(train_ds))
print(len(val_ds))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##
# define transformation
transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(224)
])

# apply transformation to dataset
train_ds.transform = transformation
val_ds.transform = transformation

# make dataloade
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=True)

## 2. MultiHead Attention


##
# Now create the Transformer Encoder Block
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size=768, drop_p=0., forward_expansion=4, forward_drop_p=0., **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )


##
# TransformerEncoder consists of L blocks of TransformerBlock
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth=12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

##
# define ClassificationHead which gives the class probability
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size=768, n_classes = 10):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))

##
# Define the ViT architecture
class ViT(nn.Sequential):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224, depth=12, n_classes=10, **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )

##
model = ViT().to(device)
summary(model, (3,224,224), device=device.type)

##

