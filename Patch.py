import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from torchvision import utils


from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary
import numpy as np
import time
import copy
import random
import math

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##

class PatchEmbedding(nn.Module):
    def __init__(self, in_channel=3, patch_szie=16, emb_size=768, img_size=224):
        super().__init__()

        self.patch_size = patch_szie

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=emb_size,
                      kernel_size=self.patch_size, stride=self.patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.randn(1,1,emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_szie) **2 + 1, emb_size))

    def forward(self,x):

        b = x.shape[0]
        x = self.projection(x)

        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)

        x += self.positions

        return x
