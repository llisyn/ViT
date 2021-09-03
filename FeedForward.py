from torch import nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Subclassing nn.Sequential to avoid writing the forward method.
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion=4, drop_p=0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )
##
# check
x = torch.randn(16,1,128).to(device)
model = FeedForwardBlock(128).to(device)
output = model(x)
print(output.shape)

##
model

##

