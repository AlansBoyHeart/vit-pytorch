import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, transformer, pool = 'cls', channels = 3):
        #       image_size = 224,patch_size = 32,num_classes = 2, dim = 128,                  ,channels = 3,
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),  #patch_size=32
            #        ( 128 3  7*32  7*32 -> 128 49  32*32*3    )
            nn.Linear(patch_dim, dim), )
            #         (  3*32*32 , 128   )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))   #(1 , 50, 128)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))                     #(1,  1,  128)
        self.transformer = transformer

        self.pool = pool     #pool = "cls"
        self.to_latent = nn.Identity()   #identity模块不改变输入，直接return input

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)    #(128,  2)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)   #x = (128,49,128)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)   #(1,1,128)->(128,1,128)
        x = torch.cat((cls_tokens, x), dim=1)   #x = (128,50,128)
        x += self.pos_embedding[:, :(n + 1)]    #n=49
        x = self.transformer(x)                 #输入(128， 50 ，128)  输出（128，50，128）

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
