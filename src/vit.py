import mlx
import mlx.nn as nn
import mlx.core as mx
from einops import rearrange
# 모든 기준은 ViT-B/16 모델을 기준으로 작성

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size # 
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
                in_channels=3,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )

    def __call__(self, x):
        # B, C, H, W = x.shape
        print(f"Input Shape: {x.shape}")
        x = self.proj(x).flatten(1, 2) # 두번째 차원인 H/patch_size와 W/patch_size를 하나의 차원으로 합침
        return x
