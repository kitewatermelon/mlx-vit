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

class MHSA(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0, "embed_dim % num_heads != 0 !!!!" 

        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)


    def __call__(self, x):
        # BND -> BN(3*D)
        x = self.qkv(x)
        # Q, K, V 3개로 chunk -> [BND, BND, BND]
        qkv = mx.split(x, axis=2, indices_or_sections=3) 
        print(qkv[0].shape, qkv[1].shape, qkv[2].shape)
        
        # 1. Q,K,V 각각 BHND로 reshape (관례)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv) 
        # 2. Mat. Mul. -> BHN(Q)N(K)
        attn = q @ k.transpose(0, 1, 3, 2) 
        # 3. Scale: sqrt(dk)
        attn_score = attn * self.scale 
        # 4. attn_w: K에 대하여 softmax하기 위해서 axis=-1로 설정
        attn_weight = mx.softmax(attn_score, axis=-1) 
        # 5. Mat. Mul. -> BHND
        out = attn_weight @ v 
        # 6. 각 헤드 concat
        out = rearrange(out, 'b h n d -> b n (h d)', h = self.num_heads) # MHSA concat

        return out, attn_weight # 최종 출력과 attn_weight 같이 출력

