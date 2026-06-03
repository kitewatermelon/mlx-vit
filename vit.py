import mlx
import mlx.nn as nn
import mlx.core as mx
from einops import rearrange
# 모든 기준은 ViT-B/16 모델을 기준으로 작성

class PatchEmbedding(nn.Module):
    def __init__(self, is_rgb=True, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = 3 if is_rgb else 1
        self.proj = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size
            )

    def __call__(self, x):
        # B, C, H, W = x.shape
        # print(f"Input Shape: {x.shape}")
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

        self.proj = nn.Linear(embed_dim, embed_dim)


    def __call__(self, x):
        # BND -> BN(3*D)
        x = self.qkv(x)
        # Q, K, V 3개로 chunk -> [BND, BND, BND]
        qkv = mx.split(x, axis=2, indices_or_sections=3) 
        # print(qkv[0].shape, qkv[1].shape, qkv[2].shape)
        
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
        # 7. concat 후 정보 섞어주기 위해 같은 차원으로 projection
        out = self.proj(out)
        return out, attn_weight # 최종 출력과 attn_weight 같이 출력

class MLP(nn.Module):
    def __init__(self, embed_dim=768, mlp_ratio=4, dropout_rate=0.1):
        super().__init__() 
        # 아래 timm-like MLP 참조함
        # https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py
        self.net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * mlp_ratio),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.LayerNorm(embed_dim * mlp_ratio),
                nn.Linear(embed_dim * mlp_ratio, embed_dim ),
                nn.Dropout(dropout_rate),
            )
        
    def __call__(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4, dropout_rate=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dims=embed_dim)
        self.mhsa = MHSA(embed_dim=embed_dim, num_heads=num_heads)
        self.mlp = MLP(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate)
    
    def __call__(self, x):
        # 1단계 - attention: 뭐가 더 중요한지 확인
        x_norm = self.norm(x) # Layer normalization
        x_attn, _ = self.mhsa(x_norm) # MHSA
        x = x_attn + x # Residual connection
        
        # 2단계 - MLP: 비선형성 증가
        x_norm = self.norm(x) # Layer normalization
        x_mlp = self.mlp(x_norm) # 비선형성 증가를 위한 MLP
        x = x_mlp + x

        return x

class ViT(nn.Module):
    """
        cls token based ViT 
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4, dropout_rate=0.1, depth=12):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.depth = depth
        self.num_patches = int((img_size // patch_size) ** 2)

        self.pos_embed = mx.zeros((1, self.num_patches + 1, embed_dim))  # 학습 가능한 파라미터로 position embedding 학습
        self.cls_token = mx.zeros((1, 1, embed_dim)) # 학습 가능한 파라미터로 cls 토큰 학습 

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size, 
            embed_dim=embed_dim
            )
        
        self.blocks = [
            Block(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                dropout_rate=dropout_rate
                )  for _ in range(depth)
            ]

    def __call__(self, x):
        x = self.patch_embed(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            print(f"{i} 번째 layer")
        return x

    def _pos_embed(self, x):

        return x


def get_vit_base():
    return ViT(patch_size=16, embed_dim=768, 
               num_heads=12, mlp_ratio=4, 
               dropout_rate=0.1, depth=12)

def get_vit_small():
    return ViT(patch_size=16, embed_dim=384, 
               num_heads=6, mlp_ratio=4, 
               dropout_rate=0.1, depth=12)

def get_vit_tiny():
    return ViT(patch_size=16, embed_dim=192, 
               num_heads=3, mlp_ratio=4, 
               dropout_rate=0.1, depth=12)

if __name__=="__main__":

    # tests 
    sample = mx.random.normal([1, 224, 224, 3]) # BHWC

    projector = PatchEmbedding()
    block = Block()

    patches = projector(sample)
    print(patches.shape)

    out = block(patches)
    print(out.shape)

    vit = get_vit_base()
    out = vit(sample)
    print(out.shape)

    vit = get_vit_small()
    out = vit(sample)
    print(out.shape)

    vit = get_vit_tiny()
    out = vit(sample)
    print(out.shape)