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
        self.norm1 = nn.LayerNorm(dims=embed_dim) # 별도의 norm1
        self.norm2 = nn.LayerNorm(dims=embed_dim) # 별도의 norm1
        self.mhsa = MHSA(embed_dim=embed_dim, num_heads=num_heads)
        self.mlp = MLP(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate)
    
    def __call__(self, x):
        # 1단계 - attention: 뭐가 더 중요한지 확인
        x_norm = self.norm1(x) # Layer normalization 1
        x_attn, _ = self.mhsa(x_norm) # MHSA
        x = x_attn + x # Residual connection
        
        # 2단계 - MLP: 비선형성 증가
        x_norm = self.norm2(x) # Layer normalization 2
        x_mlp = self.mlp(x_norm) # 비선형성 증가를 위한 MLP
        x = x_mlp + x

        return x

class ViT(nn.Module):
    """
        cls token based ViT 
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4, dropout_rate=0.1, depth=12, num_classes=1000):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout_rate = dropout_rate
        self.depth = depth
        self.num_patches = int((img_size // patch_size) ** 2)
        
        self.norm = nn.LayerNorm(dims=embed_dim) # final norm


        # !주의! self.cls_token은 single batch 기준으로 만들어졌기 때문에 (0번째 차원이 1), _pos_embed 매서드에서 동적으로 배치 차원을 늘려줘야 함. 
        self.pos_embed = mx.random.normal((1, self.num_patches + 1, embed_dim), scale=0.02)  # 학습 가능한 파라미터로 position embedding 학습
        self.cls_token = mx.random.normal((1, 1, embed_dim), scale=0.02) # 학습 가능한 파라미터로 cls 토큰 학습 

        # FOR TEST 실제 실험 시 동작 변경!
        # self.cls_token = mx.zeros((1, 1, embed_dim))  # 동작 확인 용 cls token
        # self.pos_embed = mx.ones((1, self.num_patches + 1, embed_dim)) # 동작 확인 용 position embedding

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
        
        self.head = nn.Linear(embed_dim, num_classes)

    def __call__(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            # print(f"{i} 번째 layer")
        x = self.norm(x)
        cls = x[:, 0]       # CLS token만 추출
        return self.head(cls)

    def _pos_embed(self, x):
        B, N, C = x.shape
        # print(B, N, C)
        # broadcast_to 함수로 cls token B 만큼 복제함. 
        # numpy-like 이므로 자세한 동작은 다음 문서 참고. https://numpy.org/doc/2.2/reference/generated/numpy.broadcast_to.html
        cls = mx.broadcast_to(self.cls_token, (B, 1, C))   
        x = mx.concatenate((cls, x), axis=1) # [cls] token N차원의 제일 앞에 concat, axis=1로 해줘야 N+1 됨. (BNC)
        # print(x)
        # array([[[0, 0, 0, ..., 0, 0, 0], > zeros로 설정 해놓고 제대로 추가 됐는지 테스트 완료
        #         [-0.625268, 0.0861489, 0.387924, ..., 0.565768, 1.14554, 0.420791],
        #         [1.4328, 0.820314, -0.0266257, ..., 0.0697592, -1.02677, 0.830738],
        #         ...,
        #         [0.164638, 0.335743, 0.710777, ..., 0.172818, -0.326656, 0.0479117],
        #         [-0.466198, 0.0355091, -0.264295, ..., -0.378135, 0.381905, -0.481186],
        #         [0.474137, 1.21557, -0.281954, ..., 0.562486, -0.0671904, 0.0877942]]], dtype=float32)
        
        x += self.pos_embed # position 정보 postion wise 하게 추가
        # print(x)
        # array([[[1, 1, 1, ..., 1, 1, 1], > ones로 설정 해놓고 제대로 추가 됐는지 테스트 완료
        #         [0.374732, 1.08615, 1.38792, ..., 1.56577, 2.14554, 1.42079],
        #         [2.4328, 1.82031, 0.973374, ..., 1.06976, -0.0267704, 1.83074],
        #         ...,
        #         [1.16464, 1.33574, 1.71078, ..., 1.17282, 0.673344, 1.04791],
        #         [0.533802, 1.03551, 0.735705, ..., 0.621866, 1.38191, 0.518814],
        #         [1.47414, 2.21557, 0.718046, ..., 1.56249, 0.93281, 1.08779]]], dtype=float32)
        return x
    
    def get_params_info(self):
        params = self.trainable_parameters()
        flat = mlx.utils.tree_flatten(params)
        # flat = [("layer.weight", array), ("layer.bias", array), ...]
        
        total = sum(v.size for _, v in flat)  # 언패킹 필요
        print(f"Total trainable parameters: {total:,}")
        return total

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
    sample = mx.random.normal([32, 224, 224, 3]) # BHWC

    projector = PatchEmbedding()
    block = Block()

    patches = projector(sample)
    print(patches.shape)

    out = block(patches)
    print(out.shape)

    # Total trainable parameters: 86,613,736
    vit = get_vit_base()
    out = vit(sample)
    print(out.shape)
    vit.get_params_info()
    
    # Total trainable parameters: 22,073,704
    vit = get_vit_small()
    out = vit(sample)
    print(out.shape)
    vit.get_params_info()
    
    # Total trainable parameters: 5,728,936
    vit = get_vit_tiny()
    out = vit(sample)
    print(out.shape)
    vit.get_params_info()
    