import mlx
import mlx.nn as nn
import mlx.core as mx
from einops import rearrange

from vit import PatchEmbedding, Block
# 모든 기준은 ViT-B/16 모델을 기준으로 작성
# https://arxiv.org/pdf/2111.06377 만 참고하고 작성

from util.pos_embed import get_2d_sincos_pos_embed

class MAE(nn.Module):
    def __init__(self, 
                img_size=224, 
                patch_size=16, 
                mask_ratio=0.75,
                is_rgb=True,     
                       
                # encoder
                embed_dim=768, 
                num_heads=12, 
                mlp_ratio=4, 
                dropout_rate=0.1, 
                depth=12, 
                
                # decoder
                decoder_embed_dim=512, 
                decoder_num_heads=8, 
                decoder_mlp_ratio=4, 
                decoder_dropout_rate=0.1, 
                decoder_depth=8, 
                ):
        super().__init__()
        # Common
        if is_rgb: 
            self.in_chans = 3
        else: 
            self.in_chans = 1
        self.patch_size = patch_size
        self.num_patches = int((img_size // patch_size) ** 2)
        
        self.encoder_pos_embed = mx.random.normal((1, self.num_patches, embed_dim), scale=0.02)  # 학습 가능한 파라미터로 position embedding 학습
        self.decoder_pos_embed = mx.random.normal((1, self.num_patches, decoder_embed_dim), scale=0.02)  # 학습 가능한 파라미터로 position embedding 학습

        self.mask_token = mx.zeros((1, 1, decoder_embed_dim))  # 동작 확인 용 mask token
        # self.mask_token = mx.random.normal((1, self.num_patches, embed_dim), scale=0.02)  # 학습 가능한 파라미터로 position embedding 학습

        self.patch_embed = PatchEmbedding(
            patch_size=patch_size, 
            embed_dim=embed_dim,
            is_rgb=is_rgb
            )

        self.mask_ratio = mask_ratio
        
        # Encoder
        self.encoder_blocks = [
            Block(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                dropout_rate=dropout_rate
                )  for _ in range(depth)
            ]
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # Decoder
        self.decoder_blocks = [
            Block(
                embed_dim=decoder_embed_dim, 
                num_heads=decoder_num_heads, 
                mlp_ratio=decoder_mlp_ratio, 
                dropout_rate=decoder_dropout_rate
                )  for _ in range(decoder_depth)
            ]
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * self.in_chans, bias=True) # decoder to patch
        self.initialize_weights()


    # init은 claude가 도와줌 ㅎㅎ
    def initialize_weights(self):
        # sincos pos embed
        pos_embed = get_2d_sincos_pos_embed(self.encoder_pos_embed.shape[-1], int(self.num_patches**0.5))
        self.encoder_pos_embed = mx.array(pos_embed).reshape(1, self.num_patches, -1)

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**0.5))
        self.decoder_pos_embed = mx.array(decoder_pos_embed).reshape(1, self.num_patches, -1)

        # mask_token
        self.mask_token = mx.random.normal((1, 1, self.decoder_embed_dim), scale=0.02)
        # Linear, LayerNorm 초기화
        def init_weights(module):
            if isinstance(module, nn.Linear):
                module.weight = nn.init.glorot_uniform()(module.weight)
                if module.bias is not None:
                    module.bias = mx.zeros_like(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.bias = mx.zeros_like(module.bias)
                module.weight = mx.ones_like(module.weight)
            return module  # 추가
        self.apply(init_weights)
        
    def __call__(self, imgs):
        latent, restore_indices, mask = self.forward_encoder(imgs, self.mask_ratio)
        pred = self.forward_decoder(latent, restore_indices)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred

    def forward_encoder(self, x, mask_ratio):
        # 1. patchfy
        x = self.patch_embed(x)
        x += self.encoder_pos_embed
        # 2. shuffle
        N, L, D = x.shape  # batch, length, dim 
        
        indices = mx.stack([mx.random.permutation(mx.arange(L)) for _ in range(N)]) # L개의 random index를 N개 쌓기
        restore_indices = mx.argsort(indices, axis=1) # argsort()로 원래 index 번호 얻어오기
        x_shuffle = mx.vmap(lambda x_i, idx_i: x_i[idx_i])(x, indices)  # (N, L, D)
        
        # masking
        len_keep = int(L * (1 - mask_ratio))
        x_masked = x_shuffle[:, :len_keep, :]
        
        # shuffle 순서로 mask 만들기 (앞 len_keep = 0, 뒤 masked = 1)
        mask = mx.ones((N, L))
        mask[:, :len_keep] = 0  # kept 부분은 0        
        # restore_indices로 원본 순서로 되돌리기
        mask = mx.vmap(lambda m, r: m[r])(mask, restore_indices)        

        # foreard
        for b in self.encoder_blocks:
            x_masked = b(x_masked)
            
        return self.encoder_norm(x_masked), restore_indices, mask
    
    def forward_decoder(self, x, restore_indices):
        # mask token 갖다 붙이기
        x = self.decoder_embed(x)
        N, _, D = x.shape
        len_mask = int(self.num_patches * self.mask_ratio)
        mask_token = mx.broadcast_to(self.mask_token, (N, len_mask, D))
        x = mx.concatenate([x, mask_token], axis=1)
        
        # unshuffle
        x_shuffle = mx.vmap(lambda x_i, r: x_i[r])(x, restore_indices)  # (N, L, D)
        x_shuffle += self.decoder_pos_embed

        for b in self.decoder_blocks:
            x_shuffle = b(x_shuffle)
        x = self.decoder_norm(x_shuffle)

        return self.decoder_pred(x)

    def forward_loss(self, imgs, pred, mask):
        # 패치 공간에서 loss 계산하기
        imgs_p = self.patchfy(imgs)
        
        # patch 별로 normalize
        mean = imgs_p.mean(axis=-1, keepdims=True)
        var = imgs_p.var(axis=-1, keepdims=True)
        imgs_p = (imgs_p - mean) / (var + 1.e-6)**.5        
        
        # Loss 계산
        loss = (pred - imgs_p) ** 2  # MSE
        loss = loss.mean(axis=-1) # D차원 평균 -> 패치별 스칼라 loss, (N, L)
        loss = (loss * mask).sum() / mask.sum() # mask에 대해서만 loss

        return loss

    # einops로 N H W C -> N L D로 만들기 (patch 공간으로 보내서 loss 계산하기)
    def patchfy(self, x): 
        return rearrange(
            x, 
            "b (h ph) (w pw) c -> b (h w) (ph pw c)", 
            h=int(self.num_patches**0.5), w=int(self.num_patches**0.5), ph=self.patch_size, pw=self.patch_size, c=self.in_chans
        )

    # einops로  N L D -> N H W C 로 만들기 (pixel 공간으로 보내서 시각화할 때 쓰거나 하기)
    def unpatchfy(self, z):
        return rearrange(
            z, 
            "b (h w) (ph pw c) -> b (h ph) (w pw) c", 
            h=int(self.num_patches**0.5), w=int(self.num_patches**0.5), ph=self.patch_size, pw=self.patch_size, c=self.in_chans
        )
        
    def get_params_info(self):
        params = self.trainable_parameters()
        flat = mlx.utils.tree_flatten(params)
        # flat = [("layer.weight", array), ("layer.bias", array), ...]
        
        total = sum(v.size for _, v in flat)  # 언패킹 필요
        print(f"Total trainable parameters: {total:,}")
        return total

def mae_base():
    return MAE(                
        img_size=224, patch_size=16, mask_ratio=0.75, is_rgb=True,                    
        # encoder
        embed_dim=768, num_heads=12, mlp_ratio=4, dropout_rate=0.1, depth=12, 
        # decoder
        decoder_embed_dim=512, decoder_num_heads=8, decoder_mlp_ratio=4, decoder_dropout_rate=0.1, decoder_depth=8, 
        )
    
def mae_small():
    return MAE(                
        img_size=224, patch_size=16, mask_ratio=0.75, is_rgb=True,                    
        # encoder
        embed_dim=384, num_heads=6, mlp_ratio=4, dropout_rate=0.1, depth=12, 
        # decoder
        decoder_embed_dim=256, decoder_num_heads=4, decoder_mlp_ratio=4, decoder_dropout_rate=0.1, decoder_depth=6, 
        )

def mae_tiny():
    return MAE(                
        img_size=224, patch_size=16, mask_ratio=0.75, is_rgb=True,                    
        # encoder
        embed_dim=192, num_heads=3, mlp_ratio=4, dropout_rate=0.1, depth=12, 
        # decoder
        decoder_embed_dim=128, decoder_num_heads=2, decoder_mlp_ratio=4, decoder_dropout_rate=0.1, decoder_depth=4, 
        )

if __name__=="__main__":
    x = mx.random.normal([32, 224, 224, 3])
    mae = mae_tiny()
    mae.get_params_info()
    loss, pred = mae(x)
    x_pred = mae.unpatchfy(pred)
