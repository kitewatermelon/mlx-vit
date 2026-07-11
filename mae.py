import mlx
import mlx.nn as nn
import mlx.core as mx
from einops import rearrange

from vit import PatchEmbedding, Block
# 모든 기준은 ViT-B/16 모델을 기준으로 작성
# https://arxiv.org/pdf/2111.06377 만 참고하고 작성
# shuffle, unshuffle 만들기
# _mask 및 mask_token 만들기
# patch_norm - mean, std 만들기 

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
        
        self.num_patches = int((img_size // patch_size) ** 2)
        
        self.encoder_pos_embed = mx.random.normal((1, self.num_patches, embed_dim), scale=0.02)  # 학습 가능한 파라미터로 position embedding 학습
        self.decoder_pos_embed = mx.random.normal((1, self.num_patches, embed_dim), scale=0.02)  # 학습 가능한 파라미터로 position embedding 학습

        self.mask_token = mx.zeros((1, 1, embed_dim))  # 동작 확인 용 mask token
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
        
        # Decoder
        self.decoder_blocks = [
            Block(
                embed_dim=decoder_embed_dim, 
                num_heads=decoder_num_heads, 
                mlp_ratio=decoder_mlp_ratio, 
                dropout_rate=decoder_dropout_rate
                )  for _ in range(decoder_depth)
            ]

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * self.in_chans, bias=True) # decoder to patch

    def __call__(self, imgs):
        latent, restore_indices = self.forward_encoder(imgs, self.mask_ratio)
        return latent, restore_indices

        # pred = self.forward_decoder(latent, restore_indices)  # [N, L, p*p*3]
        # loss = self.forward_loss(imgs, pred, mask)
        # return loss, pred

    def forward_encoder(self, x, mask_ratio):
        # 1. patchfy
        x = self.patch_embed(x)
        x += self.encoder_pos_embed
        # 2. shuffle
        N, L, D = x.shape  # batch, length, dim
        indices = mx.random.permutation(mx.arange(x.shape[1])) # random permuation 순열 만들기
        restore_indices = mx.argsort(indices)
        x_shuffle = x[:,indices,:]
        
        # masking
        len_keep = int(L * (1 - mask_ratio))
        x_masked = x_shuffle[:, :len_keep, :]
    
        # foreard
        for b in self.encoder_blocks:
            x_masked = b(x_masked)
        return x_masked, restore_indices
    
    def forward_decoder(self, latent, restore_indices):
        pass
        
        # for b in self.decoder_blocks:
        #     x = b(latent)
        # return x
    
    def forward_loss(self, imgs, pred, mask):
        pass
    

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
    z, ids = mae(x)
    print(z.shape, ids.shape)