import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
import numpy as np
from einops import rearrange
from typing import Final, Optional, Type
from einops_exts import check_shape, rearrange_many
from timm.models.vision_transformer import Attention
from timm.models.layers import to_2tuple
from timm.layers.config import use_fused_attn
from timm.layers._fx import register_notrace_function

def is_odd(n):
    return (n % 2) == 1

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    # print('grid_size:', grid_size)

    grid_x = np.arange(grid_size[0], dtype=np.float32)
    grid_y = np.arange(grid_size[1], dtype=np.float32)
    grid_z = np.arange(grid_size[2], dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')  # here y goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[0], grid_size[1], grid_size[2]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    # assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (X*Y*Z, D/3)
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (X*Y*Z, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (X*Y*Z, D/3)

    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1) # (X*Y*Z, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class PatchEmbed_Voxel(nn.Module):
    """ Voxel to Patch Embedding
    """
    def __init__(self, voxel_size=(16,16,16,), patch_size=2, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (voxel_size[0] // patch_size[0]) * (voxel_size[1] // patch_size[1]) * (voxel_size[2] // patch_size[2])
        self.patch_xyz = (voxel_size[0] // patch_size[0], voxel_size[1] // patch_size[1], voxel_size[2] // patch_size[2])
        self.voxel_size = voxel_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        x = x.float()
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous() # (B*4*3*4, 8, 8, 8, 8) ->proj-> (B*4*3*4, 72, 8, 8, 8) ->flatten-> (B*4*3*4, 72, 512) ->transpose-> (B*4*3*4, 512, 72) 
        return x



# ====================== DiT ====================== 
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0., eta=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        if eta is not None: # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(hidden_features), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(out_features), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

    def forward(self, x):
        x = self.gamma1 * self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gamma2 * self.fc2(x)
        x = self.drop2(x)
        return x


@torch.fx.wrap
@register_notrace_function
def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    return scores if attn_mask is None else scores + attn_mask

class CrossAttention(nn.Module):
    """Multi-head Cross Attention module with QKV projection.
    This module is implemented referred to Attention.
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            context_dim: int = 768, 
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        # Separate projections for query (from source) and key-value (from context)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)

        context_dim = context_dim or dim
        self.kv_proj = nn.Linear(context_dim, dim * 2, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of cross attention.

        Args:
            x: Query tensor of shape (B, N, C) from source sequence
            y: Key/Value tensor of shape (B, M, C) from context sequence  
            attn_mask: Optional attention mask for cross-attention

        Returns:
            Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        M = y.shape[1]
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)   # (B,N,C) -> (B,N,num_heads,head_dim) -> (B,num_heads,N,head_dim)
        kv = self.kv_proj(y).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (B,M,C) -> (B,M,2C) -> (B,N,2,num_heads,head_dim) -> (2,B,num_heads,N,head_dim)
        k, v = kv.unbind(0)  # (B, num_heads, M, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)  # nn.Identity()

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # (B, num_heads, N, M)
            attn = maybe_add_mask(attn, attn_mask)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v  # (B, num_heads, N, head_dim)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)  # nn.Identity()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, context_dim=768, mlp_ratio=4.0, skip=False, **block_kwargs):
        super().__init__()
        self.skip_linear = nn.Linear(2*hidden_size, hidden_size) if skip else None
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, context_dim=context_dim, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(4 * hidden_size, 6 * hidden_size, bias=True),
            # nn.Linear(4 * hidden_size*2, 6 * hidden_size, bias=True)   # 有resolution condition的时候才要乘2
        )

    def forward(self, x, t, y, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim = -1))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.cross_attn(x, y)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT block.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * patch_size * out_channels, bias=True) # hidden_size=72, patch_size=1, out_channels=72
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(4*hidden_size, 2 * hidden_size, bias=True)
            # nn.Linear(4*hidden_size*2, 2 * hidden_size, bias=True) # 有resolution condition的时候才要乘2
        )

    # def forward(self, x, t, **other_input):
    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x) # just a linear layer which not changing dimensions
        return x

# ====================== DiT ====================== 


# ====================== UNet ====================== 
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=6):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (3, 3, 3), padding=(1, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads # 8*32=256
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, z, h, w = x.shape
        x = rearrange(x,'b c z x y -> b (z x y) c').contiguous()
        qkv = self.to_qkv(x).chunk(3, dim=2)
        q, k, v = rearrange_many(
            qkv, 'b d (h c) -> b h d c ', h=self.heads)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=0.0, is_causal=False)
        out = rearrange(out, 'b h (z x y) c -> b (h c) z x y ',z = z, x = h ,y = w ).contiguous()
        out = self.to_out(out)
        return out

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, context_dim=768):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads # 8*32=256
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        context_dim = context_dim or dim
        self.to_kv = nn.Linear(context_dim, hidden_dim * 2, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x, y):
        b, c, z, h, w = x.shape
        x = rearrange(x, 'b c z x y -> b (z x y) c').contiguous()
        q = self.to_q(x)
        k, v = self.to_kv(y).chunk(2, dim=2)   # y: (B, L, D), where D != c
        q, k, v = rearrange_many(
            (q, k, v), 'b d (h c) -> b h d c', h=self.heads)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale, dropout_p=0.0, is_causal=False)
        out = rearrange(out, 'b h (z x y) c -> b (h c) z x y', z=z, x=h, y=w).contiguous()
        out = self.to_out(out)
        return out

class IdentityWithMultiInputs(nn.Module):
    def forward(self, x, *y, **z):
        return x

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (4, 4, 4), (2, 2, 2), (1, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (4, 4, 4), (2, 2, 2), (1, 1, 1))

# ====================== UNet ====================== 





class BiFlowNet(nn.Module):
    def __init__(
        self,
        dim,
        learn_sigma=False,
        prompt_dim=768,
        dim_mults=(1,1,2,4,8),
        sub_volume_size = (8,8,8),
        patch_size=2,
        channels=3,
        attn_heads=8, #
        init_dim=None,
        init_kernel_size=3,
        use_sparse_linear_attn=[0,0,0,1,1],
        resnet_groups=24, #
        DiT_num_heads = 8, #
        mlp_ratio=4,
        vq_size=64,
        res_condition=True,
        num_mid_DiT=1
    ):
        self.res_condition=res_condition

        super().__init__()
        self.channels = channels
        self.vq_size = vq_size
        out_dim = 2*channels if learn_sigma else channels
        self.dim = dim
        self.dim_mults = dim_mults    # [Newly added for controlnet]
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2

        self.init_conv = nn.Conv3d(channels, init_dim, (init_kernel_size, init_kernel_size,
                                   init_kernel_size), padding=(init_padding, init_padding, init_padding))

    
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]  # [72, 72, 72, 144, 288, 576]
        in_out = list(zip(dims[:-1], dims[1:]))  # [(72, 72), (72, 72), (72, 144), (144, 288), (288, 576)]
        self.feature_fusion = np.asarray([item[0]==item[1] for item in in_out]).sum()  # top 2 layers, thus self.feature_fusion=2
        self.num_mid_DiT = num_mid_DiT # default is 1

        # time conditioning
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning
        self.prompt_dim = prompt_dim
        
        # layers
        ### miniDiT blocks 
        self.sub_volume_size = sub_volume_size
        self.patch_size = patch_size # default is 1
        self.x_embedder = PatchEmbed_Voxel(sub_volume_size, patch_size, channels, dim, bias=True)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=False)
        self.IntraPatchFlow_input = nn.ModuleList()
        for i in range(self.feature_fusion):  # self.feature_fusion=2
            temp = [DiTBlock(dim, 
                     DiT_num_heads,
                     context_dim=prompt_dim,
                     mlp_ratio=mlp_ratio,
                     )]
            temp.append(FinalLayer(dim, self.patch_size, dim))  # 只是一个维度不变的linear layer + modulation
            self.IntraPatchFlow_input.append(nn.ModuleList(temp))
        self.IntraPatchFlow_input = nn.ModuleList(self.IntraPatchFlow_input)

        self.IntraPatchFlow_mid = []
        for i in range(self.num_mid_DiT): # self.num_mid_DiT=1
            self.IntraPatchFlow_mid.append(DiTBlock(dim, 
                     DiT_num_heads,
                     context_dim=prompt_dim,
                     mlp_ratio=mlp_ratio,
                     ))
        self.IntraPatchFlow_mid = nn.ModuleList(self.IntraPatchFlow_mid)

        self.IntraPatchFlow_output = nn.ModuleList()
        for i in range(self.feature_fusion):  # self.feature_fusion=2
            temp = [DiTBlock(dim, 
                     DiT_num_heads,
                     context_dim=prompt_dim,
                     mlp_ratio=mlp_ratio,
                     skip=True
                     )]
            temp.append(FinalLayer(dim, self.patch_size, dim))
            self.IntraPatchFlow_output.append(nn.ModuleList(temp))
        self.IntraPatchFlow_output = nn.ModuleList(self.IntraPatchFlow_output)


        # block type
        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=time_dim)

        num_resolutions = len(in_out) # in_out:[(72, 72), (72, 72), (72, 144), (144, 288), (288, 576)], thus num_resolutions=5

        # down layers
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (num_resolutions - 1)
            is_first = ind < self.feature_fusion - 1
            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                Residual(PreNorm(dim_out, AttentionBlock(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn[ind] else nn.Identity(),
                Residual(PreNorm(dim_out, CrossAttentionBlock(
                    dim_out, heads=attn_heads, context_dim=prompt_dim))) if use_sparse_linear_attn[ind] else IdentityWithMultiInputs(),  
                block_klass_cond(dim_out, dim_out),
                Residual(PreNorm(dim_out, AttentionBlock(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn[ind] else nn.Identity(),
                Residual(PreNorm(dim_out, CrossAttentionBlock(
                    dim_out, heads=attn_heads, context_dim=prompt_dim))) if use_sparse_linear_attn[ind] else IdentityWithMultiInputs(),  
                Downsample(dim_out) if not is_last and not is_first else nn.Identity()
            ]))

        # middle layers
        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, AttentionBlock(
                    mid_dim, heads=attn_heads)))
        self.mid_cross_attn = Residual(PreNorm(mid_dim, CrossAttentionBlock(
                    mid_dim, heads=attn_heads, context_dim=prompt_dim)))
        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        # up layers
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 2)
            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_out),
                Residual(PreNorm(dim_out, AttentionBlock(
                    dim_out, heads=attn_heads))) if use_sparse_linear_attn[len(in_out) - ind - 1] else nn.Identity(),
                Residual(PreNorm(dim_out, CrossAttentionBlock(
                    dim_out, heads=attn_heads, context_dim=prompt_dim))) if use_sparse_linear_attn[len(in_out) - ind - 1] else IdentityWithMultiInputs(), 
                block_klass_cond(dim_out * 2, dim_in),
                Residual(PreNorm(dim_in, AttentionBlock(
                    dim_in, heads=attn_heads))) if use_sparse_linear_attn[len(in_out) - ind - 1] else nn.Identity(),
                Residual(PreNorm(dim_in, CrossAttentionBlock(
                    dim_in, heads=attn_heads, context_dim=prompt_dim))) if use_sparse_linear_attn[len(in_out) - ind - 1] else IdentityWithMultiInputs(), 
                Upsample(dim_in) if not is_last  else nn.Identity()
            ]))

        # final layers
        self.final_conv = nn.Sequential(
            block_klass(dim * 2, dim),  # block_klass needn't time embeddings
            nn.Conv3d(dim, out_dim, 1)
        )
        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], (self.sub_volume_size[0]//self.patch_size, self.sub_volume_size[1]//self.patch_size , self.sub_volume_size[2]//self.patch_size))
        self.pos_embed.data.copy_(torch.Tensor(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        for blocks in self.IntraPatchFlow_input:
            for block in blocks:
                if isinstance(block, DiTBlock):
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                else:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                    nn.init.constant_(block.linear.weight, 0)
                    nn.init.constant_(block.linear.bias, 0)
                    
        for block in self.IntraPatchFlow_mid:
            if isinstance(block, DiTBlock):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            else:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                nn.init.constant_(block.linear.weight, 0)
                nn.init.constant_(block.linear.bias, 0)
        
        for blocks in self.IntraPatchFlow_output:
            for block in blocks:
                if isinstance(block, DiTBlock):
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                else:
                    nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                    nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
                    nn.init.constant_(block.linear.weight, 0)
                    nn.init.constant_(block.linear.bias, 0)


    def forward_with_cond_scale(
        self,
        *args,
        cond_scale=2.,
        **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale


    def forward(
        self,
        x,
        time,
        y=None,
        res=None,
    ):

        # ========== Processing x ==========
        b = x.shape[0]
        ori_shape = (x.shape[2]*8, x.shape[3]*8, x.shape[4]*8) # 切Patch后的sub_volume大小是(8, 8, 8), PatchSize是4*3*4, 则latent是(32, 24, 32), 变回image是(256, 192, 256)
        x_IntraPatch = x.clone()
        p = self.sub_volume_size[0]
        x_IntraPatch = x_IntraPatch.unfold(2,p,p).unfold(3,p,p).unfold(4,p,p)  # (B, 8, 4, 3, 4, 8, 8, 8)
        p1, p2, p3 = x_IntraPatch.size(2), x_IntraPatch.size(3), x_IntraPatch.size(4) # PatchSize is 4*3*4
        x_IntraPatch = rearrange(x_IntraPatch , 'b c p1 p2 p3 d h w -> (b p1 p2 p3) c d h w')  # (B*4*3*4, 8, 8, 8, 8) 把PatchSize放到Batchsize通道了？
        x = self.init_conv(x)  #  (B, 8, 32, 24, 32) -> (B, dim=72, 32, 24, 32)
        r = x.clone()  # serve as residual

        # ========== Processing t ==========
        t = self.time_mlp(time) if exists(self.time_mlp) else None # time: (B,) -> t: (B, dim*4=288)
        c = t.shape[-1]
        t_DiT = t.unsqueeze(1).repeat(1,p1*p2*p3,1).view(-1,c) # (B, dim*4=288) -> (B*4*3*4, 288)

        # ========== Processing y ==========
        y_DiT = y.unsqueeze(1).repeat(1,p1*p2*p3,1,1).view(-1,y.shape[1],y.shape[2])  # (B, L=100, D=768) -> (B, p1*p2*p3, L, D) -> (B*p1*p2*p3, L, D)



        x_IntraPatch = self.x_embedder(x_IntraPatch)  # (B*4*3*4, 8, 8, 8, 8) -> (B*4*3*4, 72, 8*8*8) -> (B*4*3*4, 512, 72) 
        x_IntraPatch = x_IntraPatch + self.pos_embed  # pos_embed: (B, 512, 72)
        h_DiT, h_Unet, h = [], [], []
        for Block, MlpLayer in self.IntraPatchFlow_input:
            x_IntraPatch = Block(x_IntraPatch, t_DiT, y_DiT) # (B*4*3*4, 512, 72) 
            h_DiT.append(x_IntraPatch)  # Recording the middile features of DiT_blocks
            Unet_feature = self.unpatchify_voxels(MlpLayer(x_IntraPatch, t_DiT)) # (B*4*3*4, 512, 72) ->MLP-> (B*4*3*4, 512, 72) -> (B*4*3*4, 72, 8, 8, 8)
            Unet_feature = rearrange(Unet_feature, '(b p) c d h w -> b p c d h w', b=b)  # (B, 4*3*4, 72, 8, 8, 8)
            Unet_feature = rearrange(Unet_feature, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                        p1=ori_shape[0]//self.vq_size, p2=ori_shape[1]//self.vq_size, p3=ori_shape[2]//self.vq_size)  # (B, 72, 32, 24, 32)
            h_Unet.append(Unet_feature) # will iterate 2 times, thus h_Unet has 2 elements

        for Block in self.IntraPatchFlow_mid:
            x_IntraPatch = Block(x_IntraPatch, t_DiT, y_DiT) # (B*4*3*4, 512, 72) 

        for Block, MlpLayer in self.IntraPatchFlow_output:
            x_IntraPatch = Block(x_IntraPatch, t_DiT, y_DiT, h_DiT.pop()) # x_IntraPatch:(B*4*3*4, 512, 72), t_DiT:(B*4*3*4, 288), h_DiT:list of (B*4*3*4, 512, 72)
            Unet_feature = self.unpatchify_voxels(MlpLayer(x_IntraPatch, t_DiT)) # (B*4*3*4, 512, 72) ->MLP-> (B*4*3*4, 512, 72) -> (B*4*3*4, 72, 8, 8, 8)
            Unet_feature = rearrange(Unet_feature, '(b p) c d h w -> b p c d h w', b=b)  # (B, 4*3*4, 72, 8, 8, 8)
            Unet_feature = rearrange(Unet_feature, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                        p1=ori_shape[0]//self.vq_size, p2=ori_shape[1]//self.vq_size, p3=ori_shape[2]//self.vq_size)  # (B, 72, 32, 24, 32)
            h_Unet.append(Unet_feature) # will iterate 2 times, thus h_Unet has 2+2 elements
        

        for idx, (block1, spatial_attn1, cross_attn1, block2, spatial_attn2, cross_attn2, downsample) in enumerate(self.downs):
            if idx < self.feature_fusion: # idx=0 and 1
                x = x + h_Unet.pop(0) # x:(B, 72, 32, 24, 32)
            x = block1(x, t)  # (B, 72, 32, 24, 32)
            x = spatial_attn1(x)  # (B, 72, 32, 24, 32)
            x = cross_attn1(x, y=y)
            h.append(x)
            x = block2(x, t)
            x = spatial_attn2(x)
            x = cross_attn2(x, y=y)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn(x)
        x = self.mid_cross_attn(x, y=y)
        x = self.mid_block2(x, t)

        for idx, (block1, spatial_attn1, cross_attn1, block2, spatial_attn2, cross_attn2, upsample) in enumerate(self.ups):
            if len(self.ups)-idx <= 2:
                x = x + h_Unet.pop(0)
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = spatial_attn1(x)
            x = cross_attn1(x, y=y)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = spatial_attn2(x)
            x = cross_attn2(x, y=y)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)  # r is residual, thus x:(B, 72*2, 32, 24, 32)
        return self.final_conv(x)  # (B, 72*2, 32, 24, 32) -> (B, 72, 32, 24, 32) ->conv3d-> (B, 8, 32, 24, 32)
    

    def unpatchify_voxels(self, x0):
        """
        input: (N, T, patch_size * patch_size * patch_size * C)    (N, 64, 8*8*8*3)
        voxels: (N, C, X, Y, Z)          (N, 3, 32, 32, 32)
        """
        c = self.dim
        p = self.patch_size
        x,y,z = np.asarray(self.sub_volume_size) // self.patch_size
        assert x * y * z == x0.shape[1]

        x0 = x0.reshape(shape=(x0.shape[0], x, y, z, p, p, p, c))
        x0 = torch.einsum('nxyzpqrc->ncxpyqzr', x0)
        volume = x0.reshape(shape=(x0.shape[0], c, x * p, y * p, z * p))
        return volume