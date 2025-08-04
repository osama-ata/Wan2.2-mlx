# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Converted to MLX by Gemini.
import logging
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from einops import rearrange
from mlx.utils import tree_flatten, tree_unflatten

__all__ = ["Wan2_2_VAE"]

CACHE_T = 2

# ##################################################################
# # Utility Functions
# ##################################################################

def patchify(x: mx.array, patch_size: int) -> mx.array:
    """Rearranges spatial dimensions into channel dimensions."""
    if patch_size == 1:
        return x
    if x.ndim == 4: # (B, C, H, W)
        return rearrange(
            x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size
        )
    elif x.ndim == 5: # (B, C, T, H, W)
        return rearrange(
            x, "b c t (h q) (w r) -> b (c t r q) h w"
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")


def unpatchify(x: mx.array, patch_size: int, t: Optional[int] = None) -> mx.array:
    """Reverses the patchify operation."""
    if patch_size == 1:
        return x
    if x.ndim == 4 and t is not None: # (B*T, C', H', W') -> (B, C, T, H, W)
         x = rearrange(
            x, "(b t) (c r q) h w -> b c t (h q) (w r)", t=t, q=patch_size, r=patch_size
        )
    elif x.ndim == 5: # (B, C', T, H', W') -> (B, C, T, H, W)
        x = rearrange(
            x, "b (c r q) t h w -> b c t (h q) (w r)", q=patch_size, r=patch_size
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")
    return x


def load_weights(model: nn.Module, path: str):
    """Loads model weights from a file into an MLX model."""
    try:
        weights = mx.load(path)
        model.update(tree_unflatten(list(weights.items())))
        logging.info(f"Loaded weights from {path}")
    except (FileNotFoundError, KeyError) as e:
        logging.warning(f"Could not load weights from {path}: {e}")
        logging.warning("Model is using initial random weights.")

# ##################################################################
# # Core MLX Modules
# ##################################################################

class CausalConv3d(nn.Module):
    """Causal 3D convolution implemented in MLX."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int], stride: Tuple[int, int, int] = (1, 1, 1), padding: Tuple[int, int, int] = (0, 0, 0)):
        super().__init__()
        # In MLX, padding is part of the Conv layer, but here we need dynamic padding.
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        
        # Pre-calculate causal padding: (pad_W_before, pad_W_after, pad_H_before, pad_H_after, pad_T_before, pad_T_after)
        self._padding = (
            padding[2], padding[2],
            padding[1], padding[1],
            (kernel_size[0] - 1) + (kernel_size[0]-1)*(stride[0]-1), 0
        )

    def __call__(self, x: mx.array, cache_x: Optional[mx.array] = None) -> mx.array:
        padding_values = list(self._padding)
        if cache_x is not None and padding_values[4] > 0:
            x = mx.concatenate([cache_x, x], axis=2)
            padding_values[4] = max(0, padding_values[4] - cache_x.shape[2])

        x = mx.pad(x, [(0, 0), (0, 0), (padding_values[4], padding_values[5]), (padding_values[2], padding_values[3]), (padding_values[0], padding_values[1])])
        return self.conv(x)


class RMSNorm(nn.Module):
    """Root Mean Square Normalization layer in MLX."""
    def __init__(self, dims: int, channel_first: bool = True, images: bool = True, bias: bool = False, eps: float = 1e-8):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (1, dims, *broadcastable_dims) if channel_first else (dims,)
        
        self.channel_first = channel_first
        self.eps = eps
        self.weight = mx.ones(shape)
        self.bias = mx.zeros(shape) if bias else 0.0

    def _norm(self, x):
        axis = 1 if self.channel_first else -1
        return x * mx.rsqrt(mx.mean(mx.square(x), axis=axis, keepdims=True) + self.eps)

    def __call__(self, x):
        return self._norm(x) * self.weight + self.bias


class Upsample2D(nn.Module):
    """MLX module for 2D nearest-neighbor upsampling."""
    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

    def __call__(self, x: mx.array) -> mx.array:
        if self.scale_factor == 1:
            return x
        return x.repeat(self.scale_factor, axis=2).repeat(self.scale_factor, axis=3)


class Resample(nn.Module):
    """Resampling block (up/down) in MLX."""
    def __init__(self, dim: int, mode: str):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample2D(scale_factor=2),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample2D(scale_factor=2),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, kernel_size=(3, 1, 1), padding=(0,0,0))
        elif mode == "downsample2d":
            self.resample = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=0)
        elif mode == "downsample3d":
            self.resample = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=0)
            self.time_conv = CausalConv3d(dim, dim, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()
    
    def __call__(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.shape
        
        # Shared logic for spatial resampling
        def spatial_resample(tensor, resampler):
            b_t, c_rt, h_rt, w_rt = tensor.shape
            if self.mode in ["downsample2d", "downsample3d"]:
                # Pad for 'same' style convolution with stride 2
                ph = h_rt % 2
                pw = w_rt % 2
                tensor = mx.pad(tensor, [(0,0), (0,0), (0, ph), (0, pw)])
            return resampler(tensor)

        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache.get(idx) is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x_val = x[:, :, -CACHE_T:, :, :]
                    if cache_x_val.shape[2] < 2 and feat_cache.get(idx) not in [None, "Rep"]:
                        cache_x_val = mx.concatenate([feat_cache[idx][:, :, -1:, :, :], cache_x_val], axis=2)
                    elif cache_x_val.shape[2] < 2 and feat_cache.get(idx) == "Rep":
                        cache_x_val = mx.concatenate([mx.zeros_like(cache_x_val), cache_x_val], axis=2)

                    x = self.time_conv(x, feat_cache.get(idx) if feat_cache.get(idx) != "Rep" else None)
                    feat_cache[idx] = cache_x_val
                    feat_idx[0] += 1
                    
                    x = x.reshape(b, 2, c, t, h, w)
                    x = mx.stack([x[:, 0], x[:, 1]], axis=3)
                    x = x.reshape(b, c, t * 2, h, w)

        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = spatial_resample(x, self.resample)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache.get(idx) is None:
                    feat_cache[idx] = x
                    feat_idx[0] += 1
                else:
                    cache_x_val = x[:, :, -1:, :, :]
                    x = self.time_conv(mx.concatenate([feat_cache[idx][:, :, -1:, :, :], x], axis=2))
                    feat_cache[idx] = cache_x_val
                    feat_idx[0] += 1
        return x


class ResidualBlock(nn.Module):
    """Residual Block in MLX."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.residual_layers = [
            RMSNorm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            RMSNorm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            CausalConv3d(out_dim, out_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        ]
        self.shortcut = CausalConv3d(in_dim, out_dim, kernel_size=1) if in_dim != out_dim else nn.Identity()

    def __call__(self, x: mx.array, feat_cache: Optional[dict] = None, feat_idx: Optional[List[int]] = [0]):
        h = self.shortcut(x)
        
        out = x
        for layer in self.residual_layers:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x_val = out[:, :, -CACHE_T:, :, :]
                if cache_x_val.shape[2] < 2 and feat_cache.get(idx) is not None:
                    cache_x_val = mx.concatenate([feat_cache[idx][:, :, -1:, :, :], cache_x_val], axis=2)
                out = layer(out, feat_cache.get(idx))
                feat_cache[idx] = cache_x_val
                feat_idx[0] += 1
            else:
                out = layer(out)
        return out + h


class AttentionBlock(nn.Module):
    """Self-attention block in MLX."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        # In MLX, initialization happens after creation
        self.proj.weight = mx.zeros_like(self.proj.weight)

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        b, c, t, h, w = x.shape
        
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(b * t, 3, c, -1).transpose(0, 2, 1, 3) # (B*T, C, 3, H*W)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        
        out = nn.scaled_dot_product_attention(q, k, v)
        
        out = out.reshape(b * t, c, h, w)
        out = self.proj(out)
        
        out = rearrange(out, "(b t) c h w -> b c t h w", t=t)
        return out + identity


class AvgDown3D(nn.Module):
    """3D Average Pooling Downsampling in MLX."""
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert (in_channels * self.factor) % out_channels == 0, "Channel dimensions are incompatible."
        self.group_size = (in_channels * self.factor) // out_channels

    def __call__(self, x: mx.array) -> mx.array:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = mx.pad(x, [(0,0), (0,0), (pad_t,0), (0,0), (0,0)])
        
        B, C, T, H, W = x.shape
        x = x.reshape(B, C, T // self.factor_t, self.factor_t, H // self.factor_s, self.factor_s, W // self.factor_s, self.factor_s)
        x = x.transpose(0, 1, 3, 5, 7, 2, 4, 6)
        x = x.reshape(B, C * self.factor, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.reshape(B, self.out_channels, self.group_size, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        return mx.mean(x, axis=2)


class DupUp3D(nn.Module):
    """3D Duplication Upsampling in MLX."""
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert (out_channels * self.factor) % in_channels == 0, "Channel dimensions are incompatible."
        self.repeats = (out_channels * self.factor) // in_channels

    def __call__(self, x: mx.array, first_chunk: bool = False) -> mx.array:
        x = x.repeat(self.repeats, axis=1)
        B, _, T, H, W = x.shape
        x = x.reshape(B, self.out_channels, self.factor, T, H, W)
        x = x.reshape(B, self.out_channels, self.factor_t, self.factor_s, self.factor_s, T, H, W)
        x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.reshape(B, self.out_channels, T * self.factor_t, H * self.factor_s, W * self.factor_s)
        
        if first_chunk:
            x = x[:, :, self.factor_t - 1:, :, :]
        return x


class Down_ResidualBlock(nn.Module):
    """Downsampling Residual Block with Shortcut in MLX."""
    def __init__(self, in_dim, out_dim, dropout, mult, temporal_downsample=False, down_flag=False):
        super().__init__()
        self.avg_shortcut = AvgDown3D(in_dim, out_dim, factor_t=2 if temporal_downsample else 1, factor_s=2 if down_flag else 1)
        
        downsamples = []
        current_dim = in_dim
        for _ in range(mult):
            downsamples.append(ResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        
        if down_flag:
            mode = "downsample3d" if temporal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))
        
        self.downsamples = downsamples

    def __call__(self, x, feat_cache=None, feat_idx=[0]):
        shortcut_out = self.avg_shortcut(x)
        
        main_out = x
        for module in self.downsamples:
            main_out = module(main_out, feat_cache, feat_idx)

        return main_out + shortcut_out


class Up_ResidualBlock(nn.Module):
    """Upsampling Residual Block with Shortcut in MLX."""
    def __init__(self, in_dim, out_dim, dropout, mult, temporal_upsample=False, up_flag=False):
        super().__init__()
        if up_flag:
            self.avg_shortcut = DupUp3D(in_dim, out_dim, factor_t=2 if temporal_upsample else 1, factor_s=2 if up_flag else 1)
        else:
            self.avg_shortcut = None
        
        upsamples = []
        current_dim = in_dim
        for _ in range(mult):
            upsamples.append(ResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        
        if up_flag:
            mode = "upsample3d" if temporal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))
        
        self.upsamples = upsamples

    def __call__(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        main_out = x
        for module in self.upsamples:
             if isinstance(module, Resample) and module.mode == "upsample3d":
                 main_out = module(main_out, feat_cache, feat_idx)
             else:
                 main_out = module(main_out, feat_cache, feat_idx)

        if self.avg_shortcut is not None:
            shortcut_out = self.avg_shortcut(x, first_chunk)
            return main_out + shortcut_out
        return main_out

class Encoder3d(nn.Module):
    """The Encoder part of the Autoencoder in MLX."""
    def __init__(self, dim, z_dim, dim_mult, num_res_blocks, attn_scales, temporal_downsample, dropout):
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]
        
        self.conv1 = CausalConv3d(12, dims[0], kernel_size=(3, 3, 3), padding=(1, 1, 1))

        downsample_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = temporal_downsample[i] if i < len(temporal_downsample) else False
            downsample_blocks.append(
                Down_ResidualBlock(
                    in_dim=in_dim, out_dim=out_dim, dropout=dropout, mult=num_res_blocks,
                    temporal_downsample=t_down_flag, down_flag=(i != len(dim_mult) - 1)
                )
            )
        self.downsamples = downsample_blocks
        
        final_dim = dims[-1]
        self.middle = [
            ResidualBlock(final_dim, final_dim, dropout),
            AttentionBlock(final_dim),
            ResidualBlock(final_dim, final_dim, dropout),
        ]
        
        self.head = [
            RMSNorm(final_dim, images=False),
            nn.SiLU(),
            CausalConv3d(final_dim, z_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        ]

    def __call__(self, x, feat_cache=None, feat_idx=[0]):
        # Initial Convolution
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if cache_x.shape[2] < 2 and feat_cache.get(idx) is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:, :, :], cache_x], axis=2)
            x = self.conv1(x, feat_cache.get(idx))
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # Downsampling blocks
        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx)

        # Middle blocks
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # Head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]
                if cache_x.shape[2] < 2 and feat_cache.get(idx) is not None:
                    cache_x = mx.concatenate([feat_cache[idx][:, :, -1:, :, :], cache_x], axis=2)
                x = layer(x, feat_cache.get(idx))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        
        return x

class Decoder3d(nn.Module):
    """The Decoder part of the Autoencoder in MLX."""
    def __init__(self, dim, z_dim, dim_mult, num_res_blocks, attn_scales, temporal_upsample, dropout):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        
        self.conv1 = CausalConv3d(z_dim, dims[0], kernel_size=(3, 3, 3), padding=(1, 1, 1))
        
        initial_dim = dims[0]
        self.middle = [
            ResidualBlock(initial_dim, initial_dim, dropout),
            AttentionBlock(initial_dim),
            ResidualBlock(initial_dim, initial_dim, dropout),
        ]
        
        upsample_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temporal_upsample[i] if i < len(temporal_upsample) else False
            upsample_blocks.append(
                Up_ResidualBlock(
                    in_dim=in_dim, out_dim=out_dim, dropout=dropout, mult=num_res_blocks + 1,
                    temporal_upsample=t_up_flag, up_flag=(i != len(dim_mult) - 1)
                )
            )
        self.upsamples = upsample_blocks
        
        final_dim = dims[-1]
        self.head = [
            RMSNorm(final_dim, images=False),
            nn.SiLU(),
            CausalConv3d(final_dim, 12, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        ]

    def __call__(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        # Initial Convolution
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if cache_x.shape[2] < 2 and feat_cache.get(idx) is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:, :, :], cache_x], axis=2)
            x = self.conv1(x, feat_cache.get(idx))
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # Middle blocks
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # Upsampling blocks
        for layer in self.upsamples:
            x = layer(x, feat_cache, feat_idx, first_chunk)

        # Head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]
                if cache_x.shape[2] < 2 and feat_cache.get(idx) is not None:
                    cache_x = mx.concatenate([feat_cache[idx][:, :, -1:, :, :], cache_x], axis=2)
                x = layer(x, feat_cache.get(idx))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x

def count_causal_conv3d(model: nn.Module):
    """Counts the number of CausalConv3d modules."""
    return sum(1 for _, m in model.leaf_modules() if isinstance(m, CausalConv3d))

class WanVAE_(nn.Module):
    """Top-level Autoencoder model in MLX."""
    def __init__(self, dim, dec_dim, z_dim, dim_mult, num_res_blocks, attn_scales, temporal_downsample, dropout):
        super().__init__()
        self.z_dim = z_dim
        temporal_upsample = temporal_downsample[::-1]
        
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, temporal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, kernel_size=1)
        self.conv2 = CausalConv3d(z_dim, z_dim, kernel_size=1)
        self.decoder = Decoder3d(dec_dim, z_dim, dim_mult, num_res_blocks, attn_scales, temporal_upsample, dropout)
        
        self.clear_cache()

    def encode(self, x, scale):
        self.clear_cache()
        x = patchify(x, patch_size=2)
        t = x.shape[2]
        iter_count = 1 + (t - 1) // 4
        
        out = None
        for i in range(iter_count):
            self._enc_conv_idx[0] = 0
            start_t = 0 if i == 0 else 1 + 4 * (i - 1)
            end_t = 1 if i == 0 else 1 + 4 * i
            
            chunk = x[:, :, start_t:end_t, :, :]
            out_chunk = self.encoder(chunk, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            
            out = out_chunk if out is None else mx.concatenate([out, out_chunk], axis=2)

        mu, _ = mx.split(self.conv1(out), 2, axis=1)
        
        if isinstance(scale[0], mx.array):
            mu = (mu - scale[0].reshape(1, self.z_dim, 1, 1, 1)) * scale[1].reshape(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]

        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], mx.array):
            z = z / scale[1].reshape(1, self.z_dim, 1, 1, 1) + scale[0].reshape(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
            
        x = self.conv2(z)
        iter_count = x.shape[2]
        
        out = None
        for i in range(iter_count):
            self._conv_idx[0] = 0
            chunk = x[:, :, i:i+1, :, :]
            is_first = (i == 0)
            
            out_chunk = self.decoder(chunk, feat_cache=self._feat_map, feat_idx=self._conv_idx, first_chunk=is_first)
            
            out = out_chunk if out is None else mx.concatenate([out, out_chunk], axis=2)
        
        out = unpatchify(out, patch_size=2, t=iter_count)
        self.clear_cache()
        return out

    def clear_cache(self):
        self._conv_num = count_causal_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = {} # Using dict for sparse caching
        
        self._enc_conv_num = count_causal_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = {}

    def __call__(self, x, scale=[0, 1]):
        mu = self.encode(x, scale)
        x_recon = self.decode(mu, scale)
        return x_recon, mu

# ##################################################################
# # Factory and Wrapper Class
# ##################################################################

def _video_vae(pretrained_path=None, z_dim=16, dim=160, dec_dim=256, **kwargs) -> WanVAE_:
    """Factory function to create and load the WanVAE_ model."""
    cfg = {
        "dim": dim,
        "dec_dim": dec_dim,
        "z_dim": z_dim,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_scales": [],
        "temporal_downsample": [True, True, True],
        "dropout": 0.0,
    }
    cfg.update(kwargs)
    
    model = WanVAE_(**cfg)
    
    if pretrained_path:
        load_weights(model, pretrained_path)
    else:
        logging.warning("No pretrained_path provided for VAE model.")
        
    return model

class Wan2_2_VAE:
    """User-facing wrapper class for the WanVAE_ autoencoder pipeline."""
    def __init__(
        self,
        z_dim: int = 48,
        c_dim: int = 160,
        vae_pth: Optional[str] = None,
        dim_mult: List[int] = [1, 2, 4, 4],
        temporal_downsample: List[bool] = [False, True, True],
        dtype=mx.float32,
    ):
        self.dtype = dtype
        
        mean_vals = [-0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230, -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667]
        std_vals = [0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659, 0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744]
        
        mean = mx.array(mean_vals, dtype=self.dtype)
        std = mx.array(std_vals, dtype=self.dtype)
        self.scale = [mean, 1.0 / std]
        
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
            dim=c_dim,
            dim_mult=dim_mult,
            temporal_downsample=temporal_downsample,
        )
        self.model.eval()

    def encode(self, videos: List[mx.array]) -> Optional[List[mx.array]]:
        try:
            if not isinstance(videos, list):
                raise TypeError("Input 'videos' should be a list of mx.array.")
            
            return [
                self.model.encode(mx.expand_dims(u, 0), self.scale).astype(mx.float32).squeeze(0)
                for u in videos
            ]
        except TypeError as e:
            logging.error(e)
            return None

    def decode(self, zs: List[mx.array]) -> Optional[List[mx.array]]:
        try:
            if not isinstance(zs, list):
                raise TypeError("Input 'zs' should be a list of mx.array.")

            return [
                mx.clip(self.model.decode(mx.expand_dims(u, 0), self.scale).astype(mx.float32), -1.0, 1.0).squeeze(0)
                for u in zs
            ]
        except TypeError as e:
            logging.error(e)
            return None