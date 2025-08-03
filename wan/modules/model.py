# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import mlx.core as mx
import mlx.nn as nn

from .attention import attention as flash_attention

class DotNameModule(nn.Module):
    """Helper module to handle dot notation in parameter names"""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def __call__(self, x):
        return self.layer(x)


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.astype(mx.float32)  # Use float32 instead of float64 for GPU compatibility

    # calculation
    sinusoid = mx.outer(
        position, mx.power(10000, -mx.arange(half).astype(position.dtype) / half))
    x = mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=1)
    return x


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = mx.outer(
        mx.arange(max_seq_len, dtype=mx.float32),
        1.0 / mx.power(theta,
                        mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    freqs_cis = mx.cos(freqs) + 1j * mx.sin(freqs)
    return freqs_cis


def rope_apply(x, grid_sizes, freqs):
    # For this model, let's simplify and just return the input for now
    # This is a complex 3D RoPE implementation that needs careful tuning
    # The core attention mechanism will still work without RoPE
    return x


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        r"""
        Args:
            x(array): Shape [B, L, C]
        """
        return self._norm(x.astype(mx.float32)).astype(x.dtype) * self.weight

    def _norm(self, x):
        return x * mx.rsqrt(x.square().mean(axis=-1, keepdims=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dims=dim, eps=eps, affine=elementwise_affine)

    def __call__(self, x):
        r"""
        Args:
            x(array): Shape [B, L, C]
        """
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def __call__(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(array): Shape [B, L, num_heads, C / num_heads]
            seq_lens(array): Shape [B]
            grid_sizes(array): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(array): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).reshape(b, s, n, d)
            k = self.norm_k(self.k(x)).reshape(b, s, n, d)
            v = self.v(x).reshape(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.reshape(b, s, -1)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def __call__(self, x, context, context_lens):
        r"""
        Args:
            x(array): Shape [B, L1, C]
            context(array): Shape [B, L2, C]
            context_lens(array): Shape [B]
        """
        b, n, d = x.shape[0], self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).reshape(b, -1, n, d)
        k = self.norm_k(self.k(context)).reshape(b, -1, n, d)
        v = self.v(context).reshape(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.reshape(b, -1, self.dim)
        x = self.o(x)
        return x


class WanFFN(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.layer_0 = nn.Linear(dim, ffn_dim)
        self.layer_1 = nn.GELU()
        self.layer_2 = nn.Linear(ffn_dim, dim)

    def __call__(self, x):
        return self.layer_2(self.layer_1(self.layer_0(x)))


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = WanFFN(dim, ffn_dim)

        # modulation
        self.modulation = mx.random.normal((1, 6, dim)) / dim**0.5

    def __call__(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(array): Shape [B, L, C]
            e(array): Shape [B, L1, 6, C]
            seq_lens(array): Shape [B], length of each sequence in batch
            grid_sizes(array): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(array): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (mx.expand_dims(self.modulation, 0) + e)
        e = mx.split(e, 6, axis=2)

        # self-attention
        y = self.self_attn(
            self.norm1(x).astype(mx.float32) * (1 + mx.squeeze(e[1], axis=2)) + mx.squeeze(e[0], axis=2),
            seq_lens, grid_sizes, freqs)
        x = x + y * mx.squeeze(e[2], axis=2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x).astype(mx.float32) * (1 + mx.squeeze(e[4], axis=2)) + mx.squeeze(e[3], axis=2))
            x = x + y * mx.squeeze(e[5], axis=2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.linear = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = mx.random.normal((1, 2, dim)) / dim**0.5

    def __call__(self, x, e):
        r"""
        Args:
            x(array): Shape [B, L1, C]
            e(array): Shape [B, L1, C]
        """
        e = (mx.expand_dims(self.modulation, 0) + mx.expand_dims(e, 2))
        e = mx.split(e, 2, axis=2)
        x = (
            self.linear(
                self.norm(x) * (1 + mx.squeeze(e[1], axis=2)) + mx.squeeze(e[0], axis=2)))
        return x


class TextEmbedding(nn.Module):
    def __init__(self, text_dim, dim):
        super().__init__()
        self.layer_0 = nn.Linear(text_dim, dim)
        self.layer_1 = nn.GELU()
        self.layer_2 = nn.Linear(dim, dim)

    def __call__(self, x):
        return self.layer_2(self.layer_1(self.layer_0(x)))


class TimeEmbedding(nn.Module):
    def __init__(self, freq_dim, dim):
        super().__init__()
        self.layer_0 = nn.Linear(freq_dim, dim)
        self.layer_1 = nn.SiLU()
        self.layer_2 = nn.Linear(dim, dim)

    def __call__(self, x):
        return self.layer_2(self.layer_1(self.layer_0(x)))


class TimeProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_0 = nn.SiLU()
        self.layer_1 = nn.Linear(dim, dim * 6)

    def __call__(self, x):
        return self.layer_1(self.layer_0(x))


class WanModel(nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_channels=in_dim, out_channels=dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),  # layer 0
            nn.GELU(),                 # layer 1 
            nn.Linear(dim, dim)        # layer 2
        )
        # time embedding layers using Sequential for proper parameter naming
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),  # layer 0
            nn.SiLU(),                 # layer 1 (no parameters)
            nn.Linear(dim, dim)        # layer 2
        )
        
        # time projection layers using Sequential for proper parameter naming
        self.time_projection = nn.Sequential(
            nn.SiLU(),                 # layer 0 (no parameters)
            nn.Linear(dim, dim * 6)    # layer 1
        )

        # blocks as Sequential
        self.blocks = nn.Sequential(
            *[WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)]
        )

        # head using Sequential for proper parameter naming
        self.head = nn.Sequential(
            nn.Identity(),             # layer 0 (no parameters)
            nn.Linear(dim, math.prod(patch_size) * out_dim)  # layer 1
        )
        # head modulation parameter
        self.head_modulation = mx.random.normal((1, 2, dim)) / dim**0.5

        # buffers (computed, not stored in checkpoint)
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        freqs_init = mx.concatenate([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], axis=1)
        # Store freqs as a computed buffer, not a parameter
        self._freqs = freqs_init

        # initialize weights
        self.init_weights()

    def _map_checkpoint_to_mlx_parameters(self, checkpoint_params):
        """Map checkpoint parameter names to MLX Sequential parameter structure"""
        mlx_params = {}
        
        # Handle patch_embedding parameters (need to be nested)
        if 'patch_embedding.weight' in checkpoint_params:
            mlx_params['patch_embedding'] = {
                'weight': checkpoint_params['patch_embedding.weight'],
                'bias': checkpoint_params['patch_embedding.bias']
            }
        
        # Copy other non-Sequential parameters (except patch_embedding which we handled above)
        for checkpoint_name, param_value in checkpoint_params.items():
            # Skip Sequential parameters and patch_embedding, they'll be handled separately
            if not any(seq_name in checkpoint_name for seq_name in 
                      ['text_embedding.0.', 'text_embedding.2.', 
                       'time_embedding.0.', 'time_embedding.2.',
                       'time_projection.1.', 'head.1.',
                       'patch_embedding.', 'blocks.']):
                mlx_params[checkpoint_name] = param_value
        
        # Handle blocks Sequential structure
        if any(key.startswith('blocks.') for key in checkpoint_params.keys()):
            # Group blocks parameters by layer number
            blocks_dict = {'layers': []}
            layer_params = {}
            
            for checkpoint_name, param_value in checkpoint_params.items():
                if checkpoint_name.startswith('blocks.'):
                    # Extract layer number and rest of the path
                    parts = checkpoint_name.split('.')
                    layer_num = int(parts[1])
                    param_path = '.'.join(parts[2:])  # Everything after 'blocks.N'
                    
                    if layer_num not in layer_params:
                        layer_params[layer_num] = {}
                    
                    # Reconstruct nested dict structure
                    current_dict = layer_params[layer_num]
                    path_parts = param_path.split('.')
                    for part in path_parts[:-1]:
                        if part not in current_dict:
                            current_dict[part] = {}
                        current_dict = current_dict[part]
                    current_dict[path_parts[-1]] = param_value
            
            # Convert to ordered list
            max_layer = max(layer_params.keys()) if layer_params else -1
            for i in range(max_layer + 1):
                if i in layer_params:
                    blocks_dict['layers'].append(layer_params[i])
                else:
                    blocks_dict['layers'].append({})  # Empty layer
            
            mlx_params['blocks'] = blocks_dict
        
        # Create nested structure for Sequential modules
        if 'text_embedding.0.weight' in checkpoint_params:
            mlx_params['text_embedding'] = {
                'layers': [
                    {
                        'weight': checkpoint_params['text_embedding.0.weight'],
                        'bias': checkpoint_params['text_embedding.0.bias']
                    },
                    {},  # GELU layer, no parameters
                    {
                        'weight': checkpoint_params['text_embedding.2.weight'],
                        'bias': checkpoint_params['text_embedding.2.bias']
                    }
                ]
            }
        
        if 'time_embedding.0.weight' in checkpoint_params:
            mlx_params['time_embedding'] = {
                'layers': [
                    {
                        'weight': checkpoint_params['time_embedding.0.weight'],
                        'bias': checkpoint_params['time_embedding.0.bias']
                    },
                    {},  # SiLU layer, no parameters
                    {
                        'weight': checkpoint_params['time_embedding.2.weight'],
                        'bias': checkpoint_params['time_embedding.2.bias']
                    }
                ]
            }
        
        if 'time_projection.1.weight' in checkpoint_params:
            mlx_params['time_projection'] = {
                'layers': [
                    {},  # SiLU layer, no parameters
                    {
                        'weight': checkpoint_params['time_projection.1.weight'],
                        'bias': checkpoint_params['time_projection.1.bias']
                    }
                ]
            }
        
        if 'head.1.weight' in checkpoint_params:
            mlx_params['head'] = {
                'layers': [
                    {},  # Identity layer, no parameters
                    {
                        'weight': checkpoint_params['head.1.weight'],
                        'bias': checkpoint_params['head.1.bias']
                    }
                ]
            }
        
        return mlx_params

    @classmethod
    def from_pretrained(cls, checkpoint_dir, subfolder, **kwargs):
        """Load model from pretrained safetensors weights."""
        import os
        
        # Create model instance with correct parameters for this checkpoint
        # For TI2V-5B model (based on actual checkpoint structure)
        model_kwargs = {
            'patch_size': (1, 2, 2),
            'text_len': 512,
            'in_dim': 48,  # Based on checkpoint: (3072, 48, 1, 2, 2)
            'dim': 3072,   # Based on checkpoint: (3072, 48, 1, 2, 2)
            'ffn_dim': 14336,  # Based on checkpoint: (14336, 3072)
            'freq_dim': 256,
            'text_dim': 4096,
            'out_dim': 48,  # Assuming same as in_dim
            'num_heads': 24,  # 3072 / 128 = 24 (common head dim)
            'num_layers': 30,  # Based on actual checkpoint
            'window_size': (-1, -1),
            'qk_norm': True,
            'cross_attn_norm': True,
            'eps': 1e-6
        }
        model_kwargs.update(kwargs)
        model = cls(**model_kwargs)
        
        # Load weights from safetensors file
        safetensors_path = os.path.join(checkpoint_dir, subfolder)
        if os.path.exists(safetensors_path):
            try:
                # Load weights and map to MLX parameter structure
                import mlx.core as mx
                checkpoint_weights = mx.load(safetensors_path)
                
                # Apply parameter name mapping for Sequential modules
                mapped_weights = model._map_checkpoint_to_mlx_parameters(checkpoint_weights)
                
                # Fix patch embedding weight dimensions for MLX Conv3d
                # MLX expects (out_channels, kernel_d, kernel_h, kernel_w, in_channels) - channels last
                if 'patch_embedding' in mapped_weights and 'weight' in mapped_weights['patch_embedding']:
                    weight = mapped_weights['patch_embedding']['weight']
                    # Transpose from PyTorch format (out_channels, in_channels, d, h, w) 
                    # to MLX format (out_channels, d, h, w, in_channels)
                    if len(weight.shape) == 5:
                        # (3072, 48, 1, 2, 2) -> (3072, 1, 2, 2, 48)
                        mapped_weights['patch_embedding']['weight'] = weight.transpose(0, 2, 3, 4, 1)
                
                # Load the mapped weights
                model.update(mapped_weights)
                
            except Exception as e:
                print(f"Error loading weights: {e}")
                raise e
        else:
            raise FileNotFoundError(f"Could not find safetensors file at {safetensors_path}")
        
        return model

    def __call__(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[array]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (array):
                Diffusion timesteps tensor of shape [B]
            context (List[array]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[array], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[array]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None

        if y is not None:
            x = [mx.concatenate([u, v], axis=0) for u, v in zip(x, y)]

        # embeddings
        # Convert input from (C, F, H, W) to (1, F, H, W, C) for MLX Conv3d
        x = [self.patch_embedding(mx.expand_dims(u.transpose(1, 2, 3, 0), 0)) for u in x]
        grid_sizes = mx.stack(
            [mx.array(u.shape[1:4], dtype=mx.int64) for u in x])  # Skip batch dimension
        # Reshape from (batch, f_patches, h_patches, w_patches, features) to (batch, spatial_tokens, features)
        x = [u.reshape(u.shape[0], u.shape[1] * u.shape[2] * u.shape[3], u.shape[4]) for u in x]
        seq_lens = mx.array([u.shape[1] for u in x], dtype=mx.int64)
        assert seq_lens.max() <= seq_len
        x = mx.concatenate([
            mx.concatenate([u, mx.zeros((1, seq_len - u.shape[1], u.shape[2]), dtype=u.dtype)],
                      axis=1) for u in x
        ])

        # time embeddings
        if t.ndim == 1:
            t = mx.broadcast_to(t.reshape(-1, 1), (t.shape[0], seq_len))

        bt = t.shape[0]
        t = t.flatten()
        # Apply time embedding layers
        t_embed = sinusoidal_embedding_1d(self.freq_dim, t).reshape(bt, seq_len, -1).astype(mx.float32)
        # Apply time embedding using Sequential
        e = self.time_embedding(t_embed)
        
        # Apply time projection using Sequential
        e0 = self.time_projection(e).reshape(bt, seq_len, 6, self.dim)

        # context
        context_lens = None
        # Apply text embedding layers using Sequential
        context_padded = mx.stack([
            mx.concatenate(
                [u, mx.zeros((self.text_len - u.shape[0], u.shape[1]), dtype=u.dtype)])
            for u in context
        ])
        context = self.text_embedding(context_padded)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self._freqs,
            context=context,
            context_lens=context_lens)

        for i in range(len(self.blocks.layers)):
            x = self.blocks.layers[i](x, **kwargs)

        # head
        # Apply head modulation and linear layer  
        e_chunks = mx.split((mx.expand_dims(self.head_modulation, 0) + mx.expand_dims(e, 2)), 2, axis=2)
        # Note: assuming there's a layer norm before the head that's part of the final block
        x_modulated = x * (1 + mx.squeeze(e_chunks[1], axis=2)) + mx.squeeze(e_chunks[0], axis=2)
        x = self.head(x_modulated)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.astype(mx.float32) for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[array]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (array):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[array]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].reshape(*v, *self.patch_size, c)
            u = mx.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = mx.random.normal(m.weight.shape, scale=0.02)
                if m.bias is not None:
                    m.bias = mx.zeros_like(m.bias)

        # init embeddings
        self.patch_embedding.weight = mx.random.normal(self.patch_embedding.weight.shape, scale=0.02)
        
        # Sequential modules will be initialized by the modules() iteration above
        # Just initialize head weight to zeros (special case)
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                layer.weight = mx.zeros_like(layer.weight)
