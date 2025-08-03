# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import mlx.core as mx
import mlx.nn as nn

from .attention import attention as flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.astype(mx.float64)

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
    n, c = x.shape[2], x.shape[3] // 2

    # split freqs
    freqs = mx.split(freqs, [c - 2 * (c // 3), c // 3, c // 3], axis=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = x[i, :seq_len].astype(mx.complex64).reshape(
            seq_len, n, -1)
        
        freqs_i = mx.concatenate([
            freqs[0][:f].reshape(f, 1, 1, -1).repeat(h, axis=1).repeat(w, axis=2),
            freqs[1][:h].reshape(1, h, 1, -1).repeat(f, axis=0).repeat(w, axis=2),
            freqs[2][:w].reshape(1, 1, w, -1).repeat(f, axis=0).repeat(h, axis=1)
        ],
                            axis=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = (x_i * freqs_i).real.reshape(seq_len, n, -1)
        x_i = mx.concatenate([x_i, x[i, seq_len:]], axis=0)

        # append to collection
        output.append(x_i)
    return mx.stack(output).astype(mx.float32)


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
            affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

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
        self.head = nn.Linear(dim, out_dim)

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
            self.head(
                self.norm(x) * (1 + mx.squeeze(e[1], axis=2)) + mx.squeeze(e[0], axis=2)))
        return x


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
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = [
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ]

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = mx.concatenate([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               axis=1)

        # initialize weights
        self.init_weights()

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
        x = [self.patch_embedding(mx.expand_dims(u, 0)) for u in x]
        grid_sizes = mx.stack(
            [mx.array(u.shape[2:], dtype=mx.int64) for u in x])
        x = [u.reshape(u.shape[0], u.shape[1], -1).transpose(0, 2, 1) for u in x]
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
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).reshape(bt, seq_len, -1).astype(mx.float32))
        e0 = self.time_projection(e).reshape(bt, seq_len, 6, self.dim)


        # context
        context_lens = None
        context = self.text_embedding(
            mx.stack([
                mx.concatenate(
                    [u, mx.zeros((self.text_len - u.shape[0], u.shape[1]), dtype=u.dtype)])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

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
                m.weight = nn.init.xavier_uniform()(m.weight.shape, m.weight.dtype)
                if m.bias is not None:
                    m.bias = mx.zeros_like(m.bias)

        # init embeddings
        self.patch_embedding.weight = nn.init.xavier_uniform()(self.patch_embedding.weight.shape, self.patch_embedding.weight.dtype)
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                m.weight = mx.random.normal(m.weight.shape, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                m.weight = mx.random.normal(m.weight.shape, std=0.02)

        # init output layer
        self.head.head.weight = mx.zeros_like(self.head.head.weight)
