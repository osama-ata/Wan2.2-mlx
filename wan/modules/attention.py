# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import mlx.core as mx
import mlx.nn as nn
import warnings

__all__ = [
    'attention',
]


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=mx.bfloat16,
    fa_version=None,
):
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
        )
    
    if window_size != (-1, -1):
        warnings.warn(
            'Sliding window local attention is not supported in this implementation.'
        )

    if fa_version is not None:
        warnings.warn(
            'Flash attention version selection is not supported. Using default MLX attention.'
        )

    if deterministic:
        warnings.warn(
            'Deterministic mode is not explicitly supported in MLX attention. Behavior may differ.'
        )

    if q_scale is not None:
        q = q * q_scale

    q = q.astype(dtype)
    k = k.astype(dtype)
    v = v.astype(dtype)

    # MLX scaled_dot_product_attention expects [B, num_heads, L, C]
    # The input is [B, L, num_heads, C], so we need to transpose
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    # Create mask for causal attention
    mask = None
    if causal:
        # MLX's scaled_dot_product_attention handles causal masking internally when `is_causal=True` is not available.
        # We can create a causal mask manually.
        score = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / mx.sqrt(q.shape[-1]))
        if softmax_scale is not None:
            score = score * softmax_scale
        
        l = q.shape[2]
        causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(l)
        score = score + causal_mask
        score = mx.softmax(score, axis=-1)
        if dropout_p > 0.0:
            score = nn.Dropout(p=dropout_p)(score)
        
        out = score @ v
    else:
        # Manual implementation of scaled dot-product attention for non-causal case
        score = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / mx.sqrt(q.shape[-1]))
        if softmax_scale is not None:
            score = score * softmax_scale
        score = mx.softmax(score, axis=-1)
        if dropout_p > 0.0:
            score = nn.Dropout(p=dropout_p)(score)
        out = score @ v

    # Transpose back to [B, L, num_heads, C]
    out = out.transpose(0, 2, 1, 3)
    return out
