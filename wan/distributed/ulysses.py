# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import mlx.core as mx

from ..modules.attention import attention


def distributed_attention(
        q,
        k,
        v,
        seq_lens,
        window_size=(-1, -1),
):
    """
    Performs attention using MLX. The distributed logic has been removed as
    MLX is primarily used on single devices.
    The original implementation was based on DeepSpeed Ulysses attention mechanism.
    please refer to https://arxiv.org/pdf/2309.14509

    Args:
        q:           [B, Lq, Nq, C1].
        k:           [B, Lk, Nk, C1].
        v:           [B, Lk, Nk, C2].
        seq_lens:    [B], length of each sequence in batch
        window_size: (left right). If not (-1, -1), apply sliding window local attention.
    """
    # The original implementation used torch.distributed for sequence parallelism.
    # Since MLX is primarily for single-device (Apple Silicon), we remove the
    # distributed logic and just perform standard attention.
    # The `attention` function from `wan.modules.attention` is already
    # using MLX's scaled_dot_product_attention.
    return attention(
        q,
        k,
        v,
        k_lens=seq_lens,
        window_size=window_size,
    )
