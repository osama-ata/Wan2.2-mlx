import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import time
import logging
import safetensors.numpy
import numpy as np

# ##################################################################
# # Utility Functions
# ##################################################################

def patchify(x: mx.array, patch_size: int) -> mx.array:
    """Rearranges spatial dimensions into channel dimensions using MLX operations."""
    if patch_size == 1:
        return x
    if x.ndim == 4: # (B, C, H, W)
        B, C, H, W = x.shape
        q, r = patch_size, patch_size
        # Reshape H and W dimensions
        x = x.reshape(B, C, H // q, q, W // r, r)
        # Transpose to bring patch dimensions to channel dimension
        x = x.transpose(0, 1, 3, 5, 2, 4)  # (B, C, q, r, H//q, W//r)
        x = x.reshape(B, C * q * r, H // q, W // r)
        return x
    elif x.ndim == 5: # (B, C, T, H, W)
        B, C, T, H, W = x.shape
        q, r = patch_size, patch_size
        # CRITICAL FIX: Collapse T into channels like original einops version
        # Original: "b c t (h q) (w r) -> b (c t r q) h w"
        x = x.reshape(B, C, T, H // q, q, W // r, r)
        # Rearrange: (B, C, T, H//q, q, W//r, r) -> (B, C, T, q, r, H//q, W//r) 
        x = x.transpose(0, 1, 2, 4, 6, 3, 5)  # (B, C, T, q, r, H//q, W//r)
        # Collapse C, T, q, r into single channel dimension  
        x = x.reshape(B, C * T * q * r, H // q, W // r)
        return x
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")


def unpatchify(x: mx.array, patch_size: int, t: Optional[int] = None) -> mx.array:
    """Reverses the patchify operation using MLX operations."""
    if patch_size == 1:
        return x
    if x.ndim == 4 and t is not None: 
        # Handle both cases: (B*T, C', H', W') -> (B, C, T, H, W) and (B, C', H', W') -> (B, C, T, H, W)
        B_or_BT, C_prime, H_prime, W_prime = x.shape
        q, r = patch_size, patch_size
        
        # Check if this is the 5D->4D->5D case (temporal collapse/restore)
        # In this case, C_prime = C * T * q * r
        if C_prime % (t * q * r) == 0:
            # This is 5D unpatchify: (B, C*T*q*r, H', W') -> (B, C, T, H, W)
            B = B_or_BT  # B is already correct
            C = C_prime // (t * q * r)
            
            # Reshape to separate all collapsed dimensions
            x = x.reshape(B, C, t, q, r, H_prime, W_prime)
            # Rearrange back: (B, C, T, q, r, H//q, W//r) -> (B, C, T, H//q, q, W//r, r)
            x = x.transpose(0, 1, 2, 5, 3, 6, 4)  # (B, C, T, H_prime, q, W_prime, r)
            # Final reshape to restore spatial dimensions
            x = x.reshape(B, C, t, H_prime * q, W_prime * r)
            return x
        else:
            # This is 4D unpatchify: (B*T, C', H', W') -> (B, C, T, H, W)
            BT = B_or_BT
            B = BT // t
            C = C_prime // (q * r)
            # Reshape to separate patch dimensions
            x = x.reshape(B, t, C, q, r, H_prime, W_prime)
            # Transpose to move patch dimensions back to spatial
            x = x.transpose(0, 2, 1, 5, 3, 6, 4)  # (B, C, T, H_prime, q, W_prime, r)
            x = x.reshape(B, C, t, H_prime * q, W_prime * r)
            return x
    else:
        raise ValueError(f"Invalid input shape: {x.shape} or missing t parameter")


# Simple test to verify the issue is not in the first convolution
class TestVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(12, 160, kernel_size=(3, 3, 3), padding=1)
    
    def encode_test(self, x):
        x = patchify(x, patch_size=2) 
        print(f"After patchify: {x.shape}")
        # Just test the first convolution
        x = self.conv1(x)
        print(f"After conv1: {x.shape}")
        return x

if __name__ == "__main__":
    print('Testing CORRECTED patchify logic...')

    # Test 5D input: (B, C, T, H, W) 
    x = mx.random.normal((1, 3, 8, 64, 64))
    print(f'Original 5D shape: {x.shape}')

    # Apply corrected patchify (should collapse T into channels)
    patched = patchify(x, patch_size=2)
    print(f'After patchify: {patched.shape}')
    print(f'Expected channels: 3 * 8 * 2 * 2 = {3 * 8 * 2 * 2}')
    print(f'Expected shape: (1, 96, 32, 32)')

    # Test unpatchify
    unpatched = unpatchify(patched, patch_size=2, t=8)
    print(f'After unpatchify: {unpatched.shape}')
    print(f'Matches original: {unpatched.shape == x.shape}')
    
    print("\nTesting with corrected first conv expecting 96 channels...")
    # Test with conv that expects the correct channels
    test_conv = nn.Conv3d(96, 160, kernel_size=(3, 3, 3), padding=1)
    try:
        result = test_conv(patched)
        print(f"Conv3d succeeded! Output shape: {result.shape}")
    except Exception as e:
        print(f"Conv3d failed: {e}")
