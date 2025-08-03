# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Simple VAE implementation for MLX
import logging
import os
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

__all__ = [
    'SimpleWanVAE',
]


class SimpleWanVAE:
    """A simplified VAE wrapper that loads pre-converted MLX weights directly."""
    
    def __init__(
        self,
        z_dim: int = 48,  # Updated to match the model weights
        vae_pth: str = None,
        dtype: mx.Dtype = mx.float32,
    ):
        self.dtype = dtype
        self.z_dim = z_dim
        
        # VAE statistics (from the original implementation)
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = mx.array(mean, dtype=dtype)
        self.std = mx.array(std, dtype=dtype)
        self.scale = [self.mean, 1.0 / self.std]
        
        # Load the converted weights
        self.weights = {}
        if vae_pth and os.path.exists(vae_pth):
            logging.info(f"Loading VAE weights from {vae_pth}")
            with safe_open(vae_pth, framework='mlx') as f:
                self.weights = {key: f.get_tensor(key) for key in f.keys()}
            logging.info(f"Loaded {len(self.weights)} weight tensors")
        else:
            logging.error(f"VAE weights file not found: {vae_pth}")
            
        # Create a mock model with the required attributes
        self.model = MockVAEModel(z_dim, self.weights, self.scale)


class MockVAEModel:
    """Mock VAE model that provides the required interface."""
    
    def __init__(self, z_dim: int, weights: dict, scale: List[mx.array]):
        self.z_dim = z_dim
        self.weights = weights
        self.scale = scale
        
    def encode(self, videos: mx.array, scale: List[mx.array]) -> mx.array:
        """Simplified encode that returns a placeholder tensor."""
        # Handle both images (4D) and videos (5D)
        if videos.ndim == 4:
            # Image: (b, c, h, w) -> add time dimension
            b, c, h, w = videos.shape
            t = 1
        elif videos.ndim == 5:
            # Video: (b, c, t, h, w)
            b, c, t, h, w = videos.shape
        else:
            raise ValueError(f"Expected 4D or 5D input, got {videos.ndim}D")
            
        latent_t = ((t - 1) // 4) + 1  # Assuming 4x temporal downsampling
        latent_h = h // 16  # Assuming 16x spatial downsampling
        latent_w = w // 16
        
        # Generate random latents for now (this should be replaced with actual encoding)
        latents = mx.random.normal((b, self.z_dim, latent_t, latent_h, latent_w), dtype=videos.dtype)
        
        logging.warning("Using placeholder latents - VAE encode not fully implemented")
        return latents
        
    def decode(self, latents: mx.array, scale: List[mx.array]) -> mx.array:
        """Simplified decode that returns a placeholder tensor."""
        # For now, return a placeholder video tensor with the correct shape
        # In a real implementation, this would pass through the decoder network
        b, c, t, h, w = latents.shape
        video_t = (t - 1) * 4 + 1  # Assuming 4x temporal upsampling
        video_h = h * 16  # Assuming 16x spatial upsampling
        video_w = w * 16
        
        # Generate random video for now (this should be replaced with actual decoding)
        video = mx.random.uniform(-1, 1, (b, 3, video_t, video_h, video_w), dtype=latents.dtype)
        
        logging.warning("Using placeholder video - VAE decode not fully implemented")
        return video


class Wan2_1_VAE:
    """VAE wrapper compatible with the existing interface."""
    
    def __init__(
        self,
        z_dim: int = 48,  # Updated to match the model weights
        vae_pth: str = 'cache/vae_step_411000.pth',
        dtype: mx.Dtype = mx.float32
    ):
        self.dtype = dtype
        self.vae = SimpleWanVAE(z_dim=z_dim, vae_pth=vae_pth, dtype=dtype)
        self.model = self.vae.model
        self.scale = self.vae.scale
        
    def encode(self, videos):
        """Encode videos to latents."""
        if not isinstance(videos, list):
            videos = [videos]
            
        results = []
        for video in videos:
            if not isinstance(video, mx.array):
                video = mx.array(video)
            latents = self.model.encode(mx.expand_dims(video, axis=0), self.scale)
            results.append(latents.squeeze(0))
            
        return results
        
    def decode(self, latents):
        """Decode latents to videos."""
        if not isinstance(latents, list):
            latents = [latents]
            
        results = []
        for latent in latents:
            if not isinstance(latent, mx.array):
                latent = mx.array(latent)
            video = self.model.decode(mx.expand_dims(latent, axis=0), self.scale)
            # Clamp to [-1, 1] range
            video = mx.clip(video.squeeze(0), -1, 1)
            results.append(video)
            
        return results
