# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import logging
import os
import os.path as osp

import imageio
import numpy as np
from mlx import core as mx

__all__ = ["save_video", "save_image", "str2bool", "masks_like", "best_output_size"]


def _make_grid_mlx(tensor: mx.array, nrow: int = 8, padding: int = 2, normalize: bool = False, value_range: tuple = None):
    """
    Creates a grid of images from a batch of images.
    MLX equivalent of torchvision.utils.make_grid.
    """
    if tensor.ndim == 3:  # (C, H, W) -> (1, C, H, W)
        tensor = mx.expand_dims(tensor, 0)

    # Handle 12-channel VAE output - convert to RGB
    if tensor.shape[1] == 12:
        logging.debug(f"🎨 Converting 12-channel VAE output to RGB in make_grid")
        # Take first 3 channels as RGB
        tensor = tensor[:, :3, :, :]
    elif tensor.shape[1] > 4:
        logging.warning(f"⚠️  Unexpected {tensor.shape[1]} channels, taking first 3 as RGB")
        tensor = tensor[:, :3, :, :]

    # Normalize if required
    if normalize and value_range:
        min_val, max_val = value_range
        tensor = (tensor - min_val) / (max_val - min_val)
        tensor = mx.clip(tensor, 0, 1)

    num_images = tensor.shape[0]
    num_cols = min(nrow, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols

    c, h, w = tensor.shape[1], tensor.shape[2], tensor.shape[3]
    grid_h = num_rows * h + padding * (num_rows - 1)
    grid_w = num_cols * w + padding * (num_cols - 1)
    
    # Initialize grid with padding color (black)
    grid = mx.zeros((c, grid_h, grid_w))

    # Paste images into the grid
    k = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if k < num_images:
                y_start = i * (h + padding)
                x_start = j * (w + padding)
                grid = mx.index_update(
                    grid,
                    (slice(None), slice(y_start, y_start + h), slice(x_start, x_start + w)),
                    tensor[k]
                )
            k += 1
    return grid


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def save_video(
    tensor: mx.array,
    save_file: str = None,
    fps: int = 30,
    suffix: str = ".mp4",
    nrow: int = 8,
    normalize: bool = True,
    value_range=(-1, 1),
):
    """Saves a batch of video frames as a single video file."""
    logging.info(f"💾 Starting video save process...")
    logging.info(f"📊 Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
    logging.info(f"⚙️  Save parameters - FPS: {fps}, Normalize: {normalize}, Value range: {value_range}")
    
    # Determine output file path
    output_file = save_file or osp.join("/tmp", rand_name(suffix=suffix))
    logging.info(f"💾 Writing video to: {output_file}")
    
    try:
        # Preprocess - clamp values to range
        tensor = mx.clip(tensor, min(value_range), max(value_range))
        
        # Handle different input formats
        if tensor.ndim == 4:
            # (B, C, H, W) -> (B, C, 1, H, W) - add time dimension
            logging.warning("⚠️  4D tensor detected, adding time dimension")
            tensor = mx.expand_dims(tensor, axis=2)
        elif tensor.ndim != 5:
            raise ValueError(f"Expected 4D or 5D tensor, got {tensor.ndim}D tensor with shape {tensor.shape}")
        
        # Input tensor shape: (B, C, T, H, W)
        num_frames = tensor.shape[2]
        logging.info(f"🎬 Processing {num_frames} frames for video creation...")
        
        frames = []
        for i in range(num_frames):
            if i % 10 == 0:
                logging.debug(f"🎞️  Processing frame {i+1}/{num_frames}")
            frame_batch = tensor[:, :, i, :, :]  # Shape: (B, C, H, W)
            grid = _make_grid_mlx(frame_batch, nrow=nrow, normalize=normalize, value_range=value_range)
            frames.append(grid)
        
        # Stack frames to create video tensor: (T, C, H_grid, W_grid)
        video_tensor = mx.stack(frames, axis=0)
        
        # Transpose to (T, H_grid, W_grid, C) for imageio
        video_tensor = video_tensor.transpose(0, 2, 3, 1)
        
        # Scale to 8-bit integer
        video_tensor = mx.clip(video_tensor * 255.0 + 0.5, 0, 255).astype(mx.uint8)
        
        # Convert to numpy for imageio
        video_np = np.array(video_tensor)
        logging.debug(f"� Final video shape: {video_np.shape}")
        
        # Write video
        logging.debug(f"🎬 Starting video encoding with codec: libx264, fps: {fps}")
        writer = imageio.get_writer(output_file, fps=fps, codec="libx264", quality=8)
        for i, frame in enumerate(video_np):
            if i % 20 == 0 or i == len(video_np) - 1:
                logging.debug(f"✍️  Writing frame {i+1}/{len(video_np)}")
            writer.append_data(frame)
        writer.close()
        
        logging.info(f"✅ Video saved successfully: {output_file}")
        logging.info(f"📊 Final video: {len(video_np)} frames, {video_np[0].shape[1]}x{video_np[0].shape[0]}")
        
    except Exception as e:
        logging.error(f"❌ save_video failed, error: {e}")
        raise


def save_image(
    tensor: mx.array,
    save_file: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range=(-1, 1),
):
    """Saves a batch of images as a single grid image file."""
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in ['.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp']:
        save_file = osp.splitext(save_file)[0] + '.png'

    try:
        # Clamp tensor values to the specified range
        tensor = mx.clip(tensor, min(value_range), max(value_range))
        
        # Create grid using MLX operations
        grid = _make_grid_mlx(tensor, nrow=nrow, normalize=normalize, value_range=value_range)
        
        # Scale to [0, 255] and cast to uint8
        img_arr = mx.clip(grid * 255.0 + 0.5, 0, 255).astype(mx.uint8)

        # Transpose from (C, H, W) to (H, W, C) for imageio
        if img_arr.shape[0] == 1:  # Grayscale
            img_arr = img_arr[0]
        else:  # RGB
            img_arr = img_arr.transpose(1, 2, 0)
        
        # Convert to numpy and save
        imageio.imwrite(save_file, np.array(img_arr))
        return save_file
        
    except Exception as e:
        logging.error(f"❌ save_image failed, error: {e}")
        raise


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')


def masks_like(tensor, zero=False, p=0.2):
    assert isinstance(tensor, list)
    out1 = [mx.ones_like(u) for u in tensor]
    out2 = [mx.ones_like(u) for u in tensor]

    if zero:
        for i in range(len(out1)):
            if np.random.rand() < p:
                # This part is tricky without a direct equivalent of torch.normal with a generator.
                # Using numpy for random number generation.
                # The original logic seems to be applying a mask based on a random condition.
                # A simplified version is implemented here.
                out1[i][:, 0] = mx.exp(
                    mx.random.normal(shape=(1,), scale=0.5) - 3.5
                ).expand_as(out1[i][:, 0])
                out2[i][:, 0] = mx.zeros_like(out2[i][:, 0])
            # No 'else' needed as the arrays are already ones.

    return out1, out2


def best_output_size(w, h, dw, dh, expected_area):
    # float output size
    ratio = w / h
    ow = (expected_area * ratio)**0.5
    oh = expected_area / ow

    # process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    # process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    # compare ratios
    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2,
                                                 ratio2 / ratio):
        return ow1, oh1
    else:
        return ow2, oh2
