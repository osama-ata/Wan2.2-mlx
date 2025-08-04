# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import logging
import os
import os.path as osp

import imageio
import numpy as np
from mlx import core as mx

__all__ = ["save_video", "save_image", "str2bool"]


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
    # Ensure tensor is on the CPU and convert to numpy
    logging.info(f"💾 Starting video save process...")
    logging.info(f"📊 Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
    logging.info(f"⚙️  Save parameters - FPS: {fps}, Normalize: {normalize}, Value range: {value_range}")
    
    tensor_np = np.array(tensor, copy=False)
    
    # Debug: Print tensor shape information
    print(f"DEBUG save_video: Original tensor shape: {tensor.shape}")
    print(f"DEBUG save_video: Numpy array shape: {tensor_np.shape}")
    print(f"DEBUG save_video: Numpy array ndim: {tensor_np.ndim}")

    # Clamp and normalize
    logging.debug("🔧 Applying clamping and normalization...")
    tensor_np = np.clip(tensor_np, value_range[0], value_range[1])
    if normalize:
        tensor_np = (tensor_np - value_range[0]) / (value_range[1] - value_range[0])
        logging.debug(f"📊 After normalization - Min: {tensor_np.min():.4f}, Max: {tensor_np.max():.4f}")

    # Handle different tensor formats
    if tensor_np.ndim == 4:
        # If 4D, assume it's (B, C, H, W) - single frame or missing time dimension
        print("DEBUG save_video: Detected 4D tensor, adding time dimension")
        logging.warning("⚠️  4D tensor detected, adding time dimension")
        tensor_np = np.expand_dims(tensor_np, axis=2)  # Add time dimension
    elif tensor_np.ndim != 5:
        raise ValueError(f"Expected 4D or 5D tensor, got {tensor_np.ndim}D tensor with shape {tensor_np.shape}")

    # Create a grid for each time step
    grid_frames = []
    print(f"DEBUG save_video: Processing {tensor_np.shape[2]} frames")
    logging.info(f"🎬 Processing {tensor_np.shape[2]} frames for video creation...")
    for t in range(tensor_np.shape[2]):
        if t % 10 == 0:
            logging.debug(f"🎞️  Processing frame {t+1}/{tensor_np.shape[2]}")
        frame_data = tensor_np[:, :, t, :, :]
        # Implement make_grid functionality with numpy
        # This is a simplified version. For a more robust implementation,
        # a dedicated grid-making function would be needed.
        # This assumes the input is (B, C, H, W) for each frame
        b, c, h, w = frame_data.shape
        ncols = min(b, nrow)
        nrows = (b + ncols - 1) // ncols
        grid = np.zeros(
            (c, h * nrows + (nrows - 1), w * ncols + (ncols - 1)), dtype=tensor_np.dtype
        )
        for i in range(b):
            row = i // ncols
            col = i % ncols
            grid[
                :,
                row * (h + 1) : row * (h + 1) + h,
                col * (w + 1) : col * (w + 1) + w,
            ] = frame_data[i]
        grid_frames.append(grid)

    # Stack frames and permute
    logging.debug("🔄 Stacking frames and converting format...")
    video_grid = np.stack(grid_frames, axis=0)  # T, C, H, W
    video_grid = np.transpose(video_grid, (0, 2, 3, 1))  # T, H, W, C
    logging.debug(f"📊 Video grid shape after transpose: {video_grid.shape}")

    # Scale to 8-bit integer
    logging.debug("🎨 Converting to 8-bit format...")
    video_grid = (video_grid * 255).astype(np.uint8)
    logging.debug(f"📊 Final video stats - Min: {video_grid.min()}, Max: {video_grid.max()}")

    # Determine output file path
    output_file = save_file or osp.join("/tmp", rand_name(suffix=suffix))
    logging.info(f"💾 Writing video to: {output_file}")

    # Write video
    try:
        logging.debug(f"🎬 Starting video encoding with codec: libx264, fps: {fps}")
        writer = imageio.get_writer(output_file, fps=fps, codec="libx264", quality=8)
        for i, frame in enumerate(video_grid):
            if i % 20 == 0 or i == len(video_grid) - 1:
                logging.debug(f"✍️  Writing frame {i+1}/{len(video_grid)}")
            writer.append_data(frame)
        writer.close()
        logging.info(f"✅ Video saved successfully: {output_file}")
        logging.info(f"📊 Final video: {len(video_grid)} frames, {video_grid[0].shape[1]}x{video_grid[0].shape[0]}")
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
    # Convert to numpy
    img_np = np.array(tensor, copy=False)

    # Clamp and normalize
    img_np = np.clip(img_np, value_range[0], value_range[1])
    if normalize:
        img_np = (img_np - value_range[0]) / (value_range[1] - value_range[0])

    # Create grid
    b, c, h, w = img_np.shape
    ncols = min(b, nrow)
    nrows = (b + ncols - 1) // ncols
    grid = np.zeros(
        (c, h * nrows + (nrows - 1), w * ncols + (ncols - 1)), dtype=img_np.dtype
    )
    for i in range(b):
        row = i // ncols
        col = i % ncols
        grid[
            :,
            row * (h + 1) : row * (h + 1) + h,
            col * (w + 1) : col * (w + 1) + w,
        ] = img_np[i]

    # Transpose and scale
    grid = np.transpose(grid, (1, 2, 0))
    grid = (grid * 255).astype(np.uint8)

    # Save image
    try:
        imageio.imwrite(save_file, grid)
        return save_file
    except Exception as e:
        logging.info(f"save_image failed, error: {e}")


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
