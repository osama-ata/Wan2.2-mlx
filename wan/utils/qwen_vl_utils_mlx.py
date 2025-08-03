# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MLX-compatible video and image utilities

import logging
import os
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from PIL import Image

# Constants
VIDEO_FACTOR = 2
IMAGE_FACTOR = 28
FPS = 2.0


def process_vision_info(
    conversations: List[dict]
) -> Tuple[Optional[List[Image.Image]], Optional[List[Union[mx.array, List[Image.Image]]]]]:
    """
    Process vision information from conversations
    MLX-compatible version with simplified functionality
    """
    image_files = []
    video_files = []
    
    for conversation in conversations:
        if isinstance(conversation, dict) and "content" in conversation:
            content = conversation["content"]
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if "image" in item:
                            image_files.append(item["image"])
                        elif "video" in item:
                            video_files.append(item["video"])
    
    # Process images
    images = None
    if image_files:
        images = []
        for img_path in image_files:
            try:
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    images.append(img)
                else:
                    logging.warning(f"Image file not found: {img_path}")
            except Exception as e:
                logging.error(f"Error loading image {img_path}: {e}")
    
    # Process videos - simplified to return image list for now
    videos = None
    if video_files:
        videos = []
        for vid_path in video_files:
            try:
                # For MLX compatibility, we'll implement a simplified video reader
                frames = _read_video_simple(vid_path)
                if frames is not None:
                    videos.append(frames)
            except Exception as e:
                logging.error(f"Error loading video {vid_path}: {e}")
    
    return images, videos


def _read_video_simple(video_path: str, nframes: int = 8) -> Optional[List[Image.Image]]:
    """
    Simple video reader that returns PIL Images
    This is a placeholder implementation - for production use, 
    you would need a proper video reading library compatible with MLX
    """
    try:
        # Try to use cv2 if available, otherwise return None
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return None
            
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, nframes).astype(int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                frames.append(pil_img)
        
        cap.release()
        return frames if frames else None
        
    except ImportError:
        logging.warning("OpenCV not available, video reading not supported")
        return None
    except Exception as e:
        logging.error(f"Error reading video: {e}")
        return None


def expand2square(pil_img: Image.Image, background_color: Tuple[int, int, int] = (122, 116, 104)) -> Image.Image:
    """
    Expand image to square by adding padding
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def smart_resize(
    pil_img: Image.Image,
    size: int,
    factor: int = IMAGE_FACTOR,
    interpolation: str = "LANCZOS"
) -> Image.Image:
    """
    Smart resize that maintains aspect ratio and aligns to factor
    """
    # Get interpolation method
    interp_method = getattr(Image, interpolation, Image.LANCZOS)
    
    # Calculate new size maintaining aspect ratio
    width, height = pil_img.size
    aspect_ratio = width / height
    
    if aspect_ratio > 1:
        # Width is larger
        new_width = size
        new_height = int(size / aspect_ratio)
    else:
        # Height is larger or square
        new_height = size
        new_width = int(size * aspect_ratio)
    
    # Align to factor
    new_width = (new_width // factor) * factor
    new_height = (new_height // factor) * factor
    
    # Ensure minimum size
    new_width = max(new_width, factor)
    new_height = max(new_height, factor)
    
    return pil_img.resize((new_width, new_height), interp_method)


def pil_to_mlx(pil_img: Image.Image, normalize: bool = True) -> mx.array:
    """
    Convert PIL Image to MLX array
    """
    # Convert to numpy first
    img_array = np.array(pil_img)
    
    # Handle different image modes
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Drop alpha channel
    
    # Convert to float and normalize
    img_array = img_array.astype(np.float32)
    if normalize:
        img_array = img_array / 255.0
        img_array = (img_array - 0.5) / 0.5  # Normalize to [-1, 1]
    
    # Convert to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Convert to MLX array
    return mx.array(img_array)


def mlx_to_pil(mlx_img: mx.array, denormalize: bool = True) -> Image.Image:
    """
    Convert MLX array to PIL Image
    """
    # Convert to numpy
    img_array = np.array(mlx_img)
    
    # Convert from CHW to HWC
    if len(img_array.shape) == 3 and img_array.shape[0] in [1, 3]:
        img_array = np.transpose(img_array, (1, 2, 0))
    
    # Denormalize if needed
    if denormalize:
        img_array = img_array * 0.5 + 0.5  # From [-1, 1] to [0, 1]
        img_array = np.clip(img_array, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    
    # Handle single channel
    if len(img_array.shape) == 3 and img_array.shape[-1] == 1:
        img_array = img_array[:, :, 0]
    
    return Image.fromarray(img_array)


# Backward compatibility functions
def process_image(image_path: str) -> mx.array:
    """Process a single image file"""
    try:
        img = Image.open(image_path)
        return pil_to_mlx(img)
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None


def process_video(video_path: str, nframes: int = 8) -> Optional[mx.array]:
    """Process a video file and return frames as MLX array"""
    frames = _read_video_simple(video_path, nframes)
    if frames is None:
        return None
    
    # Convert frames to MLX arrays and stack
    mlx_frames = []
    for frame in frames:
        mlx_frame = pil_to_mlx(frame)
        mlx_frames.append(mlx_frame)
    
    # Stack along time dimension
    return mx.stack(mlx_frames, axis=0)  # Shape: (T, C, H, W)
