# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
from contextlib import contextmanager
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae_simple import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import best_output_size, masks_like


class WanTI2V:
    def __init__(
        self,
        config,
        checkpoint_dir,
        convert_model_dtype=False,
    ):
        logging.info("🔧 Initializing WanTI2V pipeline...")
        self.config = config
        self.param_dtype = config.param_dtype if convert_model_dtype else mx.float32
        logging.info(f"📊 Using parameter dtype: {self.param_dtype}")

        logging.info("🔤 Loading T5 text encoder...")
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        )
        logging.info("✅ T5 text encoder loaded successfully")

        logging.info("🎨 Loading VAE model...")
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
        )
        logging.info("✅ VAE model loaded successfully")
        
        logging.info("🧠 Loading main transformer model...")
        # Load model using from_pretrained method that handles parameter mapping
        self.model = WanModel.from_pretrained(checkpoint_dir, "wan_model_mlx.safetensors", model_type='ti2v')
        logging.info("✅ Main transformer model loaded successfully")

        self.sample_neg_prompt = config.sample_neg_prompt
        logging.info("🚀 WanTI2V pipeline initialization completed")

    def _pil_to_mx(self, img):
        img = np.array(img, dtype=np.float32) / 255.0
        img = (img - 0.5) / 0.5
        return mx.array(img).transpose(2, 0, 1)

    def generate(
        self,
        input_prompt,
        img=None,
        size=(1280, 704),
        max_area=704 * 1280,
        frame_num=81,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
    ):
        logging.info(f"🎬 Starting generation with prompt: '{input_prompt}'")
        logging.info(f"📐 Parameters - Size: {size}, Frames: {frame_num}, Steps: {sampling_steps}")
        logging.info(f"⚙️  Solver: {sample_solver}, Shift: {shift}, Guide scale: {guide_scale}")
        
        if img is not None:
            logging.info("🖼️  Image provided - using image-to-video mode")
            return self.i2v(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
            )
        logging.info("📝 No image provided - using text-to-video mode")
        return self.t2v(
            input_prompt=input_prompt,
            size=size,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
        )

    def t2v(
        self,
        input_prompt,
        size=(1280, 704),
        frame_num=121,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
    ):
        logging.info("🎬 Starting text-to-video generation...")
        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.config.vae_stride[0] + 1,
            size[1] // self.config.vae_stride[1],
            size[0] // self.config.vae_stride[2],
        )
        logging.info(f"🎯 Target latent shape: {target_shape}")
        
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.config.patch_size[1] * self.config.patch_size[2])
            * target_shape[1]
        )
        logging.info(f"📏 Sequence length: {seq_len}")

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
            logging.info("🚫 Using default negative prompt")
        else:
            logging.info(f"🚫 Using custom negative prompt: '{n_prompt}'")
            
        if seed >= 0:
            logging.info(f"🎲 Setting random seed: {seed}")
            mx.random.seed(seed)

        logging.info("🔤 Encoding text prompts...")
        context = self.text_encoder([input_prompt])
        context_null = self.text_encoder([n_prompt])
        logging.info(f"✅ Text encoding completed - Context type: {type(context)}, Null context type: {type(context_null)}")
        if hasattr(context, 'shape'):
            logging.info(f"📊 Context shapes: {context.shape}, {context_null.shape}")
        else:
            logging.info(f"📊 Context lengths: {len(context) if isinstance(context, (list, tuple)) else 'unknown'}")

        logging.info(f"🎲 Generating initial noise with shape: {target_shape}")
        noise = mx.random.normal(target_shape, dtype=mx.float32)

        logging.info(f"⚙️  Initializing {sample_solver} scheduler...")
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps
            )
            sample_scheduler.set_timesteps(sampling_steps, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            from .utils.fm_solvers import FlowDPMSolverMultistepScheduler
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps
            )
            sample_scheduler.set_timesteps(sampling_steps)
            timesteps = sample_scheduler.timesteps
        else:
            raise NotImplementedError("Unsupported solver.")
        
        logging.info(f"✅ Scheduler initialized - {len(timesteps)} timesteps")
        logging.info(f"🕐 Timestep range: {timesteps[0]:.1f} → {timesteps[-1]:.1f}")

        latents = noise
        logging.info("🔄 Starting denoising loop...")
        for i, t in enumerate(tqdm(timesteps)):
            logging.debug(f"🔄 Denoising step {i+1}/{len(timesteps)} - Timestep: {t}")
            timestep = mx.array([t])
            noise_pred_cond = self.model(
                [latents], timestep, context, seq_len
            )[0]
            noise_pred_uncond = self.model(
                [latents], timestep, context_null, seq_len
            )[0]
            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            latent_output = sample_scheduler.step(noise_pred, t, latents)
            # MLX schedulers always return SchedulerOutput with prev_sample attribute  
            latents = latent_output.prev_sample
            
            if i % 10 == 0 or i == len(timesteps) - 1:
                logging.info(f"📊 Step {i+1}/{len(timesteps)} - Latent stats: min={mx.min(latents):.4f}, max={mx.max(latents):.4f}, mean={mx.mean(latents):.4f}")

        logging.info("🎨 Decoding latents to video...")
        videos = self.vae.decode([latents])
        logging.info("✅ Text-to-video generation completed")
        return videos[0]

    def i2v(
        self,
        input_prompt,
        img,
        max_area=704 * 1280,
        frame_num=121,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
    ):
        logging.info("🎬 Starting image-to-video generation...")
        ih, iw = img.height, img.width
        logging.info(f"🖼️  Input image size: {iw}x{ih}")
        
        dh, dw = (
            self.config.patch_size[1] * self.config.vae_stride[1],
            self.config.patch_size[2] * self.config.vae_stride[2],
        )
        logging.info(f"📐 Downsample factors: {dh}x{dw}")
        
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)
        logging.info(f"🎯 Target output size: {ow}x{oh} (max area: {max_area})")
        
        scale = max(ow / iw, oh / ih)
        logging.info(f"📏 Scale factor: {scale:.4f}")
        
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
        logging.info(f"🔄 Resized to: {img.width}x{img.height}")
        
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        logging.info(f"✂️  Cropped to: {img.width}x{img.height} (offset: {x1}, {y1})")
        
        img = self._pil_to_mx(img)
        logging.info(f"📊 Image tensor shape: {img.shape}")

        F = frame_num
        seq_len = ((F - 1) // self.config.vae_stride[0] + 1) * (
            oh // self.config.vae_stride[1]
        ) * (ow // self.config.vae_stride[2]) // (
            self.config.patch_size[1] * self.config.patch_size[2]
        )
        logging.info(f"📏 Sequence length: {seq_len}")

        if seed >= 0:
            logging.info(f"🎲 Setting random seed: {seed}")
            mx.random.seed(seed)
            
        noise_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.config.vae_stride[0] + 1,
            oh // self.config.vae_stride[1],
            ow // self.config.vae_stride[2],
        )
        logging.info(f"🎲 Generating noise with shape: {noise_shape}")
        noise = mx.random.normal(noise_shape, dtype=mx.float32)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
            logging.info("🚫 Using default negative prompt")
        else:
            logging.info(f"🚫 Using custom negative prompt: '{n_prompt}'")

        logging.info("🔤 Encoding text prompts...")
        context = self.text_encoder([input_prompt])
        context_null = self.text_encoder([n_prompt])
        logging.info(f"✅ Text encoding completed - Context type: {type(context)}, Null context type: {type(context_null)}")
        if hasattr(context, 'shape'):
            logging.info(f"📊 Context shapes: {context.shape}, {context_null.shape}")
        else:
            logging.info(f"📊 Context lengths: {len(context) if isinstance(context, (list, tuple)) else 'unknown'}")
        
        logging.info("🎨 Encoding input image...")
        z = self.vae.encode([img])
        logging.info(f"✅ Image encoding completed - Latent shape: {z[0].shape}")

        logging.info(f"⚙️  Initializing {sample_solver} scheduler...")
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps
            )
            sample_scheduler.set_timesteps(sampling_steps, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            from .utils.fm_solvers import FlowDPMSolverMultistepScheduler
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps
            )
            sample_scheduler.set_timesteps(sampling_steps)
            timesteps = sample_scheduler.timesteps
        else:
            raise NotImplementedError("Unsupported solver.")
        
        logging.info(f"✅ Scheduler initialized - {len(timesteps)} timesteps")
        logging.info(f"🕐 Timestep range: {timesteps[0]:.1f} → {timesteps[-1]:.1f}")

        latent = noise
        logging.info("🎭 Creating temporal mask...")
        mask2 = masks_like([noise], zero=True)[1][0]
        latent = (1.0 - mask2) * z[0] + mask2 * latent
        logging.info("✅ Initial latent with image conditioning created")

        logging.info("🔄 Starting denoising loop...")
        for i, t in enumerate(tqdm(timesteps)):
            logging.debug(f"🔄 Denoising step {i+1}/{len(timesteps)} - Timestep: {t}")
            timestep = mx.array([t])
            noise_pred_cond = self.model([latent], timestep, context, seq_len)[0]
            noise_pred_uncond = self.model([latent], timestep, context_null, seq_len)[0]
            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            latent_output = sample_scheduler.step(noise_pred, t, latent)
            # MLX schedulers always return SchedulerOutput with prev_sample attribute
            latent = latent_output.prev_sample
            latent = (1.0 - mask2) * z[0] + mask2 * latent
            
            if i % 10 == 0 or i == len(timesteps) - 1:
                logging.info(f"📊 Step {i+1}/{len(timesteps)} - Latent stats: min={mx.min(latent):.4f}, max={mx.max(latent):.4f}, mean={mx.mean(latent):.4f}")

        logging.info("🎨 Decoding latents to video...")
        videos = self.vae.decode([latent])
        logging.info("✅ Image-to-video generation completed")
        return videos[0]
