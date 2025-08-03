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
from .modules.vae2_2 import Wan2_2_VAE
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
        self.config = config
        self.param_dtype = config.param_dtype if convert_model_dtype else mx.float32

        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
        )

        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
        )
        # Create WanModel with proper configuration
        self.model = WanModel(
            patch_size=config.patch_size,
            text_len=config.text_len,
            in_dim=48,  # Based on checkpoint structure
            dim=config.dim,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            text_dim=4096,
            out_dim=48,  # Based on checkpoint structure
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            window_size=config.window_size,
            qk_norm=config.qk_norm,
            cross_attn_norm=config.cross_attn_norm,
            eps=config.eps
        )
        # Load weights
        self.model.load_weights(os.path.join(checkpoint_dir, "wan_model_mlx.safetensors"))

        self.sample_neg_prompt = config.sample_neg_prompt

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
        if img is not None:
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
        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.config.vae_stride[0] + 1,
            size[1] // self.config.vae_stride[1],
            size[0] // self.config.vae_stride[2],
        )
        seq_len = math.ceil(
            (target_shape[2] * target_shape[3])
            / (self.config.patch_size[1] * self.config.patch_size[2])
            * target_shape[1]
        )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        if seed >= 0:
            mx.random.seed(seed)

        context = self.text_encoder([input_prompt])
        context_null = self.text_encoder([n_prompt])

        noise = mx.random.normal(target_shape, dtype=mx.float32)

        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps
            )
            sample_scheduler.set_timesteps(sampling_steps, shift=shift)
            timesteps = sample_scheduler.timesteps
        else:
            raise NotImplementedError("Unsupported solver.")

        latents = noise
        for t in tqdm(timesteps):
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
            latents = sample_scheduler.step(noise_pred, t, latents)

        videos = self.vae.decode([latents])
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
        ih, iw = img.height, img.width
        dh, dw = (
            self.config.patch_size[1] * self.config.vae_stride[1],
            self.config.patch_size[2] * self.config.vae_stride[2],
        )
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)
        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        img = self._pil_to_mx(img)

        F = frame_num
        seq_len = ((F - 1) // self.config.vae_stride[0] + 1) * (
            oh // self.config.vae_stride[1]
        ) * (ow // self.config.vae_stride[2]) // (
            self.config.patch_size[1] * self.config.patch_size[2]
        )

        if seed >= 0:
            mx.random.seed(seed)
        noise = mx.random.normal(
            (
                self.vae.model.z_dim,
                (F - 1) // self.config.vae_stride[0] + 1,
                oh // self.config.vae_stride[1],
                ow // self.config.vae_stride[2],
            ),
            dtype=mx.float32,
        )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        context = self.text_encoder([input_prompt])
        context_null = self.text_encoder([n_prompt])
        z = self.vae.encode([img])

        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.config.num_train_timesteps
            )
            sample_scheduler.set_timesteps(sampling_steps, shift=shift)
            timesteps = sample_scheduler.timesteps
        else:
            raise NotImplementedError("Unsupported solver.")

        latent = noise
        mask2 = masks_like([noise], zero=True)[1][0]
        latent = (1.0 - mask2) * z[0] + mask2 * latent

        for t in tqdm(timesteps):
            timestep = mx.array([t])
            noise_pred_cond = self.model([latent], timestep, context, seq_len)[0]
            noise_pred_uncond = self.model([latent], timestep, context_null, seq_len)[0]
            noise_pred = noise_pred_uncond + guide_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            latent = sample_scheduler.step(noise_pred, t, latent)
            latent = (1.0 - mask2) * z[0] + mask2 * latent

        videos = self.vae.decode([latent])
        return videos[0]
