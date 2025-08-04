# MLX-compatible Flow UniPC Multistep Scheduler
# Simplified version for MLX compatibility
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
    SchedulerOutput,
)
from diffusers.utils import deprecate


class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    MLX-compatible UniPC Flow Matching Scheduler
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: List[int] = [],
        solver_p: SchedulerMixin = None,
        use_karras_sigmas: Optional[bool] = False,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        final_sigmas_type: str = "zero",
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
    ):
        # Configuration
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        
        # Initialize arrays
        self.sigmas = None
        self.timesteps = None
        self.num_inference_steps = None
        self._step_index = None
        
        # Model state
        self.model_outputs = []
        self.lower_order_nums = 0
        self.last_sample = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, None] = None,  # MLX doesn't need device
        shift: float = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain.
        """
        if shift is not None:
            self.shift = shift
            
        self.num_inference_steps = num_inference_steps

        # Create timestep schedule
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy()
        
        # Apply time shifting for flow matching
        if self.use_dynamic_shifting:
            # Dynamic shifting logic
            sigmas = timesteps / self.config.num_train_timesteps
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        else:
            # Simple linear scaling
            sigmas = timesteps / self.config.num_train_timesteps
            sigmas = sigmas * self.shift
            
        # Ensure sigmas are in [0, 1] range
        sigmas = np.clip(sigmas, 0.0, 1.0)
        timesteps = sigmas * self.config.num_train_timesteps
        
        # Add final sigma
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        
        # Convert to MLX arrays
        self.sigmas = mx.array(sigmas)
        self.timesteps = mx.array(timesteps).astype(mx.int64)
        
        # Initialize model outputs
        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self._step_index = None

    @property 
    def step_index(self):
        return self._step_index

    def _threshold_sample(self, sample: mx.array) -> mx.array:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."
        """
        dtype = sample.dtype
        
        if not self.config.thresholding:
            return sample
            
        if dtype not in (mx.float32, mx.float64):
            sample = sample.astype(mx.float32)
            
        # Calculate dynamic threshold
        abs_sample = mx.abs(sample)
        s = mx.quantile(abs_sample.flatten(), self.config.dynamic_thresholding_ratio)
        s = mx.maximum(s, mx.array(1.0))
        s = mx.minimum(s, mx.array(self.config.sample_max_value))
        
        # Apply thresholding
        sample = mx.clip(sample, -s, s) / s
        sample = sample.astype(dtype)
        
        return sample

    def convert_model_output(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array = None,
        order: int = None,
    ) -> mx.array:
        """
        Convert the model output to the corresponding type the algorithm needs.
        """
        # For flow matching, we typically predict velocity or noise
        if self.config.prediction_type == "epsilon":
            # Model predicts noise
            return model_output
        elif self.config.prediction_type == "v_prediction":
            # Model predicts velocity
            return model_output
        else:
            raise ValueError(f"Unsupported prediction type: {self.config.prediction_type}")

    def step(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array,
        return_dict: bool = True,
        generator=None,  # MLX doesn't use generators
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE.
        """
        if self.num_inference_steps is None:
            raise ValueError("Number of inference steps is not set")
            
        if self._step_index is None:
            self._init_step_index(timestep)
            
        # Convert model output
        model_output = self.convert_model_output(model_output, timestep, sample)
        
        # Store model output for multi-step  
        # Ensure we don't exceed the list bounds
        if self.lower_order_nums < len(self.model_outputs):
            self.model_outputs[self.lower_order_nums] = model_output
        else:
            # If list is too small, extend it
            while len(self.model_outputs) <= self.lower_order_nums:
                self.model_outputs.append(None)
            self.model_outputs[self.lower_order_nums] = model_output
        
        # Simple Euler step for now (can be extended to higher order)
        if self._step_index < len(self.timesteps) - 1:
            dt = self.timesteps[self._step_index + 1] - self.timesteps[self._step_index]
            dt = dt / self.config.num_train_timesteps
        else:
            dt = -self.timesteps[self._step_index] / self.config.num_train_timesteps
            
        # Flow matching step: x_{t+dt} = x_t + dt * v_t
        prev_sample = sample + dt * model_output
        
        # Apply thresholding if enabled
        if self.config.thresholding:
            prev_sample = self._threshold_sample(prev_sample)
            
        # Update state
        self.lower_order_nums = min(self.lower_order_nums + 1, self.config.solver_order)
        self._step_index += 1
        
        if not return_dict:
            return prev_sample
            
        return SchedulerOutput(prev_sample=prev_sample)

    def _init_step_index(self, timestep):
        """Initialize step index based on timestep"""
        if isinstance(timestep, mx.array):
            timestep = float(timestep)
            
        # Find closest timestep
        diffs = mx.abs(self.timesteps - timestep)
        self._step_index = int(mx.argmin(diffs))

    def add_noise(
        self,
        original_samples: mx.array,
        noise: mx.array,
        timesteps: mx.array,
    ) -> mx.array:
        """
        Add noise to original samples according to the flow matching framework
        """
        # Simple linear interpolation for flow matching
        if isinstance(timesteps, (int, float)):
            timesteps = mx.array([timesteps])
            
        # Normalize timesteps to [0, 1]
        t = timesteps / self.config.num_train_timesteps
        t = mx.expand_dims(t, axis=list(range(1, len(original_samples.shape))))
        
        # Flow matching: x_t = (1-t) * x_0 + t * noise
        noisy_samples = (1 - t) * original_samples + t * noise
        
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
