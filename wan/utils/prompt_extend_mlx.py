# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# MLX-compatible prompt extension utilities

import json
import logging
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional, Union

import dashscope
import mlx.core as mx
from PIL import Image

from .system_prompt import *

DEFAULT_SYS_PROMPTS = {
    "t2v-A14B": {
        "zh": T2V_A14B_ZH_SYS_PROMPT,
        "en": T2V_A14B_EN_SYS_PROMPT,
    },
    "i2v-A14B": {
        "zh": I2V_A14B_ZH_SYS_PROMPT,
        "en": I2V_A14B_EN_SYS_PROMPT,
    },
}


@dataclass
class ExtendPromptConfig:
    """Configuration for prompt extension"""
    system_prompts: dict = None
    api_key: str = None
    model_name: str = "qwen-vl-max"
    max_tokens: int = 2048
    temperature: float = 0.8
    top_p: float = 0.8


class PromptExtender:
    """
    MLX-compatible prompt extension utility
    Simplified version that focuses on text processing without PyTorch dependencies
    """
    
    def __init__(
        self,
        config: ExtendPromptConfig,
        device: str = None,  # Kept for compatibility but not used
    ):
        self.config = config
        self.system_prompts = config.system_prompts or DEFAULT_SYS_PROMPTS
        
        # Set up API if available
        if config.api_key:
            dashscope.api_key = config.api_key
        elif os.environ.get("DASHSCOPE_API_KEY"):
            dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY")
        else:
            logging.warning("No API key provided for prompt extension")

    def extend_prompt(
        self,
        prompt: str,
        task_type: str = "t2v-A14B",
        language: str = "en",
        image: Optional[Image.Image] = None,
    ) -> str:
        """
        Extend a prompt using the configured method
        """
        if not dashscope.api_key:
            logging.warning("No API key available, returning original prompt")
            return prompt
            
        try:
            return self._extend_with_api(prompt, task_type, language, image)
        except Exception as e:
            logging.error(f"Failed to extend prompt: {e}")
            return prompt

    def _extend_with_api(
        self,
        prompt: str,
        task_type: str,
        language: str,
        image: Optional[Image.Image] = None,
    ) -> str:
        """
        Extend prompt using external API
        """
        system_prompt = self._get_system_prompt(task_type, language)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Add image if provided
        if image is not None:
            # Convert image to temporary file for API
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                image.save(tmp_file.name, "JPEG")
                messages[1]["content"] = [
                    {"image": f"file://{tmp_file.name}"},
                    {"text": prompt}
                ]
        
        try:
            response = dashscope.Generation.call(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                result_format='message'
            )
            
            if response.status_code == HTTPStatus.OK:
                extended_prompt = response.output.choices[0].message.content
                return extended_prompt.strip()
            else:
                logging.error(f"API call failed: {response.message}")
                return prompt
                
        except Exception as e:
            logging.error(f"API call error: {e}")
            return prompt
        finally:
            # Clean up temporary file if created
            if image is not None and 'tmp_file' in locals():
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass

    def _get_system_prompt(self, task_type: str, language: str) -> str:
        """
        Get system prompt for the given task type and language
        """
        if task_type in self.system_prompts and language in self.system_prompts[task_type]:
            return self.system_prompts[task_type][language]
        else:
            # Fallback to English or default
            if task_type in self.system_prompts:
                return self.system_prompts[task_type].get("en", "")
            return ""

    def batch_extend_prompts(
        self,
        prompts: list[str],
        task_type: str = "t2v-A14B",
        language: str = "en",
        images: Optional[list[Image.Image]] = None,
    ) -> list[str]:
        """
        Extend multiple prompts in batch
        """
        extended_prompts = []
        images = images or [None] * len(prompts)
        
        for i, prompt in enumerate(prompts):
            image = images[i] if i < len(images) else None
            extended = self.extend_prompt(prompt, task_type, language, image)
            extended_prompts.append(extended)
            
        return extended_prompts


def create_prompt_extender(
    system_prompts: dict = None,
    api_key: str = None,
    model_name: str = "qwen-vl-max",
    **kwargs
) -> PromptExtender:
    """
    Factory function to create a PromptExtender
    """
    config = ExtendPromptConfig(
        system_prompts=system_prompts,
        api_key=api_key,
        model_name=model_name,
        **kwargs
    )
    return PromptExtender(config)


# Compatibility functions for existing code
def extend_prompt_with_qwen(
    prompt: str,
    task_type: str = "t2v-A14B",
    language: str = "en",
    image: Optional[Image.Image] = None,
) -> str:
    """
    Compatibility function for existing code
    """
    extender = create_prompt_extender()
    return extender.extend_prompt(prompt, task_type, language, image)
