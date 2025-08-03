# Wan2.2 (MLX Edition)

<p align="center">
    <img src="assets/logo.png" width="400"/>
<p>

<p align="center">
    💜 <a href="https://wan.video"><b>Wan</b></a> &nbsp&nbsp ｜ &nbsp&nbsp 🖥️ <a href="https://github.com/Wan-Video/Wan2.2">Original GitHub</a> &nbsp&nbsp  | &nbsp&nbsp🤗 <a href="https://huggingface.co/Wan-AI/">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/Wan-AI">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2503.20314">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg">Blog</a> &nbsp&nbsp |  &nbsp&nbsp 💬  <a href="https://discord.gg/AKNgpMK4Yj">Discord</a>&nbsp&nbsp
    <br>
    📕 <a href="https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz">使用指南(中文)</a>&nbsp&nbsp | &nbsp&nbsp 📘 <a href="https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y">User Guide(English)</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://gw.alicdn.com/imgextra/i2/O1CN01tqjWFi1ByuyehkTSB_!!6000000000015-0-tps-611-1279.jpg">WeChat(微信)</a>&nbsp&nbsp
<br>

---

## About This Fork

This repository is a pure MLX-native port of [Wan2.2](https://github.com/Wan-Video/Wan2.2), designed for Apple Silicon (M1/M2/M3) and macOS 14+. All PyTorch code and dependencies have been removed in favor of [Apple MLX](https://github.com/ml-explore/mlx). For full technical/model details, architecture, and research, please refer to the [original Wan2.2 repository](https://github.com/Wan-Video/Wan2.2), [paper](https://arxiv.org/abs/2503.20314), and [official documentation](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg).

---

## Quickstart (MLX)

Clone the repo:
```sh
git clone https://github.com/osama-ata/Wan2.2-mlx.git
cd Wan2.2
```


Install dependencies (with [uv](https://github.com/astral-sh/uv)):
```sh
# Requires Apple Silicon (M1/M2/M3) and macOS 14+
uv pip install -e .
```

> **Note:** This project uses [uv](https://github.com/astral-sh/uv) and `pyproject.toml` for all dependency and environment management. The old requirements.txt is deprecated.

Download model weights from [Hugging Face](https://huggingface.co/Wan-AI/) or [ModelScope](https://modelscope.cn/organization/Wan-AI) as needed.


---

## Running Scripts (MLX)

### 🎬 Text-to-Video Generation

Generate videos from text prompts using the T2V model:

```sh
# Basic text-to-video generation
uv run python generate.py -- --task t2v-A14B --size 1280*720 --ckpt_dir ./Models/Wan2.2-T2V-A14B --prompt "A cinematic cat boxing match in slow motion"

# High-quality generation with custom parameters
uv run python generate.py -- --task t2v-A14B --size 1920*1080 --ckpt_dir ./Models/Wan2.2-T2V-A14B --prompt "Sunset over a mystical forest with floating lanterns" --num_inference_steps 50 --guidance_scale 7.5
```

### 🖼️➡️🎬 Image-to-Video Generation

Extend images into videos using the I2V model:

```sh
# Convert image to video
uv run python generate.py -- --task i2v-A14B --size 1280*720 --ckpt_dir ./Models/Wan2.2-I2V-A14B --image ./examples/i2v_input.JPG --prompt "The scene comes alive with gentle movement"

# Image-to-video with motion control
uv run python generate.py -- --task i2v-A14B --size 1920*1080 --ckpt_dir ./Models/Wan2.2-I2V-A14B --image ./path/to/image.jpg --prompt "Camera slowly zooms in while leaves gently sway" --motion_strength 0.8
```

### 🖼️📝➡️🎬 Text+Image-to-Video Generation

Combine text prompts with image conditioning using the TI2V model:

```sh
# Text + Image to Video (5B parameter model)
uv run python generate.py -- --task ti2v-5B --size 1280*720 --ckpt_dir ./Models/Wan2.2-TI2V-5B --image ./examples/input.jpg --prompt "The character starts dancing to upbeat music"

# High-quality TI2V with extended parameters
uv run python generate.py -- --task ti2v-5B --size 1920*1080 --ckpt_dir ./Models/Wan2.2-TI2V-5B --image ./path/to/image.jpg --prompt "Epic cinematic scene with dramatic lighting" --num_inference_steps 60 --guidance_scale 8.0 --length 16


# Another example
uv run python generate.py --task ti2v-5B --ckpt_dir '/Applications/Data/Models/Wan2.2-TI2V-5B/src/converted_models' --prompt "A golden retriever in a sunflower field at sunset" --image "examples/i2v_input.JPG" --size "1280*704" --base_seed 42 --sample_solver "dpm++" --sample_steps 10
```

### 🛠️ Model Inspection and Testing

Inspect model structure and verify functionality:

```sh
# Inspect model parameters and structure
uv run python inspect_model.py '/Applications/Data/Models/Wan2.2-TI2V-5B/Models/Converted/wan_model_mlx.safetensors'

# Test model loading and forward pass
uv run python -c "
from wan.modules.model import WanModel
model = WanModel.from_pretrained('/path/to/model')
print(f'Model loaded successfully with {sum(p.size for p in model.parameters())} parameters')
"
```

### 📋 Common Parameters

All generation scripts support these common parameters:

```sh
--task           # Model task: t2v-A14B, i2v-A14B, ti2v-5B
--size           # Output resolution: 1280*720, 1920*1080, etc.
--ckpt_dir       # Path to model checkpoint directory
--prompt         # Text description for generation
--image          # Input image path (for i2v and ti2v tasks)
--output_dir     # Output directory (default: ./outputs)
--num_inference_steps  # Number of denoising steps (default: 30)
--guidance_scale # CFG scale for prompt adherence (default: 7.0)
--length         # Video length in frames (default: 8)
--fps            # Output video framerate (default: 8)
--seed           # Random seed for reproducibility
--motion_strength # Motion intensity for i2v (0.0-1.0)
```

### 🔧 Advanced Usage

#### Batch Processing
```sh
# Process multiple prompts
for prompt in "A cat playing piano" "A dog surfing waves" "A bird building nest"; do
    uv run python generate.py -- --task t2v-A14B --size 1280*720 --ckpt_dir ./Models/Wan2.2-T2V-A14B --prompt "$prompt" --output_dir "./outputs/batch_$(date +%s)"
done
```

#### Custom Configuration
```sh
# Use custom model configurations
export WAN_MODEL_PATH="/path/to/custom/model"
export WAN_CONFIG_PATH="/path/to/config.json"
uv run python generate.py -- --task ti2v-5B --config_override "$WAN_CONFIG_PATH"
```

#### Memory Optimization
```sh
# For large models on systems with limited memory
uv run python generate.py -- --task ti2v-5B --low_mem_mode --gradient_checkpointing --size 720*480
```

---

## 📂 Model Download and Setup

### Automatic Model Download
```sh
# Download T2V model
uv run python -c "
from wan.text2video import Text2VideoGeneration
generator = Text2VideoGeneration()
generator.download_model('t2v-A14B', './Models')
"

# Download I2V model  
uv run python -c "
from wan.image2video import Image2VideoGeneration
generator = Image2VideoGeneration()
generator.download_model('i2v-A14B', './Models')
"

# Download TI2V model (5B parameters)
uv run python -c "
from wan.textimage2video import TextImage2VideoGeneration
generator = TextImage2VideoGeneration()
generator.download_model('ti2v-5B', './Models')
"
```

### Manual Model Setup
1. Download models from [Hugging Face](https://huggingface.co/Wan-AI/) or [ModelScope](https://modelscope.cn/organization/Wan-AI)
2. Extract to appropriate directories:
   ```
   Models/
   ├── Wan2.2-T2V-A14B/
   ├── Wan2.2-I2V-A14B/
   └── Wan2.2-TI2V-5B/
       └── Models/Converted/
           ├── wan_model_mlx.safetensors
           ├── vae_mlx.safetensors
           ├── t5_encoder_mlx.safetensors
           └── config.json
   ```

---

## 🚀 Performance Tips

- **Apple Silicon Optimization**: MLX is optimized for M1/M2/M3 chips with unified memory
- **Memory Usage**: TI2V-5B requires ~20GB+ unified memory for optimal performance
- **Resolution**: Start with 720p for testing, scale to 1080p+ for final outputs
- **Inference Steps**: 30-50 steps for good quality, 60+ for highest quality
- **Batch Size**: MLX automatically optimizes batch processing for your hardware

---

## 🔍 Troubleshooting

### Common Issues
```sh
# Check MLX installation
uv run python -c "import mlx.core as mx; print(f'MLX version: {mx.__version__}')"

# Verify model loading
uv run python -c "
from wan.modules.model import WanModel
try:
    model = WanModel.from_pretrained('/path/to/model')
    print('✅ Model loads successfully')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"

# Test generation pipeline
uv run python generate.py -- --task t2v-A14B --size 512*512 --prompt "test" --length 4 --num_inference_steps 10
```

### Performance Monitoring
```sh
# Monitor memory usage during generation
uv run python -c "
import psutil
import time
from wan.text2video import Text2VideoGeneration

print(f'Memory before: {psutil.virtual_memory().used / 1e9:.1f}GB')
generator = Text2VideoGeneration()
print(f'Memory after load: {psutil.virtual_memory().used / 1e9:.1f}GB')
"
```

---

## MLX Migration Note

**Wan2.2** has been fully migrated from PyTorch to [Apple MLX](https://github.com/ml-explore/mlx) for native, high-performance video generation on Apple Silicon. All model code, training, and inference now use MLX APIs and unified memory. PyTorch and torch dependencies have been completely removed. See `MLX_CONVERSION_SUMMARY.md` for complete migration details and verification results.

**✅ Verified Features:**
- 🎯 5B parameter TI2V model fully functional
- 🚀 Complete forward pass with proper outputs  
- 🔧 MLX Conv3d compatibility with channels-last format
- 💾 Sophisticated PyTorch→MLX parameter mapping
- ⚡ Apple Silicon Metal Performance Shaders optimization
- 🧠 MLX scaled_dot_product_attention integration

---

For advanced usage, model architecture, training, and research details, please see the [original Wan2.2 repository](https://github.com/Wan-Video/Wan2.2) and [official documentation](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg).

> 💡 **Requirements:** Apple Silicon (M1/M2/M3/M4) and macOS 14+
