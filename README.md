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


Run text-to-video (MLX):
```sh
uv run python generate.py -- --task t2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-T2V-A14B --prompt "A cinematic cat boxing match."
```

> **Note:** Multi-GPU and distributed inference are not currently supported in MLX. All computation runs on unified Apple Silicon memory.

---

## MLX Migration Note

**Wan2.2** has been fully migrated from PyTorch to [Apple MLX](https://github.com/ml-explore/mlx) for native, high-performance video generation on Apple Silicon. All model code, training, and inference now use MLX APIs and unified memory. PyTorch and torch dependencies have been completely removed. See `.github/instructions/mlx.instructions.md` for migration details and acceptance criteria.

---

For advanced usage, model architecture, training, and research details, please see the [original Wan2.2 repository](https://github.com/Wan-Video/Wan2.2) and [official documentation](https://wan.video/welcome?spm=a2ty_o02.30011076.0.0.6c9ee41eCcluqg).


> 💡 This command requires Apple Silicon (M1/M2/M3) and macOS 14+.
