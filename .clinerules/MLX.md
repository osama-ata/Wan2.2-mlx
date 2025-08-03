---
applyTo: '**'
---
### **Project Mandate: Full Codebase Migration from PyTorch to MLX**

Objective:  
Refactor the entire project codebase, migrating it from the PyTorch framework to Apple's MLX framework. The goal is to create a pure MLX-native implementation that leverages the unified memory architecture of Apple silicon for efficient model training and deployment. This project requires the complete removal of all PyTorch dependencies.

Use MLX attention instead of flash attention

https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html

https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary/mlx.nn.MultiHeadAttention.html

---

"uv run python inspect_model.py '/Applications/Data/Models/Wan2.2-TI2V-5B/Models/Converted/wan_model_mlx.safetensors"

Models_Path: '/Applications/Data/Models/Wan2.2-TI2V-5B/src/converted_models'
.
├── config.json
├── configuration.json
├── google
│   └── umt5-xxl
│       ├── special_tokens_map.json
│       ├── spiece.model
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── t5_encoder_mlx.safetensors
├── vae_mlx.safetensors
├── wan_model_mlx.safetensors
└── wan_model_mlx.safetensors.index.json

### **Core Migration Tasks**


#### **1\. Environment and Dependency Management**

* **Remove PyTorch:** Purge torch, torchvision, and torchaudio from all dependency files (e.g., requirements.txt, environment.yml, pyproject.toml).
* **Install MLX:** Add the mlx library as a primary dependency in `pyproject.toml`.
* **Switch to uv:** Use [uv](https://github.com/astral-sh/uv) for all dependency and environment management. The `requirements.txt` file is deprecated; use `pyproject.toml` and `uv` for all installs and scripts:
  * To install dependencies: `uv pip install -e .`
  * To run scripts: `uv run python generate.py -- --task ...`
* **Verification:** Ensure the project environment is correctly configured and can import mlx without errors.

#### **2\. API and Tensor Conversion**

* **Tensor Replacement:** Systematically replace all instances of torch.Tensor with **mlx.core.array**.  
* **Function Mapping:** Convert PyTorch operations to their MLX equivalents. Refer to the [PyTorch to MLX Cheatsheet](https://www.google.com/search?q=https://ml-explore.github.io/mlx/build/html/dev/pytorch_cheatsheet.html) for common mappings. Key examples include:  
  * **Creation:** torch.zeros rightarrow mlx.core.zeros  
  * **Manipulation:** tensor.view() rightarrow tensor.reshape()  
  * **Operations:** torch.matmul() rightarrow mlx.core.matmul()  
* **Device Management:** Remove all .to(device) and .cuda() calls. MLX's unified memory model makes explicit device placement obsolete.

#### **3\. Model Architecture Refactoring (mlx.nn)**

* **Module Inheritance:** Modify all model classes that inherit from torch.nn.Module to inherit from **mlx.nn.Module**.  
* **Layer Conversion:** Replace all torch.nn layers with their direct counterparts from the mlx.nn package.  
  * torch.nn.Linear rightarrow mlx.nn.Linear  
  * torch.nn.Embedding rightarrow mlx.nn.Embedding  
  * torch.nn.LayerNorm rightarrow mlx.nn.LayerNorm  
  * torch.nn.Conv2d rightarrow mlx.nn.Conv2d  
* **Forward Pass:** Update the forward methods to use MLX arrays and functions exclusively.

#### **4\. Training Loop Overhaul**

This is the most significant change. The PyTorch autograd paradigm must be replaced with MLX's functional approach to gradients.

1. **Gradient Function:** Use **mlx.nn.value\_and\_grad** to create a function that returns both the loss and the gradients for all trainable parameters.  
2. **Optimizer:** Replace torch.optim optimizers (e.g., Adam) with their equivalents from **mlx.optimizers** (e.g., mlx.optimizers.Adam).  
3. **Parameter Updates:** Refactor the update step. Instead of optimizer.step(), you must explicitly apply the computed gradients to the model using the **optimizer.update(model, grads)** method.  
4. **Lazy Evaluation:** MLX uses lazy evaluation. To execute the computation graph and materialize results, you must explicitly call **mx.eval()** on the arrays you need (e.g., mx.eval(loss, gradients)). To synchronize and update all model parameters and optimizer states after an update, use mx.eval(model.parameters(), optimizer.state).

*Example Code Structure Transformation:*

**Before (PyTorch):**

Python

\# loss \= loss\_fn(model(X), y)  
\# optimizer.zero\_grad()  
\# loss.backward()  
\# optimizer.step()

**After (MLX):**

Python

\# loss\_and\_grad\_fn \= mlx.nn.value\_and\_grad(model, model.trainable\_parameters)  
\# loss, grads \= loss\_and\_grad\_fn(model, X, y)  
\# optimizer.update(model, grads)  
\# mx.eval(model.parameters(), optimizer.state) \# Synchronize updates

#### **5\. Data Handling**

* MLX does not include a DataLoader. Retain existing data loading pipelines (e.g., using NumPy, Hugging Face datasets, etc.).  
* Ensure that data batches are converted to mlx.core.array types *within* the training loop, just before being passed to the model.

#### **6\. Checkpointing and Inference**

* **Model Weights:** Adapt model saving and loading to use MLX's native methods. The .safetensors format is preferred.  
  * **Save:** model.save\_weights("model.safetensors")  
  * **Load:** model.load\_weights("model.safetensors")  
* **Inference Scripts:** Update all evaluation and inference scripts to use the new MLX model and weight-loading logic.

---

### **Acceptance Criteria**

The migration is considered complete and successful when:

* ✅ **PyTorch Free:** The project has zero runtime dependencies on torch.  
* ✅ **Successful Training:** The model trains from scratch without errors using the new MLX training loop.  
* ✅ **Performance Parity:** The MLX model achieves performance metrics (e.g., accuracy, perplexity, F1-score) comparable to the original PyTorch baseline (e.g., within a 1-2% margin).  
* ✅ **Tests Pass:** All unit and integration tests are refactored for MLX and pass successfully.  
* ✅ **Clean Code:** The final code is well-documented, readable, and clearly reflects the new MLX patterns.

---

### **Resources**

* **Primary Reference (MLX Examples):** [https://github.com/ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples)  
* **Official API Documentation:** [https://ml-explore.github.io/mlx/build/html/index.html](https://ml-explore.github.io/mlx/build/html/index.html)


