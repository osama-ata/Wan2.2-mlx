# Inference Log

This log documents the process of running inference on the Wan2.2-TI2V-5B model.

## Attempt 1:

**Objective:** Run inference using the `ti2v-5B` model with the provided MLX-converted weights.

**Plan:**
1.  **Command:** `uv run python generate.py`
2.  **Arguments:**
    *   `--task ti2v-5B`
    *   `--ckpt_dir '/Applications/Data/Models/Wan2.2-TI2V-5B/Models/Converted'`
    *   `--image examples/i2v_input.JPG`

**Execution:**
Executed the following command:
```bash
uv run python generate.py --task ti2v-5B --ckpt_dir '/Applications/Data/Models/Wan2.2-TI2V-5B/Models/Converted' --image examples/i2v_input.JPG
```

**Outcome:**
Failed with an `AssertionError`. The default size `1280*720` is not supported for the `ti2v-5B` task. Supported sizes are `704*1280` and `1280*704`.

---

## Attempt 2:

**Objective:** Run inference with a supported video size.

**Plan:**
1.  **Command:** `uv run python generate.py`
2.  **Arguments:**
    *   `--task ti2v-5B`
    *   `--ckpt_dir '/Applications/Data/Models/Wan2.2-TI2V-5B/Models/Converted'`
    *   `--image examples/i2v_input.JPG`
    *   `--size 1280*704` (Corrected size)

**Execution:**
Executed the following command:
```bash
uv run python generate.py --task ti2v-5B --ckpt_dir '/Applications/Data/Models/Wan2.2-TI2V-5B/Models/Converted' --image examples/i2v_input.JPG --size '1280*704'
```

**Outcome:**
Failed with a `ValueError`. The T5 model definition in `wan/modules/t5.py` does not match the parameters in the `t5_encoder_mlx.safetensors` checkpoint. Specifically, the model is missing weights for `blocks.*.ffn.gate.0.weight`.

---

## Attempt 3:

**Objective:** Correct the T5 model definition to match the checkpoint.

**Plan:**
1.  Analyze `wan/modules/t5.py` to identify the structural mismatch.
2.  Modify the `T5EncoderModel` or its sub-modules to correctly align with the checkpoint's architecture.
3.  Re-run the inference command.

**Execution:**
Executed the following command after modifying `wan/modules/t5.py`:
```bash
uv run python generate.py --task ti2v-5B --ckpt_dir '/Applications/Data/Models/Wan2.2-TI2V-5B/Models/Converted' --image examples/i2v_input.JPG --size '1280*704'
```

**Outcome:**
The T5 loading error was resolved, but a new `ValueError` occurred during VAE model initialization. The `Wan2_2_VAE` model definition in `wan/modules/vae2_2.py` has a significant structural mismatch with the `vae_mlx.safetensors` checkpoint, with 196 unexpected parameters.

---

## Attempt 4:

**Objective:** Correct the VAE model definition to match the checkpoint.

**Plan:**
1.  Analyze `wan/modules/vae2_2.py` and compare it with other VAE definitions in the project to identify the correct architecture.
2.  Modify the pipeline to use the correct VAE model definition.
3.  Re-run the inference command.

**Execution:**
Executed the following command after the first attempt to fix `wan/modules/vae2_2.py`:
```bash
uv run python generate.py --task ti2v-5B --ckpt_dir '/Applications/Data/Models/Wan2.2-TI2V-5B/Models/Converted' --image examples/i2v_input.JPG --size '1280*704'
```

**Outcome:**
Failed again with the same `ValueError` for the VAE model. The previous fixes were insufficient. The error points to a fundamental structural mismatch in the VAE definition.

---

## Attempt 5:

**Objective:** Perform a more thorough correction of the VAE model definition.

**Plan:**
1.  Refactor `CausalConv3d` in `wan/modules/vae2_2.py` to inherit from `nn.Conv3d` to ensure correct parameter naming.
2.  Rename the `weight` parameter to `gamma` in the `RMSNorm` layer to match the checkpoint.
3.  Add the missing `conv1` and `conv2` layers to the `WanVAE_` model and integrate them into the `encode` and `decode` methods.
4.  Re-run the inference command.

**Execution:**
Executed the command again after the second attempt to fix `wan/modules/vae2_2.py`.

**Outcome:**
Failed again with the same `ValueError` for the VAE model. The fixes were still insufficient. The error log clearly indicates a nested parameter structure (e.g., `decoder.upsamples.0.upsamples.0...`) that is not being correctly replicated.

---

## Final Attempt:

**Objective:** Completely refactor the VAE to match the checkpoint's nested structure.

**Plan:**
1.  Re-implement the `Encoder3d` and `Decoder3d` in `wan/modules/vae2_2.py` using nested `nn.ModuleList` to create the exact parameter names required by the checkpoint.
2.  Update the `__call__` methods to correctly iterate through the new nested structure.
3.  This is the final attempt to fix the VAE. If this fails, the checkpoint is likely incompatible with the current code.

**Execution:**
Executed the command after the final VAE refactor.

**Outcome:**
Failed with an `AttributeError: module 'mlx.nn' has no attribute 'ModuleList'`. The `nn.ModuleList` is not a valid MLX class.

---

## Final Attempt (Revised):

**Objective:** Correct the `AttributeError` and run the inference.

**Plan:**
1.  Replace `nn.ModuleList()` with a standard Python list (`[]`) in `wan/modules/vae2_2.py`. MLX automatically discovers modules in Python lists.
2.  Re-run the inference command.

**Execution:**
Executed the command after the final VAE refactor.

**Outcome:**
Failed again with a `ValueError`. The nested module structure was still incorrect. The previous refactoring was not deep enough to match the checkpoint's architecture.

---

## Final Attempt (Complete VAE Rewrite):

**Objective:** Perform a complete and accurate port of the VAE model from PyTorch to MLX.

**Plan:**
1.  Meticulously rewrite `wan/modules/vae2_2.py` by porting the architecture from `wan/modules/vae2_1.py` to MLX, ensuring all nested modules, parameter names, and layer dimensions are perfectly aligned with the `vae_mlx.safetensors` checkpoint.
2.  This is the final attempt. If this fails, the model checkpoint is fundamentally incompatible with the codebase.

**Execution:**
Executed the command after the final VAE refactor.

**Outcome:**
Failed again with a `ValueError`. The nested module structure was still incorrect. A critical error in the dimension handling within the `Decoder3d` class was identified as the root cause.

---

## Final Attempt (Corrected VAE Rewrite):

**Objective:** Perform a complete and accurate port of the VAE model from PyTorch to MLX with corrected dimension logic.

**Plan:**
1.  Meticulously rewrite `wan/modules/vae2_2.py` to correctly handle the changing dimensions between nested blocks in the `Decoder3d`.
2.  Ensure the `Resample` layer correctly halves the output dimension.
3.  This is the absolute final attempt.

**Execution:**
Successfully fixed VAE architecture and parameter structure to match `vae_mlx.safetensors` checkpoint. The VAE model now loads correctly.

**Outcome:**
VAE loading error resolved. Progress moved to main transformer model (WanModel) loading.

---

## Attempt 7:

**Objective:** Fix WanModel structure to match the checkpoint parameters.

**Plan:**
1. Analyze the model structure mismatch in `wan/modules/model.py`.
2. Restructure the `WanModel` class to match the checkpoint's parameter naming.
3. Focus on TI2V-5B specific dimensions: 3072 dim, 14336 ffn_dim, 24 heads, 30 layers.

**Execution:**
Modified `wan/modules/model.py` to restructure embedding and projection layers to match checkpoint parameter names:
- Restructured `text_embedding` and `time_embedding` as sequential layers
- Updated `head` structure to match checkpoint naming
- Added `head_modulation` parameter
- Modified `from_pretrained` method for TI2V-5B configuration

**Outcome:**
Successfully resolved model structure issues. T5 encoder loads correctly, VAE loads correctly, and model parameter count matches (30 layers). However, discovered a Conv3d dimension ordering issue.

---

## Attempt 8:

**Objective:** Fix Conv3d weight dimension ordering mismatch between PyTorch and MLX.

**Current Issue:**
Conv3d layer expects weights in shape `(3072, 1, 2, 2, 48)` but checkpoint contains `(3072, 48, 1, 2, 2)`. This is due to different dimension ordering conventions between PyTorch and MLX for Conv3d weights.

**Plan:**
1. Identify all Conv3d layers in the model that need weight transposition.
2. Add weight reshaping logic in the model loading process to convert from PyTorch format to MLX format.
3. Specifically handle the dimension reordering from `(out_channels, in_channels, d, h, w)` to MLX's expected format.

**Status:**
In progress. Conv3d dimension ordering fix needed to complete the inference pipeline.

**Current Progress:**
- ✅ T5 encoder loading successfully
- ✅ VAE architecture completely fixed and loading
- ✅ WanModel structure matching checkpoint parameters
- ✅ Parameter count verification (30 layers for TI2V-5B)
- ❌ Conv3d weight dimension ordering preventing final model load

**Next Steps:**
Fix Conv3d weight transposition to enable successful model loading and inference execution.
