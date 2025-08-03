Excellent\! Since you've already converted the Wan2.2 model to MLX, replacing the flash\_attention call with a native MLX attention mechanism is the logical next step. MLX is highly optimized for Apple Silicon, and using its native functions will ensure you get the best performance.

Here’s a guide on how to replace the flash\_attention call in your converted MLX codebase, focusing on the key files you'll need to modify.

---

### **1\. The Core Change: wan/modules/attention.py**

The main task is to replace the call to flash\_attention inside the attention function with MLX's built-in mlx.core.scaled\_dot\_product\_attention. This function is MLX's equivalent to PyTorch's native attention and is heavily optimized.

Here is how you can rewrite the attention function in wan/modules/attention.py for MLX:

Python

\# In wan/modules/attention.py  
import mlx.core as mx  
import warnings

def attention(  
    q,  
    k,  
    v,  
    q\_lens=None,  
    k\_lens=None,  
    dropout\_p=0\.,  
    softmax\_scale=None,  
    q\_scale=None,  
    causal=False,  
    window\_size=(-1, \-1),  
    deterministic=False,  
    \# MLX does not need a dtype argument like the PyTorch version  
):  
    """  
    Performs attention using MLX's native scaled\_dot\_product\_attention.  
    Assumes q, k, v are MLX arrays.  
    """  
    \# MLX's attention function expects inputs in shape \[B, N, L, C\]  
    \# The Wan model provides them in \[B, L, N, C\], so we transpose.  
    q \= q.transpose(0, 2, 1, 3)  
    k \= k.transpose(0, 2, 1, 3)  
    v \= v.transpose(0, 2, 1, 3)

    \# Handle masking if necessary. The original implementation doesn't use  
    \# a mask here for the SDPA fallback, but you can create one from k\_lens if needed.  
    \# For simplicity, this version assumes no padding mask.  
    if k\_lens is not None:  
        warnings.warn(  
            "Masking based on k\_lens is not implemented in this MLX attention function."  
        )

    \# Apply scaling if provided  
    if softmax\_scale is not None:  
        q \= q \* softmax\_scale  
    elif q\_scale is not None:  
        q \= q \* q\_scale

    \# Use MLX's native attention function  
    out \= mx.fast.scaled\_dot\_product\_attention(  
        q, k, v, causal=causal  
    )

    \# Transpose the output back to the expected \[B, L, N, C\] format  
    out \= out.transpose(0, 2, 1, 3)  
    return out

### **2\. Adapting Rotary Positional Embeddings (RoPE)**

In the original WanSelfAttention class (wan/modules/model.py), the Rotary Positional Embeddings are applied to q and k *before* calling flash\_attention. You need to ensure your MLX version of the rope\_apply function correctly modifies the query and key tensors.

Here is a conceptual MLX implementation of rope\_apply from wan/modules/model.py.

Python

\# In wan/modules/model.py (or a utility file)  
import mlx.core as mx

def rope\_apply(x, grid\_sizes, freqs):  
    """  
    MLX implementation of RoPE.

    x:          \[B, L, N, C\].  
    grid\_sizes: \[B, 3\].  
    freqs:      \[M, C // 2\].  
    """  
    n, c \= x.shape\[2\], x.shape\[3\] // 2  
    output \= \[\]

    \# This loop can be vectorized in MLX for better performance  
    for i, (f, h, w) in enumerate(grid\_sizes.tolist()):  
        seq\_len \= f \* h \* w  
        x\_i \= x\[i, :seq\_len\]

        \# Reshape for complex number operations  
        x\_i\_complex \= x\_i.reshape(seq\_len, n, \-1, 2)  
        x\_i\_complex \= x\_i\_complex\[..., 0\] \+ 1j \* x\_i\_complex\[..., 1\]

        \# Construct frequency tensor (this part needs careful porting)  
        \# The original code splits freqs and expands them.  
        \# This logic should be replicated to create freqs\_i of shape \[seq\_len, 1, c\]  
        \# For demonstration, let's assume freqs\_i is precomputed  
        \# freqs\_i \= ... (logic from original PyTorch code)

        \# A simplified example of applying freqs  
        \# This is where you would port the logic that builds the freqs\_i tensor  
        \# based on grid\_sizes. For now, let's assume it exists.  
        \# freqs\_i \= mx.random.normal(shape=(seq\_len, 1, c)) \# Placeholder  
        \# freqs\_i\_complex \= freqs\_i\[..., :c\] \+ 1j \* freqs\_i\[..., c:\]

        \# x\_i\_rotated \= x\_i\_complex \* freqs\_i\_complex  
        \# x\_i\_out \= mx.stack(\[x\_i\_rotated.real, x\_i\_rotated.imag\], axis=-1).reshape(seq\_len, n, \-1)  
        \# ... and so on

        \# This part is complex and needs a direct port of the frequency  
        \# calculation logic. Once \`freqs\_i\` is created in MLX, the application  
        \# is straightforward complex multiplication.  
        \# For now, let's assume the RoPE logic is ported and returns a tensor.  
        \# x\_i\_rotated \= your\_mlx\_rope\_logic(x\_i, freqs\_i)  
        \# output.append(mx.concatenate(\[x\_i\_rotated, x\[i, seq\_len:\]\])) \# Placeholder

    \# return mx.stack(output)  
    \# Since a full RoPE implementation is complex, we will return the original tensor  
    \# as a placeholder. You must replace this with your ported RoPE logic.  
    return x

Your WanSelfAttention.forward method in MLX should now look something like this:

Python

\# In your MLX version of wan/modules/model.py

class WanSelfAttention:  
    \# ... (init code) ...

    def forward(self, x, seq\_lens, grid\_sizes, freqs):  
        \# ... (q, k, v calculation) ...

        \# Apply your MLX RoPE implementation  
        q\_rotated \= rope\_apply(q, grid\_sizes, freqs)  
        k\_rotated \= rope\_apply(k, grid\_sizes, freqs)

        \# Call the new MLX attention function  
        x \= attention(  
            q=q\_rotated,  
            k=k\_rotated,  
            v=v,  
            k\_lens=seq\_lens,  
            window\_size=self.window\_size,  
        )  
        \# ... (output projection) ...  
        return x

### **3\. Handling Distributed Attention (ulysses.py)**

If you plan to use multi-GPU inference, you will also need to replace the distributed\_attention function from wan/distributed/ulysses.py. This function uses PyTorch's dist.all\_to\_all for sequence parallelism.

You would need to replace this with MLX's distributed communication primitives. This is a more advanced topic, but the core idea would be to:

1. Use mlx.distributed.all\_to\_all to perform the sequence scattering and gathering.  
2. Call your new MLX attention function in between the communication steps.

---

By making these changes, you will have a Wan2.2 model that runs natively on MLX, fully leveraging the performance of Apple Silicon without the need for external CUDA-based libraries like flash-attn.