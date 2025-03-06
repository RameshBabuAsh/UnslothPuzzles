# Torch.compile Integration for QLoRA With Zero Graph Breaks

## Problem Statement

The goal of this task was to integrate `torch.compile` into the QLoRA training pipeline without triggering graph breaks or excessive recompilations (targeting a maximum of 30 compilations, with over 60 being unacceptable). The compiled model must yield the same loss as the non-compiled version while boosting training speed. This task required making specific modifications—using patching and custom operators—to overcome the limitations imposed by certain in-place operations and unsupported code paths.

## Introduction

As someone new to `torch.compile`, I embarked on this task eager to explore its performance benefits. My first experiments were encouraging: compiling the base model resulted in no graph breaks and only one recompilation. However, extending the compilation to include LoRA brought a single graph break to light. This README tells the story of my troubleshooting, the challenges I faced, and the targeted modifications I made to ensure a seamless, efficient, and accurate compiled training run for QLoRA.

## Performance Comparison: Compiled vs. Uncompiled Models

- **Compiled Model:**
  - **Main Training:** 145 steps in ~41 seconds
  - **Training Loss:** 3.8655

- **Uncompiled Model:**
  - **Main Training:** 145 steps in ~79 seconds
  - **Training Loss:** 3.8709 

- **Overview**
  - Compiled Model is approximately 2x faster than the uncompiled version, once it got warmed up.
  - There are zero graph breaks.
  - Number of recompilations is just three at max (over multiple tests).

## Early Experiments with torch.compile and LoRA

Initially, I compiled the model without any LoRA modifications. This run was smooth—with no graph breaks and minimal recompilation. Emboldened, I then compiled the model including LoRA. Almost immediately, I encountered a graph break. The culprit turned out to be an in-place modification in the following piece of code:

```python
def enable_input_require_grads(self):
    def make_inputs_require_grads(module, input, output):
        output.requires_grad_(True)
    self._require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
```

The in-place operation (`output.requires_grad_(True)`) was causing the graph break, prompting a search for a solution.

### The In-Place Modification Issue and Its Resolution

To overcome this, I explored alternatives and eventually leveraged a PyTorch custom operator to perform the tensor modification without in-place operations:

```python
@torch.library.custom_op("mylib::gradChanger", mutates_args=())
def gradChanger(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().clone().requires_grad_()
    return y

@gradChanger.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    y = x.detach().clone().requires_grad_()
    return y

def modified_enable_input_require_grads(self):
    def make_inputs_require_grads(module, input, output):
        return gradChanger(output)
    self._require_grads_hook = self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

PreTrainedModel.enable_input_require_grads = modified_enable_input_require_grads
```

This change effectively removed the graph break caused by the in-place modification, allowing LoRA to compile without issues. With this fix, all modules—including the loss function—compiled successfully.

## Tackling QLoRA: The Toughest Challenge

Transitioning from LoRA to QLoRA presented an entirely new level of complexity. My initial compile for QLoRA encountered over 40 graph breaks. Thanks to detailed insights from `torch._dynamo.explanation`, I was able to identify the root causes. One of the major culprits was the Bitsandbytes module, specifically the `Linear4bit` layer. In this layer, a call was made on a `Params4bit` object:

```python
bnb.matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state).to(inp_dtype)
```

Here, calling `t()` on a `Params4bit` (a `UserDefinedObjectVariable`) was not traceable by Dynamo. My initial attempt to use `self.weight.data.t()` was also blocked because the `GetAttribute` operation isn’t supported. After considerable research and testing—and with a little help from the latest nightly version of PyTorch (`torch==2.7.0.dev20250226+cu124`)—I found a path forward.

### Overcoming Additional Graph Breaks

Two more persistent graph breaks stemmed from:
1. The use of `Tensor.data_ptr()` within the helper function:
    ```python
    def get_ptr(A: Optional[Tensor]) -> Optional[ct.c_void_p]:
        if A is None:
            return None
        return ct.c_void_p(A.data_ptr())
    ```
2. The usage of ctypes operations, which are not traceable by Dynamo.

At one point, I considered modifying the Dynamo internals by creating a `custom VariableTracker for ctypes`. The idea was to make operations involving `Tensor.data_ptr()` and `ctypes` more traceable. However, after a full day of investigation, I realized that developing a robust custom VariableTracker was a non-trivial task—it would take far longer than a day or two. Thus, I pivoted to a more targeted solution using custom operators.

To resolve these issues, I once again turned to custom operators. For nearly every graph break encountered, I wrote a custom operator to bypass the unsupported operations. One notable example is the dequantization function:

```python
@torch.library.custom_op("mylib::dynamoCompatibleDequantize", mutates_args={"out"})
def dynamoCompatibleDequantize(A: torch.Tensor,
    qcode: Optional[torch.Tensor] = None,
    absmax: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    qblocksize: int = 4096
) -> None:
    args = (
        ct.c_void_p(qcode.data_ptr()),
        ct.c_void_p(A.data_ptr()),
        ct.c_void_p(absmax.data_ptr()),
        ct.c_void_p(out.data_ptr()),
        ct.c_int(qblocksize),
        ct.c_int(A.numel()),
        ct.c_void_p(torch._C._cuda_getCurrentRawStream(A.device.index)),
    )

    if out.dtype == torch.float16:
        lib.cdequantize_blockwise_fp16(*args)
    elif out.dtype == torch.bfloat16:
        lib.cdequantize_blockwise_bf16(*args)
    elif out.dtype == torch.float32:
        lib.cdequantize_blockwise_fp32(*args)
    else:
        raise ValueError(f"Blockwise quantization only supports 16/32-bit floats, but got {out.dtype}")
```

This operator ensured that the dequantization process was Dynamo-compatible, addressing the graph break issues in a targeted manner. I applied such modifications only where they were strictly necessary.

## The Final Push: Compiling the Complete Pipeline

Once the Bitsandbytes and dequantization issues were resolved, I moved on to compile other critical components of the LLaMA model:
- **MLP, RMSNorm, and Rotary Embedding:** Each of these modules was compiled with careful consideration to avoid dynamic control flows.
- **Attention Layers and Decoder Layers:** These were compiled with zero graph breaks.
- **Loss Function:** Even the loss function was compiled successfully, ensuring that the overall training process was fully optimized.

The training run on a sample model confirmed that the recompilation count was reduced to just three during the warmup phase—far below the excessive counts observed earlier. Importantly, the loss values between the compiled and uncompiled versions were nearly identical.

# Installation Guide

This guide explains how to set up your environment to run the QLoRA compilation project. Follow these steps carefully to ensure all dependencies and modifications are correctly applied.

## Prerequisites

- **GPU Requirement:**  
  A GPU with at least SM 80 or higher is required (this is needed for bfloat operations). The code has been successfully tested on RTX 3060. Note: T4 GPUs of kaggle and colab are lesser than SM 80, so this won't work on those GPUs. You can use A100 as an alternative. bfloat16 operations are only performed by T4 in eager mode but not in compile. I was not able to find a solution for that yet.

- **Operating System:**  
  Linux is preferred. Windows users may need use WSL (version 2).

## Setup Options

You can either create a new Python virtual environment or manually install the required packages into your existing environment.

### Option 1: Using a New Virtual Environment (Recommended)

1. **Download the Repository**  
   Clone or download the project repository to your local machine.

2. **Run the Setup Script**  
   Use the provided `setup.sh` script to create a new virtual environment, install the required libraries, and apply necessary modifications to BitsandBytes.

   ```bash
   bash setup.sh
   ```

   The `setup.sh` script contains the following commands:

   ```bash
   echo "Creating Virtual Environment"
   python3 -m venv unsloth_3
   source unsloth_3/bin/activate

   echo "Installing Required Libraries"

   pip install --no-deps bitsandbytes==0.45.3 accelerate==1.4.0 peft==0.14.0 triton==3.2.0 trl==0.15.2
   pip install rich==13.9.4 transformers==4.49.0 psutil==7.0.0 safetensors==0.5.3
   pip install sentencepiece==0.2.0 protobuf==5.29.3 datasets==3.3.2 huggingface-hub==0.29.1 hf_transfer==0.1.9
   pip install --upgrade --pre --force-reinstall --no-cache-dir torch==2.7.0.dev20250226+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
   pip install notebook

   echo "Modifying BitsandBytes To Maintain The Compatibility With Pytorch Compile"
   SITE_PACKAGES=$(python -c "from sysconfig import get_paths; print(get_paths()['purelib'])")
   echo "Site-packages directory: $SITE_PACKAGES"

   cp functional.py "$SITE_PACKAGES/bitsandbytes/functional.py"
   cp modules.py "$SITE_PACKAGES/bitsandbytes/nn/modules.py"

   echo "Opening Jupyter Notebook"
   jupyter notebook --no-browser --port=8888
   ```

   **Key points in the script:**
   - **Virtual Environment Creation:**  
     A new environment named `unsloth_3` is created and activated.
   - **Dependency Installation:**  
     The script installs specific versions of BitsandBytes, Accelerate, PEFT, Triton, TRL, and other libraries, along with a nightly build of PyTorch.
   - **BitsandBytes Modification:**  
     To ensure compatibility with `torch.compile`, modified versions of `functional.py` and `modules.py` (provided in your repository) are copied into the BitsandBytes installation directory.
   - **Jupyter Notebook Launch:**  
     The script launches a Jupyter Notebook server on port 8888. Open the `unsloth_compiled.ipynb` notebook from the directory for further experimentation.

### Option 2: Manual Installation in an Existing Environment

If you prefer not to create a new virtual environment, you can manually execute the commands in the `setup.sh` script one-by-one in your existing Python environment. Ensure you have the necessary permissions and that any existing package versions do not conflict.

## Post-Installation

- **Jupyter Notebook:**  
  After running the script (or manual commands), navigate to the directory where your notebooks are located and open the `unsloth_compiled.ipynb` file.  
  From here, you can run the cells, modify them as needed, and perform your experiments.

- **GPU Setup:**  
  Make sure your GPU drivers and CUDA toolkit are up-to-date to support the nightly version of PyTorch and the bfloat operations used in this project.

Following these steps will set up your environment for a successful run of the QLoRA compilation project. If you encounter any issues, verify that each command executes successfully and that you have met all prerequisites. Enjoy exploring the performance enhancements brought by `torch.compile`!

## Note

The solution currently doesn't work with `flex-attention` due to unresolved compilation issues, and I haven't found any online fixes yet. I'm still working on this part.

## Conclusion

This task was both challenging and rewarding. Starting with minimal knowledge of `torch.compile`, I navigated through multiple hurdles—from in-place modifications causing graph breaks to the intricacies of Bitsandbytes and ctypes incompatibilities. By applying selective patching and employing custom operators, I was able to transform the QLoRA training pipeline into a fully compiled, efficient, and accurate system.

While the full dynamo compatibility for Bitsandbytes remains an exciting prospect for future work, this task demonstrates significant progress in optimizing deep learning pipelines using `torch.compile`.
