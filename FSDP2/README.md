# QLoRA with FSDP2: Fine-Tuning Llama 3.1 8B on Multi-GPU Setups

## Overview

This task demonstrates how to fine-tune the unsloth/Meta-Llama-3.1-8B model using QLoRA with FSDP2 across 2 or more GPUs. The goal was to showcase a setup where all FSDP2 features (offloading, checkpointing, mixed precision, etc.) are enabled and working as expected—yielding loss equivalent to single GPU training. In this assignment, we target both uncompiled and torch-compiled versions. The solution has been demonstrated in a free Kaggle notebook using 2× Tesla T4 GPUs.

In this folder we have:
- Single GPU notebook (NB 1)
- Multi GPU FSDP2 + QLoRA notebook (no torch compile) (NB2)
- Multi GPU FSDP2 + QLoRA notebook (with torch compile but some graph breaks) (python file -- install libraries as per task 3)
- For getting a model with FSDP2 + QLoRA + torch compile + no graph break follow task 3 readme and setup the environment in a supported hardware and modify the python file `compiled_multigpu.py`.
- For understanding the code in py file, I recommend you to checkout task 3 first.

Talking about the results, both multi GPU and single GPU yielded approximately equal loss but multi GPU notebook has taken a bit longer time because the communication overhead.

## Problem Statement

- **Objective:**  
  Fine-tune the unsloth/Meta-Llama-3.1-8B model with QLoRA using FSDP2 across multiple GPUs.
  
- **Requirements:**  
  - Must use FSDP2 (or related strategies) with full transformers compatibility (i.e., via `TrainingArguments`, `Trainer`, or TRL classes).
  - Loss must be equivalent to single GPU training.
  - Enable all FSDP2 features (offloading, checkpointing, mixed precision training).
  - Demonstrate the solution in a free Kaggle notebook with 2× Tesla T4 GPUs.
  - Optionally integrate torch compile for QLoRA.

- **Additional Considerations:**  
  We may use a pre-quantized 4-bit BnB safetensor file from Unsloth’s HF page or a full 16-bit model. The implementation supports accelerate, with distributed training initialized via FSDP2.

## Code Walkthrough

The provided Python script covers the following key steps:

1. **Distributed Setup:**  
   The script initializes the distributed process group using NCCL and creates a synchronization barrier across GPUs.
   ```python
   def setup_distributed():
       if torch.distributed.is_available() and not torch.distributed.is_initialized():
           torch.distributed.init_process_group(backend="nccl")
       if torch.distributed.is_initialized():
           local_rank = int(os.environ.get("LOCAL_RANK", "0"))
           torch.distributed.barrier(device_ids=[local_rank])
   ```
   Cleanup is performed at the end:
   ```python
   def cleanup_distributed():
       if torch.distributed.is_initialized():
           torch.distributed.destroy_process_group()
   ```

2. **Accelerator Integration:**  
   An `Accelerator` object (from Hugging Face’s Accelerate) is instantiated to manage device allocation and mixed-precision training and cpu offloading.

3. **Model and Quantization Configuration:**  
   - The unsloth/Meta-Llama-3.1-8B model is loaded with a BitsAndBytes configuration for 4-bit quantization.
   - The model is set up with LoRA for parameter-efficient fine-tuning:
     ```python
     model = AutoModelForCausalLM.from_pretrained(
         model_name,
         attn_implementation = "sdpa",
         quantization_config = bnb_config,
         device_map={"": accelerator.process_index}
     )
     ```
   - The LoRA configuration targets key projection layers and freezes non-LoRA parameters.

4. **Model Enhancements:**  
   - Caching is disabled.
   - Gradient checkpointing is enabled to save memory.
   - Input gradients are enabled via a patched function (similar to our previous QLoRA modifications).

5. **Dataset Loading:**  
   The dataset is fetched from Hugging Face (using only the first 10% of the training split) to demonstrate training.
   ```python
   url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
   dataset = load_dataset("json", data_files={"train": url}, split="train[:10%]")
   ```

6. **Training Configuration:**  
   The training itself is managed by an `SFTTrainer` (or related TRL class) with typical training arguments (batch size, gradient accumulation, mixed precision settings, etc.).

7. **Launching and Cleaning Up:**  
   The training is launched, followed by cleanup of distributed resources.

## Installation Guide

This installation guide is similar to the one provided for the previous assignment (question 3). Use the same steps to set up your environment.

### Prerequisites

- **Hardware:**  
  1. For the uncompiled & compiled FSDP2 & QLoRA, we can use kaggle T4 GPUs (demostrated in `uncompiled_kaggle_multigpu.ipynb`)
  2. In the compiled version because of the graph breaks the performance gets reduced. So for getting a compiled `no graph-break` model follow the instructions mentioned in `Task 3` and do the setup, if you want to have `no graph-break` model with FSDP2.
  2. GPU with SM 80 or higher is needed for the compiled no graph-break version (the uncompiled & compiled with graph break versions work on Tesla T4 GPUs). Code tested on RTX 3060 and RTX 4050. (demonstrated in `compiled_multipgpu.py`)
  
- **Software:**  
  Linux is recommended (Windows users might need to use WSL).

### Setup Options

#### Option 1: Create a New Virtual Environment

1. **Run the Setup Script**  
   Checkout `question 3 readme` for more info. Question 3 readme is useful for setting up the compiled no graph-break version of the model.

### Running the Script

For local multi-GPU execution, use the following command (ensure that your `config.yaml` is properly set for your GPU configuration):
```bash
accelerate launch --config_file "config.yaml" compiled_multigpu.py
```
Modify the number of GPUs in your configuration as needed.

### Kaggle Notebook

- Free Kaggle notebooks with 2× Tesla T4 GPUs have been set up for the uncompiled & compiled versions.  
- Refer to the notebooks for access to the Kaggle demo.

## Conclusion

This task demonstrates a robust solution to fine-tune a QLoRA model with FSDP2, leveraging state-of-the-art distributed training techniques and full transformers compatibility. Whether using the uncompiled version on accessible hardware like Tesla T4 GPUs or the torch-compiled version for advanced setups, the training loss remains consistent with single GPU training. This solution paves the way for efficient, multi-GPU fine-tuning with all FSDP2 features enabled.