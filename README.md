# UnslothPuzzles

This repository contains my open‐source solutions to Daniel Han's Unsloth challenges—five tasks that focus on advanced deep learning optimizations and efficiency improvements. Each challenge has its own folder with a detailed README explaining the approach and providing the source code:

- **Convert nf4 to Triton:** Converting 4-bit quantization (nf4) operations into efficient Triton kernels.
- **Make QLoRA work with FSDP2:** Enabling QLoRA fine-tuning with Fully Sharded Data Parallel (FSDP2).
- **Torch.compile without Graph Breaks for QLoRA:** Removing graph breaks when compiling QLoRA models.
- **Memory Efficient Backprop:** Implementing backpropagation algorithms that drastically reduce memory usage.

Working through these challenges has deepened my understanding of PyTorch, quantization (BitsAndBytes), FSDP2, and torch.compile—and it even introduced me to writing my first code in Triton!

Feel free to explore and improve upon these solutions.

**For Unsloth People** : I am Ramesh Babu, a third-year B.Tech CSE student at Indian Institute of Information Technology, Design and Manufacturing, Jabalpur, India, with a keen interest in Artificial Intelligence. 

## Challenge Progress:

- Part A:
  - [x] Single triton kernel (+3)
  - Speedup checks:
    - [ ] If speedup <= 1.00 (-3)
    - [x] If speedup >= 1.05 (+1)
    - [x] If speedup >= 1.10 (+2)
    - [x] If speedup >= 1.15 (+2)
  - [x] Kernel works in torch compile (+1)
    - [ ] If not (-1)
  - [ ] Custom ASM works (+3)
  - [ ] Uses cache eviction (+1)
  - [x] Tested in f16 and bf16 (+1) (In GPUs with SM >= 80, f16 works on colab & kaggle free GPUs as well)
    - [ ] If not (-1)


- Part B:
  - FSDP2 works with QLoRA:
    - [x] With torch compile (+5)
    - [ ] Without torch compile (+3)
    - [ ] Uses part A and single kernel and faster (+3)
    - Uses torchAO:
      - [ ] If torchAO slower than BnB (-3)
  - TP or PP with QLoRA:
    - [ ] With zero bubble (+3)
    - [ ] Without zero bubble (+2)
  - [ ] FSDP1 works with QLoRA (+1)
  - [x] Kaggle notebook 2 tesla t4 example (+2)
    - [ ] If not (score = 0)
  - [ ] If not attempted (-2)


- Part C:
  - Uses flex attention:
    - [ ] Dynamic sequence length works (+3)
    - [ ] If not (+1)
  - [ ] No torch compile BnB (-2)
  - [ ] Use part A (+1)
  - [x] Torch compile BnB (+1)
  - Attention compiled:
    - [ ] With excessive recompilation (-3)
    - [x] Without excessive recompilation (+2)
  - MLP compiled:
    - [ ] With excessive recompilation (-3)
    - [x] Without excessive recompilation (+1)
  - [ ] Loss not compiled (-1)
  - [ ] Layernorms not compiled (-3)
  - Max autotune triton matmul:
    - [ ] With excessive recompilation (-2)
    - [x] Without excessive recompilation (+2)
  - [ ] If not attempted (-1)


- Part E:
  - [x] VRAM 50% reduction (+2)
  - [ ] Remove float32 upcast (score = 0)
  - [x] Show CE loss works (+1)
  - [x] Show other functions work (+1)
  - [ ] Hardcoded gradients (score = 0)
  - [x] Allows dynamic chunk sizes (+1)
  - [x] Llama 1B training loss matches (+1)
    - [ ] If not (score = 0)
  - [x] GRPO memory efficient linear works (+4)
 
Next Steps: Over the weekend, I'll either work on refining the four current solutions further or tackle an additional challenge in task D :)

Thanks for the template @parnox/unsloth-notes
