# nf4-to-Triton

An efficient Triton kernel implementation for dequantizing nf4 quantized tensors into fp16 or bf16 formats. This implementation fuses the double dequantization steps (computing the absolute maximum and forming the final weights) into a single Triton kernel. The solution outperforms Unsloth’s fast_dequantize by an average factor of ~1.22× and is fully compatible with `torch.compile`.

## Overview

This project implements a high-performance conversion of nf4 quantized tensors into floating-point representations (fp16 or bf16) using a custom Triton kernel. Key highlights include:

- **Single-Kernel Double Dequantization:** Both the dequantization of the absolute maximum (absmax) and weight formation are computed in one kernel.
- **Optimized Performance:** Benchmarked speedup ratios compared to Unsloth’s fast_dequantize are on average 1.218×, with individual runs showing improvements from 1.145× to 1.417×.
- **Memory Efficiency:** No large intermediate memory buffers are used.
- **Torch.compile Compatibility:** The Triton kernel works seamlessly with `torch.compile`, aiding debugging and performance tuning.
- **Targeted for Tesla T4:** The kernel is specifically tuned for the Tesla T4 GPU architecture.

## Features

- **Fused Kernel:** Performs both absmax and weight dequantization in a single Triton kernel launch.
- **High Speedup:** Demonstrates a performance boost of more than 1.15× (up to 1.42× in some runs) over previous implementations.
- **Dual Precision Support:** Supports conversion into fp16 and bf16, verified through tests.
- **Optimized for Tesla T4:** Specially optimized for Tesla T4 GPUs to achieve maximum throughput.
- **torch.compile Compatible:** Ensures compatibility with PyTorch's compilation toolchain, verified via tests.
- **Custom Assembly & Cache Eviction:** (Optional) Hooks for integrating custom assembly routines and cache eviction strategies for further performance gains.

## Performance Tests

The ratio of execution times (Unsloth’s vs. Custom Code) across multiple runs. Sample output:

```
1.1454133907231538
1.1647778856211157
1.417303095420293
1.1747517897251316
1.1881643938005837
Ratio Of Speed between Unsloth & Custom Code Is 1.2180821110580555
```

## Implementation Details

### Kernel Components

- **lookup_const Function:**  
  A helper function that maps nibble values (0–15) to their corresponding floating-point constants using Triton’s elementwise operations.

- **_your_dequantize_nf4_kernel:**  
  The main Triton kernel that:
  - Splits each weight value into high and low nibbles.
  - Converts these nibbles into fp32 constants via `lookup_const`.
  - Loads the corresponding absmax values, decodes them using a code lookup, and computes a scaling factor.
  - Applies the scaling factor to the high and low values.
  - Interleaves the dequantized values and stores the result.

- **Wrapper Functions:**  
  - `_your_dequantize_nf4`: Prepares the output tensor and launches the Triton kernel.
  - `your_dequantize_nf4`: Exposes the dequantization functionality by accessing the nf4 tensor’s quantization state.

### Kernel Configuration

- **BLOCK_SIZE:**  
  Set to 1024 for optimized parallel processing.
  
- **Grid Configuration:**  
  The grid is calculated based on the total number of elements ensuring full coverage of the tensor.

## Performance and Testing

- **Benchmarking:**  
  The implementation is benchmarked against Unsloth’s fast_dequantize. The tests report speedup ratios ranging from ~1.145× to ~1.417× with an average of approximately 1.218× speedup.

- **Torch.compile Compatibility:**  
  The solution has been verified to work with `torch.compile`, providing an additional layer of optimization and compatibility.

- **Precision Verification:**  
  Tests have been conducted in fp16 & bf16 modes to ensure correctness across different precision formats. Devices with SM ≥ 80 support dequantization to both fp16 and bf16. Tesla T4 (SM 75) only supports fp16 because bf16 with Triton triggers errors, even though bf16 works in eager mode. The solution is verified on RTX 3060 and RTX 4050 for both formats, while T4 remains limited to fp16.

## Acknowledgements

- **Unsloth:** For the inspiration and baseline fast_dequantize function.
- **bitsandbytes:** For their dequantize_blockwise implementation, which guided parts of this project.
- **Triton & PyTorch Communities:** For providing powerful tools and frameworks that make such optimizations possible.