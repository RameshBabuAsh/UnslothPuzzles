# Memory Efficient Backprop for Large Language Models

This repository demonstrates a memory-efficient approach to backpropagation for large language models. The primary challenge we address is the high VRAM usage when computing logits for large vocabularies. In standard implementations, materializing these logits leads to significant memory spikes, which can hinder training, especially on devices with limited VRAM.

## Problem Description

Large language models often use a final projection layer to calculate the logits for predicting the next token. When the vocabulary is very large (for example, 128K tokens), the intermediate tensors generated during this process consume a lot of memory. This becomes a critical issue during both the forward pass and the backpropagation phase, where gradients need to be stored.

The solution tackles this problem by processing the inputs in smaller batches rather than materializing the full tensor at once. This method computes the necessary outputs on the fly during both the forward and backward passes, leading to a significant reduction in VRAM usage.

## Sources & Acknowledgments

- **Memory Efficient Linear:**  
  The core idea behind the memory-efficient linear layer is based on the concepts presented in [this Medium article](https://medium.com/@yash9439/unslothais-innovative-hiring-challenge-memory-efficient-backprop-a5dc2372d469). I adapted the approach from the article to suit our needs.

- **Code Tweaks for Compatibility:**  
  The code provided in the article served as the core. I made minor modifications to ensure compatibility with both the Llama model and the GRPO implementation. These changes were minimal and focused solely on integration.

- **GRPO Sample Implementation:**  
  The GRPO sample implementation is based on the Unsloth GRPO Colab notebook. I integrated the custom memory-efficient linear layer into the notebook, and aside from this modification, the implementation remains as originally provided.

## Results

### Cross Entropy (CE) Loss
- **Standard Loss (CE):** 9.875000  
- **Custom Loss (CE):** 9.850885  
- **Loss Difference:** 0.024115  

- **Standard Time (CE):** 1.023671 seconds  
- **Custom Time (CE):** 1.390536 seconds  

- **Standard VRAM (CE):** 4.063050 GiB  
- **Custom VRAM (CE):** 1.760712 GiB  

- **Input Gradient Comparison (CE):**  
  Maximum difference between standard and custom input gradients: 7.450580596923828e-09

### MSE Loss Test (with ReLU activation)
- **Standard MSE Loss:** 1.166796  
- **Custom MSE Loss:** 1.333750  
- **Loss Difference:** 0.166953

### Llama Model Compatibility

#### New Linear in Llama 1B Model
- **Training Loss:** Approximately 4.111236  
- **Metrics:** Included runtime, samples per second, steps per second, and total floating point operations indicate that our custom linear layer achieves performance on par with the standard implementation.

#### Standard nn.Linear in Llama 1B Model
- **Training Loss:** Approximately 4.111237  

The tests confirm that the custom memory-efficient linear layer is fully compatible with the Llama model and yields equivalent loss outcomes.

### GRPO Compatibility

In GRPO experiments, I used the unsloth patched Llama 8B Instruct model along with the custom patched memory-efficient linear layer. The results remained consistent, confirming that the approach is compatible with GRPO as well.

## Conclusion

This repository provides a proof-of-concept for reducing VRAM usage during training by computing logits on the fly. The custom approach achieves nearly identical loss values compared to the standard method while significantly lowering memory consumption. Although the custom method requires a slight increase in processing time, it offers a valuable solution for training large language models on memory-constrained hardware.