
# This version will have graph breaks and recompilations -- if you want better performance follow task 3 and setup the environment first
# and then modify the script as per task 3 to obtain `no graph break and less recompiled` version.
# This is just a demo version which shows that the kaggle uncompiled version is compilable with proper setup.


# Importing Required Libraries

import os
import logging
from typing import Optional, Tuple
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch
import transformers.models.llama.modeling_llama
from transformers.models.llama.modeling_llama import Cache, Unpack, FlashAttentionKwargs, Callable, eager_attention_forward, apply_rotary_pos_emb, ALL_ATTENTION_FUNCTIONS, logger, BaseModelOutputWithPast, Union, DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.modeling_utils import PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType
import warnings
from accelerate import Accelerator
import time

# Setting Seed for Reproducibility

set_seed(42)


# Configuring Environment Variables and PyTorch Settings

# This script sets environment variables for optimizing PyTorch execution and debugging. It enables verbose logging, configures CUDA settings, and sets TorchInductor and TorchDynamo options for efficient model compilation.

# **Torch Compile Options:**
# Defines optimization settings like epilogue fusion, autotuning, and shape padding.

os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore")

torch._dynamo.config.suppress_errors = True

torch_compile_options = torch_compile_options = {
    "epilogue_fusion"   : True,
    "max_autotune"      : True,
    "shape_padding"     : True,
    "trace.enabled"     : True,
    "triton.cudagraphs" : False,
}


## Custom Gradient Modification in PyTorch

# This script defines a custom PyTorch operation `gradChanger` to modify tensor gradients, ensuring detached clones retain `requires_grad=True`.  

# - **`@torch.library.custom_op`**: Registers a custom operation (`noName::gradChanger`).  
# - **`@gradChanger.register_fake`**: Defines a fake implementation for testing.  
# - **`modified_enable_input_require_grads`**: Hooks into the forward pass of `PreTrainedModel` to apply `gradChanger` to input embeddings.  

@torch.library.custom_op("noName::gradChanger", mutates_args=())
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


## Compiling Almost Every LLaMA Model Components With `torch.compile`

# This script compiles different components of the LLaMA model using `torch.compile` for performance optimization.

#- **`compiled_llama_mlp`**: Optimized MLP forward pass.
#- **`compiled_llama_rms_norm`**: Efficient RMSNorm implementation.
#- **`compiled_llama_rotary_embedding`**: Computes rotary position embeddings.
#- **`compiled_llama_attention`**: Custom attention mechanism.
#- **`compiled_llama_decoder_layer`**: Processes each decoder layer.
#- **`compiled_llama_model`**: Full LLaMA forward pass with caching.
#- **`compiled_compute_loss`**: Computes cross-entropy loss for training.

# Each function is compiled with `torch.compile(fullgraph=True, dynamic=True, options=torch_compile_options)`, and replaces corresponding methods in `transformers.models.llama.modeling_llama`.


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def compiled_llama_mlp(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def compiled_llama_rms_norm(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)

@torch.no_grad()
@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def compiled_llama_rotary_embedding(self, x, position_ids):
    if "dynamic" in self.rope_type:
        self._dynamic_frequency_update(position_ids, device=x.device)
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
    cos = cos * self.attention_scaling
    sin = sin * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def compiled_llama_attention(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def compiled_llama_decoder_layer(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def compiled_llama_model(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def compiled_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

    kwargs = {
        "input_ids": inputs.data["input_ids"],
        "attention_mask": inputs.data["attention_mask"],
        "labels": inputs.data["labels"],
    }

    outputs = model(**kwargs)
    logits = outputs.logits  # shape: (B, S, V)
    
    # For causal language modeling, shift logits and labels so that
    # prediction at time t is compared with label at time t+1.
    shift_logits = logits[:, :-1, :]      # shape: (B, S-1, V)
    shift_labels = inputs["labels"][:, 1:]  # shape: (B, S-1)
    
    # Flatten the tensors for cross entropy: (B*(S-1), V) and (B*(S-1))
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100
    )
    return loss

transformers.models.llama.modeling_llama.LlamaMLP.forward = compiled_llama_mlp
transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = compiled_llama_rms_norm
transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward = compiled_llama_rotary_embedding
transformers.models.llama.modeling_llama.LlamaAttention.forward = compiled_llama_attention
transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = compiled_llama_decoder_layer
transformers.models.llama.modeling_llama.LlamaModel.forward = compiled_llama_model
SFTTrainer.compute_loss = compiled_compute_loss

# Accelerator Setup

def setup_distributed():
    if torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    if torch.distributed.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        # Use a list for device_ids as required.
        torch.distributed.barrier(device_ids=[local_rank])

def cleanup_distributed():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

setup_distributed()

accelerator = Accelerator()


## Setting Up unsloth/Meta-Llama-3.1-8B with LoRA and 4-bit Quantization (QLoRA)

model_name = "unsloth/Meta-Llama-3.1-8B"

# Set default data type and define quantization configuration.
torch.set_default_dtype(torch.float16)
dtype = torch.float16
bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = dtype,
    bnb_4bit_quant_storage    = torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation = "sdpa",
    quantization_config = bnb_config,
    device_map={"": accelerator.process_index}
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Configure LoRA for parameter-efficient fine-tuning.
lora_config = LoraConfig(
    r = 64,
    lora_alpha = 128,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    lora_dropout = 0,
    bias = "none",
    task_type = TaskType.CAUSAL_LM,
)

# Apply LoRA to the model and freeze non-LoRA parameters.
model = get_peft_model(model, lora_config)
with torch.no_grad():
    for name, param in model.named_parameters():
        if ".lora_A." in name or ".lora_B." in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

model.config.use_cache = False

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# Loading Dataset  

# - Fetches the `unified_chip2.jsonl` dataset from Hugging Face.  
# - Loads it as a JSON dataset.  
# - Uses only the first 10% of the training split.  

# Get dataset
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files = {"train" : url}, split = "train[:10%]")

# Timing Callback for Separate Warmup and Main Training Runs
# In this cell, we define a custom TrainerCallback called `TimingCallback`.  
#- **Warmup Steps:** The first three steps (where recompilation occurs) are timed separately.  
#- **Main Training Steps:** The remaining steps are timed separately.  

# When added to SFTTrainer, this callback will print the time for each step and summarize the total times at the end of training.

class TimingCallback(TrainerCallback):
    def __init__(self, warmup_steps: int = 3):
        self.warmup_steps = warmup_steps
        self.warmup_time = 0.0
        self.main_time = 0.0

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Record the start time of the step.
        self.step_start = time.time()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        step_time = time.time() - self.step_start
        # If the current step is within the warmup steps, add time to warmup_time.
        if state.global_step <= self.warmup_steps:
            self.warmup_time += step_time
        else:
            self.main_time += step_time

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        total_steps = state.global_step
        main_steps = total_steps - self.warmup_steps
        print(f"\nTotal warmup time for {self.warmup_steps} steps: {self.warmup_time:.4f} seconds")
        print(f"Total main training time for {main_steps} steps: {self.main_time:.4f} seconds")


# Training Configuration  

# - Uses `SFTTrainer` to fine-tune the model.  
# - Loads dataset and tokenizer for training.  
# - Training settings:  
#   - Batch size: 1  
#   - Gradient accumulation: 2 steps  
#   - Warmup: 3 steps  
#   - Max steps: 100  
#   - Logs every step  
#   - Outputs saved to `"outputs"`  
#   - Uses FP16 or BF16 based on model dtype  
# - Starts training with `.train()`  

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    processing_class = tokenizer,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 3,
        max_steps = 100,
        logging_steps = 1,
        output_dir = "outputs",
        seed = 3407,
        max_seq_length = 2048,
        fp16 = model.get_input_embeddings().weight.dtype == torch.float16,
        bf16 = model.get_input_embeddings().weight.dtype == torch.bfloat16,
        report_to = "none", # For W&B
        dataset_num_proc = 4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        label_names = ["input_ids", "labels", "attention_mask"]
    ),
    callbacks = [TimingCallback(warmup_steps=3)]
)

accelerator.print(f"Model Summary:\n{trainer.model}")
    
# Optionally print trainable parameters if the method is available.
if hasattr(trainer.model, "print_trainable_parameters"):
    trainer.model.print_trainable_parameters()

# Begin training.
trainer.train()

# Clean up distributed resources.
cleanup_distributed()
