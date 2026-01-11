"""Model loading utilities with LoRA support."""

from typing import Optional
from pathlib import Path
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)

from .base import BaseModel, ModelConfig, LoRAConfig, GenerationConfig


# Default LoRA target modules for common architectures
DEFAULT_TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "default": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def _get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def _detect_target_modules(model_name: str) -> list[str]:
    """Detect appropriate LoRA target modules based on model architecture."""
    model_lower = model_name.lower()
    
    for arch, modules in DEFAULT_TARGET_MODULES.items():
        if arch in model_lower:
            return modules
    
    return DEFAULT_TARGET_MODULES["default"]


class TrainableModel(BaseModel):
    """
    Model wrapper for training with optional LoRA.
    
    Handles:
    - Model loading with appropriate dtype
    - LoRA adapter setup
    - Checkpoint saving/loading
    - Generation utilities
    """
    
    def __init__(
        self,
        config: ModelConfig,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ):
        """
        Initialize trainable model.
        
        Use load_model_for_training() factory function instead of direct init.
        """
        self._config = config
        self._model = model
        self._tokenizer = tokenizer
    
    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def config(self) -> ModelConfig:
        return self._config
    
    def save(self, path: str):
        """Save model checkpoint."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self._config.use_lora:
            # Save only LoRA weights
            self._model.save_pretrained(save_path)
        else:
            # Save full model
            self._model.save_pretrained(save_path)
        
        # Always save tokenizer
        self._tokenizer.save_pretrained(save_path)
        
        print(f"Model saved to {save_path}")
    
    def merge_and_save(self, path: str):
        """Merge LoRA weights into base model and save."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self._config.use_lora:
            # Merge LoRA weights
            merged_model = self._model.merge_and_unload()
            merged_model.save_pretrained(save_path)
        else:
            self._model.save_pretrained(save_path)
        
        self._tokenizer.save_pretrained(save_path)
        print(f"Merged model saved to {save_path}")


def load_model_for_training(config: ModelConfig) -> TrainableModel:
    """
    Load model for training with optional LoRA.
    
    Args:
        config: Model configuration
        
    Returns:
        TrainableModel ready for training
    """
    print(f"Loading model: {config.name}")
    
    # Determine dtype
    dtype = _get_dtype(config.dtype)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.name,
        trust_remote_code=config.trust_remote_code,
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Model loading kwargs
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": config.trust_remote_code,
        "device_map": "auto",
    }
    
    # Add attention implementation if specified
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        **model_kwargs,
    )
    
    # Enable gradient checkpointing if requested
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    # Apply LoRA if requested
    if config.use_lora:
        # Determine target modules
        target_modules = config.lora.target_modules
        if target_modules is None:
            target_modules = _detect_target_modules(config.name)
        
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=target_modules,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load from checkpoint if specified
    if config.resume_from:
        print(f"Loading checkpoint from {config.resume_from}")
        if config.use_lora:
            model = PeftModel.from_pretrained(
                model.base_model.model if hasattr(model, 'base_model') else model,
                config.resume_from,
            )
        else:
            # Load state dict for full model
            state_dict = torch.load(
                Path(config.resume_from) / "pytorch_model.bin",
                map_location="auto",
            )
            model.load_state_dict(state_dict)
    
    return TrainableModel(config, model, tokenizer)


def load_model_for_inference(
    path: str,
    base_model: Optional[str] = None,
    dtype: str = "bfloat16",
    is_lora: bool = True,
) -> TrainableModel:
    """
    Load a saved model for inference.
    
    Args:
        path: Path to saved model/adapter
        base_model: Base model name (required if loading LoRA adapter)
        dtype: Data type
        is_lora: Whether the checkpoint is LoRA adapter
        
    Returns:
        TrainableModel for inference
    """
    torch_dtype = _get_dtype(dtype)
    
    if is_lora:
        if base_model is None:
            raise ValueError("base_model required when loading LoRA adapter")
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, path)
        
        tokenizer = AutoTokenizer.from_pretrained(path)
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    config = ModelConfig(name=path, dtype=dtype, use_lora=False)
    return TrainableModel(config, model, tokenizer)
