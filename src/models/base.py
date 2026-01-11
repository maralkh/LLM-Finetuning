"""Base model interface and configuration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[list[str]] = None  # Auto-detect if None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class ModelConfig:
    """Model loading configuration."""
    name: str  # HuggingFace name or local path
    dtype: str = "bfloat16"  # bfloat16, float16, float32
    use_lora: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    
    # Memory optimization
    gradient_checkpointing: bool = True
    
    # For continuing training
    resume_from: Optional[str] = None  # Path to checkpoint
    
    # Trust remote code (for some models)
    trust_remote_code: bool = True
    
    # Attention implementation
    attn_implementation: Optional[str] = None  # "flash_attention_2", "sdpa", None


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = -1
    do_sample: bool = True
    num_return_sequences: int = 1


class BaseModel(ABC):
    """Abstract base for trainable models."""
    
    @property
    @abstractmethod
    def model(self):
        """Return the underlying model."""
        pass
    
    @property
    @abstractmethod
    def tokenizer(self):
        """Return the tokenizer."""
        pass
    
    @property
    @abstractmethod
    def config(self) -> ModelConfig:
        """Return model configuration."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model checkpoint (LoRA weights if using LoRA)."""
        pass
    
    @abstractmethod
    def merge_and_save(self, path: str):
        """Merge LoRA weights and save full model."""
        pass
    
    def generate(
        self,
        prompts: list[str],
        gen_config: Optional[GenerationConfig] = None,
    ) -> list[list[str]]:
        """
        Generate responses for prompts.
        
        Args:
            prompts: List of input prompts
            gen_config: Generation configuration
            
        Returns:
            List of response lists (one list per prompt)
        """
        gen_config = gen_config or GenerationConfig()
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_config.max_new_tokens,
                temperature=gen_config.temperature if gen_config.do_sample else 1.0,
                top_p=gen_config.top_p if gen_config.do_sample else 1.0,
                top_k=gen_config.top_k if gen_config.top_k > 0 else None,
                do_sample=gen_config.do_sample,
                num_return_sequences=gen_config.num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode (remove prompt)
        input_len = inputs["input_ids"].shape[1]
        results = []
        
        for i in range(len(prompts)):
            prompt_results = []
            start_idx = i * gen_config.num_return_sequences
            end_idx = start_idx + gen_config.num_return_sequences
            
            for j in range(start_idx, end_idx):
                response = self.tokenizer.decode(
                    outputs[j][input_len:],
                    skip_special_tokens=True,
                )
                prompt_results.append(response)
            
            results.append(prompt_results)
        
        return results


# Import torch here to avoid issues if not installed
try:
    import torch
except ImportError:
    torch = None
