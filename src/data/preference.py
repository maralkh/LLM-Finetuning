"""Preference datasets for DPO training."""

import random
from typing import Iterator, Optional, Callable, Any
from dataclasses import dataclass

import torch
from tqdm import tqdm

from .base import BasePreferenceDataset, PreferenceExample, BaseDataset, RLPrompt


class PreferenceDataset(BasePreferenceDataset):
    """
    Generic preference dataset from a list of examples.
    """
    
    def __init__(self, examples: list[PreferenceExample], name: str = "preference"):
        self._examples = examples
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    def __len__(self) -> int:
        return len(self._examples)
    
    def __iter__(self) -> Iterator[PreferenceExample]:
        return iter(self._examples)


class HFPreferenceDataset(BasePreferenceDataset):
    """
    Load preference dataset from HuggingFace.
    
    Supports common formats like:
    - Anthropic HH-RLHF
    - UltraFeedback
    - Custom datasets with chosen/rejected columns
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        prompt_column: str = "prompt",
        chosen_column: str = "chosen",
        rejected_column: str = "rejected",
    ):
        """
        Initialize from HuggingFace dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split
            max_samples: Maximum samples to load
            prompt_column: Column name for prompts
            chosen_column: Column name for chosen responses
            rejected_column: Column name for rejected responses
        """
        from datasets import load_dataset
        
        self._name = dataset_name.split("/")[-1]
        
        dataset = load_dataset(dataset_name, split=split)
        
        self._examples = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            
            self._examples.append(PreferenceExample(
                prompt=item[prompt_column],
                chosen=item[chosen_column],
                rejected=item[rejected_column],
                id=f"{self._name}_{i}",
            ))
    
    @property
    def name(self) -> str:
        return self._name
    
    def __len__(self) -> int:
        return len(self._examples)
    
    def __iter__(self) -> Iterator[PreferenceExample]:
        return iter(self._examples)


def generate_preferences_rejection_sampling(
    model,
    tokenizer,
    dataset: BaseDataset,
    reward_fn: Callable[[str, str, dict], float],
    num_samples: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    max_examples: Optional[int] = None,
    margin_threshold: float = 0.0,
    batch_size: int = 4,
    show_progress: bool = True,
) -> list[PreferenceExample]:
    """
    Generate preference pairs via rejection sampling.
    
    For each prompt:
    1. Generate N responses
    2. Score each with reward function
    3. Create preference pair from highest/lowest scoring
    
    Args:
        model: Model for generation
        tokenizer: Tokenizer
        dataset: Dataset with prompts
        reward_fn: Function (prompt, response, metadata) -> score
        num_samples: Samples per prompt
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        max_examples: Maximum preference pairs to generate
        margin_threshold: Minimum score difference for valid pair
        batch_size: Batch size for generation
        show_progress: Show progress bar
        
    Returns:
        List of PreferenceExample
    """
    model.eval()
    
    rl_prompts = dataset.get_rl_prompts(limit=max_examples)
    preference_examples = []
    
    iterator = rl_prompts
    if show_progress:
        iterator = tqdm(rl_prompts, desc="Generating preferences")
    
    for rl_prompt in iterator:
        prompt = rl_prompt.prompt
        metadata = rl_prompt.metadata
        
        # Generate multiple responses
        inputs = tokenizer(
            [prompt] * num_samples,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode responses
        input_len = inputs["input_ids"].shape[1]
        responses = []
        for output in outputs:
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            responses.append(response)
        
        # Score responses
        scored = []
        for response in responses:
            score = reward_fn(prompt, response, metadata)
            scored.append((response, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Get best and worst
        best_response, best_score = scored[0]
        worst_response, worst_score = scored[-1]
        
        # Check margin
        margin = best_score - worst_score
        if margin >= margin_threshold:
            preference_examples.append(PreferenceExample(
                prompt=prompt,
                chosen=best_response,
                rejected=worst_response,
                id=rl_prompt.id,
                metadata={
                    **metadata,
                    "chosen_score": best_score,
                    "rejected_score": worst_score,
                    "margin": margin,
                },
                margin=margin,
            ))
    
    model.train()
    
    return preference_examples


def generate_preferences_best_of_n(
    model,
    tokenizer,
    dataset: BaseDataset,
    reward_fn: Callable[[str, str, dict], float],
    n: int = 8,
    num_pairs_per_prompt: int = 1,
    **kwargs,
) -> list[PreferenceExample]:
    """
    Generate preference pairs using best-of-n sampling.
    
    Similar to rejection sampling but can generate multiple pairs
    per prompt by pairing different responses.
    
    Args:
        model: Model for generation
        tokenizer: Tokenizer
        dataset: Dataset with prompts
        reward_fn: Reward function
        n: Number of samples per prompt
        num_pairs_per_prompt: Number of preference pairs per prompt
        **kwargs: Additional args for generation
        
    Returns:
        List of PreferenceExample
    """
    model.eval()
    
    rl_prompts = dataset.get_rl_prompts(limit=kwargs.get("max_examples"))
    preference_examples = []
    
    for rl_prompt in tqdm(rl_prompts, desc="Best-of-N sampling"):
        prompt = rl_prompt.prompt
        metadata = rl_prompt.metadata
        
        # Generate N responses
        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 512),
                temperature=kwargs.get("temperature", 0.8),
                do_sample=True,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        input_len = inputs["input_ids"].shape[1]
        
        # Score all responses
        scored = []
        for output in outputs:
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            score = reward_fn(prompt, response, metadata)
            scored.append((response, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Create pairs
        for i in range(min(num_pairs_per_prompt, len(scored) // 2)):
            chosen_response, chosen_score = scored[i]
            rejected_response, rejected_score = scored[-(i + 1)]
            
            if chosen_score > rejected_score:
                preference_examples.append(PreferenceExample(
                    prompt=prompt,
                    chosen=chosen_response,
                    rejected=rejected_response,
                    id=f"{rl_prompt.id}_pair{i}",
                    metadata={
                        **metadata,
                        "chosen_score": chosen_score,
                        "rejected_score": rejected_score,
                    },
                    margin=chosen_score - rejected_score,
                ))
    
    model.train()
    
    return preference_examples


def filter_preferences(
    examples: list[PreferenceExample],
    min_margin: float = 0.0,
    max_length: Optional[int] = None,
    tokenizer=None,
) -> list[PreferenceExample]:
    """
    Filter preference examples.
    
    Args:
        examples: List of preference examples
        min_margin: Minimum margin between chosen/rejected scores
        max_length: Maximum combined length (requires tokenizer)
        tokenizer: Tokenizer for length filtering
        
    Returns:
        Filtered list
    """
    filtered = []
    
    for ex in examples:
        # Check margin
        if ex.margin is not None and ex.margin < min_margin:
            continue
        
        # Check length
        if max_length and tokenizer:
            full_chosen = ex.prompt + ex.chosen
            full_rejected = ex.prompt + ex.rejected
            
            chosen_len = len(tokenizer.encode(full_chosen, add_special_tokens=False))
            rejected_len = len(tokenizer.encode(full_rejected, add_special_tokens=False))
            
            if chosen_len > max_length or rejected_len > max_length:
                continue
        
        filtered.append(ex)
    
    return filtered
