"""Data collators for training."""

from dataclasses import dataclass
from typing import Any, Optional, Union
import torch
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForSFT:
    """
    Data collator for supervised finetuning.
    
    Handles tokenization and padding for SFT examples.
    """
    
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    padding: Union[bool, str] = True
    return_tensors: str = "pt"
    
    # Label masking
    mask_prompt: bool = True  # Don't compute loss on prompt tokens
    label_pad_token_id: int = -100
    
    def __call__(self, examples: list[dict]) -> dict:
        """
        Collate examples into a batch.
        
        Args:
            examples: List of dicts with 'text' or 'prompt'+'response' keys
            
        Returns:
            Batch dict with input_ids, attention_mask, labels
        """
        # Get full texts
        texts = []
        prompt_lengths = []
        
        for ex in examples:
            if "text" in ex:
                # Pre-formatted text
                text = ex["text"]
                # Try to detect prompt length (hacky, but works for most templates)
                prompt_len = ex.get("prompt_length", 0)
            else:
                # Separate prompt and response
                text = ex["prompt"] + ex["response"]
                prompt_len = len(self.tokenizer.encode(
                    ex["prompt"], add_special_tokens=False
                ))
            
            texts.append(text)
            prompt_lengths.append(prompt_len)
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors=self.return_tensors,
        )
        
        # Create labels
        labels = tokenized["input_ids"].clone()
        
        # Mask prompt tokens and padding
        if self.mask_prompt:
            for i, prompt_len in enumerate(prompt_lengths):
                if prompt_len > 0:
                    labels[i, :prompt_len] = self.label_pad_token_id
        
        # Mask padding
        labels[tokenized["attention_mask"] == 0] = self.label_pad_token_id
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }


@dataclass
class DataCollatorForDPO:
    """
    Data collator for DPO training.
    
    Prepares chosen and rejected pairs.
    """
    
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    max_prompt_length: int = 512
    return_tensors: str = "pt"
    label_pad_token_id: int = -100
    
    def __call__(self, examples: list[dict]) -> dict:
        """
        Collate preference pairs.
        
        Args:
            examples: List of dicts with 'prompt', 'chosen', 'rejected'
            
        Returns:
            Batch dict for DPO training
        """
        prompts = []
        chosen_responses = []
        rejected_responses = []
        
        for ex in examples:
            prompts.append(ex["prompt"])
            chosen_responses.append(ex["chosen"])
            rejected_responses.append(ex["rejected"])
        
        # Tokenize prompts
        prompt_tokens = self.tokenizer(
            prompts,
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=True,
        )
        
        # Build full sequences
        batch = self._build_sequences(
            prompts, chosen_responses, rejected_responses, prompt_tokens
        )
        
        return batch
    
    def _build_sequences(
        self,
        prompts: list[str],
        chosen: list[str],
        rejected: list[str],
        prompt_tokens: dict,
    ) -> dict:
        """Build tokenized sequences for DPO."""
        
        chosen_input_ids = []
        chosen_attention_mask = []
        chosen_labels = []
        
        rejected_input_ids = []
        rejected_attention_mask = []
        rejected_labels = []
        
        for i, (prompt, c, r) in enumerate(zip(prompts, chosen, rejected)):
            prompt_len = len(prompt_tokens["input_ids"][i])
            
            # Tokenize chosen
            chosen_full = prompt + c
            chosen_tok = self.tokenizer(
                chosen_full,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            # Tokenize rejected
            rejected_full = prompt + r
            rejected_tok = self.tokenizer(
                rejected_full,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
            
            # Create labels (mask prompt)
            chosen_lab = chosen_tok["input_ids"].clone()
            chosen_lab[:, :prompt_len] = self.label_pad_token_id
            
            rejected_lab = rejected_tok["input_ids"].clone()
            rejected_lab[:, :prompt_len] = self.label_pad_token_id
            
            chosen_input_ids.append(chosen_tok["input_ids"].squeeze(0))
            chosen_attention_mask.append(chosen_tok["attention_mask"].squeeze(0))
            chosen_labels.append(chosen_lab.squeeze(0))
            
            rejected_input_ids.append(rejected_tok["input_ids"].squeeze(0))
            rejected_attention_mask.append(rejected_tok["attention_mask"].squeeze(0))
            rejected_labels.append(rejected_lab.squeeze(0))
        
        # Pad to same length
        chosen_input_ids = self._pad_sequences(chosen_input_ids)
        chosen_attention_mask = self._pad_sequences(chosen_attention_mask, pad_value=0)
        chosen_labels = self._pad_sequences(chosen_labels, pad_value=self.label_pad_token_id)
        
        rejected_input_ids = self._pad_sequences(rejected_input_ids)
        rejected_attention_mask = self._pad_sequences(rejected_attention_mask, pad_value=0)
        rejected_labels = self._pad_sequences(rejected_labels, pad_value=self.label_pad_token_id)
        
        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels,
        }
    
    def _pad_sequences(
        self,
        sequences: list[torch.Tensor],
        pad_value: int = 0,
    ) -> torch.Tensor:
        """Pad sequences to same length."""
        max_len = max(s.size(0) for s in sequences)
        padded = []
        
        for seq in sequences:
            if seq.size(0) < max_len:
                padding = torch.full(
                    (max_len - seq.size(0),),
                    pad_value,
                    dtype=seq.dtype,
                )
                seq = torch.cat([seq, padding])
            padded.append(seq)
        
        return torch.stack(padded)


@dataclass
class DataCollatorForRL:
    """
    Data collator for RL training (PPO, GRPO).
    
    Only tokenizes prompts (responses generated during training).
    """
    
    tokenizer: PreTrainedTokenizer
    max_length: int = 512  # Prompt max length
    return_tensors: str = "pt"
    
    def __call__(self, examples: list[dict]) -> dict:
        """
        Collate prompts for RL.
        
        Args:
            examples: List of dicts with 'prompt' key
            
        Returns:
            Batch dict with tokenized prompts
        """
        prompts = [ex["prompt"] for ex in examples]
        
        tokenized = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors=self.return_tensors,
        )
        
        # Include metadata for reward computation
        metadata = [ex.get("metadata", {}) for ex in examples]
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "metadata": metadata,
        }
