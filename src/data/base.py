"""Base data structures for training datasets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Iterator


@dataclass
class SFTExample:
    """
    Single supervised finetuning example.
    
    The model learns to generate `response` given `prompt`.
    """
    prompt: str
    response: str
    id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def to_text(self, template: str = "default") -> str:
        """Convert to full text for training."""
        return self.prompt + self.response


@dataclass
class PreferenceExample:
    """
    Preference pair for DPO training.
    
    The model learns to prefer `chosen` over `rejected`.
    """
    prompt: str
    chosen: str
    rejected: str
    id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    # Optional: margin for preference (higher = stronger preference)
    margin: Optional[float] = None


@dataclass
class RLPrompt:
    """
    Prompt for RL training (PPO, GRPO).
    
    The model generates responses, which are scored by reward function.
    Metadata contains info needed for reward computation (e.g., gold_answer).
    """
    prompt: str
    id: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    @property
    def gold_answer(self) -> Any:
        """Get gold answer for reward computation."""
        return self.metadata.get("gold_answer")


@dataclass
class DatasetConfig:
    """Configuration for loading a dataset."""
    name: str  # Dataset identifier
    split: str = "train"
    max_samples: Optional[int] = None
    seed: int = 42
    
    # Prompt template
    template: str = "default"
    
    # For code datasets
    include_tests_in_prompt: bool = False


class BaseDataset(ABC):
    """Abstract base for training datasets."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset identifier."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Number of examples."""
        pass
    
    @abstractmethod
    def __iter__(self) -> Iterator:
        """Iterate over examples."""
        pass
    
    def get_sft_examples(self, limit: Optional[int] = None) -> list[SFTExample]:
        """
        Get examples formatted for SFT.
        
        Args:
            limit: Maximum number of examples
            
        Returns:
            List of SFTExample
        """
        examples = list(self)
        if limit:
            examples = examples[:limit]
        return examples
    
    def get_rl_prompts(self, limit: Optional[int] = None) -> list[RLPrompt]:
        """
        Get prompts for RL training.
        
        Default implementation extracts prompt from SFT examples.
        Override for custom behavior.
        
        Args:
            limit: Maximum number of prompts
            
        Returns:
            List of RLPrompt
        """
        sft_examples = self.get_sft_examples(limit)
        return [
            RLPrompt(
                prompt=ex.prompt,
                id=ex.id,
                metadata=ex.metadata,
            )
            for ex in sft_examples
        ]


class BasePreferenceDataset(ABC):
    """Abstract base for preference datasets (DPO)."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset identifier."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Number of preference pairs."""
        pass
    
    @abstractmethod
    def __iter__(self) -> Iterator[PreferenceExample]:
        """Iterate over preference pairs."""
        pass
    
    def get_examples(self, limit: Optional[int] = None) -> list[PreferenceExample]:
        """Get preference examples."""
        examples = list(self)
        if limit:
            examples = examples[:limit]
        return examples
