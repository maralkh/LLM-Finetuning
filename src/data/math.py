"""Math datasets for training: GSM8K, MATH, MetaMathQA."""

import re
from typing import Iterator, Optional, Any

from .base import BaseDataset, SFTExample, RLPrompt, DatasetConfig
from .templates import get_template, MATH_COT_TEMPLATE


class GSM8KDataset(BaseDataset):
    """
    GSM8K: Grade School Math 8K
    
    Contains ~7.5K training and ~1.3K test problems.
    Good starting point for math reasoning.
    """
    
    def __init__(
        self,
        split: str = "train",
        template: str = "math_cot",
        max_samples: Optional[int] = None,
    ):
        """
        Initialize GSM8K dataset.
        
        Args:
            split: Dataset split (train or test)
            template: Prompt template name
            max_samples: Maximum samples to load
        """
        from datasets import load_dataset
        
        self.split = split
        self.template = get_template(template)
        
        # Load from HuggingFace
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        self._data = list(dataset)
        
        if max_samples:
            self._data = self._data[:max_samples]
    
    @property
    def name(self) -> str:
        return "gsm8k"
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[SFTExample]:
        for i, item in enumerate(self._data):
            # Extract numeric answer from solution
            # GSM8K format: "#### 42" at the end
            solution = item["answer"]
            answer_match = re.search(r"####\s*(-?\d+)", solution)
            gold_answer = int(answer_match.group(1)) if answer_match else None
            
            # Format prompt and response
            prompt = self.template.format_prompt(item["question"])
            response = self.template.format_response(solution)
            
            yield SFTExample(
                prompt=prompt,
                response=response,
                id=f"gsm8k_{self.split}_{i}",
                metadata={
                    "gold_answer": gold_answer,
                    "question": item["question"],
                    "full_solution": solution,
                }
            )
    
    def get_rl_prompts(self, limit: Optional[int] = None) -> list[RLPrompt]:
        """Get prompts for RL training with gold answers."""
        examples = list(self)
        if limit:
            examples = examples[:limit]
        
        return [
            RLPrompt(
                prompt=ex.prompt,
                id=ex.id,
                metadata={"gold_answer": ex.metadata["gold_answer"]},
            )
            for ex in examples
        ]


class MATHDataset(BaseDataset):
    """
    MATH: Competition mathematics dataset.
    
    12.5K problems from AMC, AIME, etc. with solutions.
    Harder than GSM8K.
    """
    
    def __init__(
        self,
        split: str = "train",
        template: str = "math_cot",
        max_samples: Optional[int] = None,
        levels: Optional[list[int]] = None,  # Filter by difficulty 1-5
        subjects: Optional[list[str]] = None,  # Filter by subject
    ):
        """
        Initialize MATH dataset.
        
        Args:
            split: Dataset split
            template: Prompt template name
            max_samples: Maximum samples
            levels: Filter to specific difficulty levels (1-5)
            subjects: Filter to specific subjects
        """
        from datasets import load_dataset
        
        self.split = split
        self.template = get_template(template)
        self.levels = levels
        self.subjects = subjects
        
        # Load from HuggingFace
        dataset = load_dataset("hendrycks/competition_math", split=split)
        
        # Filter
        self._data = []
        for item in dataset:
            if levels and item["level"] not in [f"Level {l}" for l in levels]:
                continue
            if subjects and item["type"] not in subjects:
                continue
            self._data.append(item)
        
        if max_samples:
            self._data = self._data[:max_samples]
    
    @property
    def name(self) -> str:
        return "math"
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[SFTExample]:
        for i, item in enumerate(self._data):
            # Extract answer from solution
            # MATH uses \boxed{answer} format
            solution = item["solution"]
            answer_match = re.search(r"\\boxed\{([^}]+)\}", solution)
            gold_answer = answer_match.group(1) if answer_match else None
            
            prompt = self.template.format_prompt(item["problem"])
            response = self.template.format_response(solution)
            
            yield SFTExample(
                prompt=prompt,
                response=response,
                id=f"math_{self.split}_{i}",
                metadata={
                    "gold_answer": gold_answer,
                    "level": item["level"],
                    "subject": item["type"],
                }
            )


class MetaMathQADataset(BaseDataset):
    """
    MetaMathQA: Augmented math dataset.
    
    ~395K samples created by augmenting GSM8K and MATH.
    Good for boosting math performance.
    """
    
    def __init__(
        self,
        split: str = "train",
        template: str = "math_cot",
        max_samples: Optional[int] = None,
        source_filter: Optional[str] = None,  # gsm8k, MATH
    ):
        """
        Initialize MetaMathQA dataset.
        
        Args:
            split: Dataset split
            template: Prompt template name
            max_samples: Maximum samples
            source_filter: Filter to specific source dataset
        """
        from datasets import load_dataset
        
        self.split = split
        self.template = get_template(template)
        
        # Load from HuggingFace
        dataset = load_dataset("meta-math/MetaMathQA", split=split)
        
        self._data = []
        for item in dataset:
            if source_filter:
                source = item.get("type", "")
                if source_filter.lower() not in source.lower():
                    continue
            self._data.append(item)
        
        if max_samples:
            self._data = self._data[:max_samples]
    
    @property
    def name(self) -> str:
        return "metamathqa"
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[SFTExample]:
        for i, item in enumerate(self._data):
            question = item["query"]
            response_text = item["response"]
            
            # Try to extract answer
            gold_answer = self._extract_answer(response_text)
            
            prompt = self.template.format_prompt(question)
            response = self.template.format_response(response_text)
            
            yield SFTExample(
                prompt=prompt,
                response=response,
                id=f"metamath_{i}",
                metadata={
                    "gold_answer": gold_answer,
                    "type": item.get("type", "unknown"),
                }
            )
    
    def _extract_answer(self, response: str) -> Optional[Any]:
        """Extract answer from response."""
        # Try boxed format
        match = re.search(r"\\boxed\{([^}]+)\}", response)
        if match:
            return match.group(1)
        
        # Try "The answer is X" format
        match = re.search(r"[Aa]nswer is[:\s]*(-?\d+)", response)
        if match:
            return int(match.group(1))
        
        # Try #### format
        match = re.search(r"####\s*(-?\d+)", response)
        if match:
            return int(match.group(1))
        
        return None


# ============================================================================
# Answer extraction and checking utilities
# ============================================================================

def extract_math_answer(response: str) -> Optional[Any]:
    """
    Extract numerical answer from model response.
    
    Supports multiple formats:
    - "Answer: 42"
    - "#### 42"
    - "\\boxed{42}"
    - "= 42" at end
    """
    patterns = [
        r"[Aa]nswer[:\s]*\$?(-?\d+(?:\.\d+)?)",
        r"####\s*(-?\d+(?:\.\d+)?)",
        r"\\boxed\{(-?\d+(?:\.\d+)?)\}",
        r"=\s*\$?(-?\d+(?:\.\d+)?)\s*$",
        r"(?:is|equals?)\s*\$?(-?\d+(?:\.\d+)?)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.MULTILINE)
        if match:
            try:
                val = match.group(1)
                # Try int first, then float
                if "." in val:
                    return float(val)
                return int(val)
            except ValueError:
                continue
    
    # Last resort: find last number
    numbers = re.findall(r"(-?\d+(?:\.\d+)?)", response)
    if numbers:
        try:
            val = numbers[-1]
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            pass
    
    return None


def check_math_answer(predicted: Any, gold: Any, tolerance: float = 1e-6) -> bool:
    """
    Check if predicted answer matches gold.
    
    Args:
        predicted: Predicted answer
        gold: Gold answer
        tolerance: Tolerance for float comparison
        
    Returns:
        True if answers match
    """
    if predicted is None or gold is None:
        return False
    
    # Convert to comparable types
    try:
        if isinstance(gold, str):
            # Try to parse gold as number
            gold = float(gold) if "." in gold else int(gold)
        if isinstance(predicted, str):
            predicted = float(predicted) if "." in predicted else int(predicted)
    except (ValueError, TypeError):
        # Fall back to string comparison
        return str(predicted).strip() == str(gold).strip()
    
    # Numeric comparison
    if isinstance(gold, float) or isinstance(predicted, float):
        return abs(float(predicted) - float(gold)) < tolerance
    
    return predicted == gold
