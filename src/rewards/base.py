"""Base reward interface for RL training."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RewardOutput:
    """Output from reward computation."""
    reward: float  # Main reward value
    info: dict = None  # Additional info (for debugging/logging)
    
    def __post_init__(self):
        if self.info is None:
            self.info = {}


class BaseReward(ABC):
    """
    Abstract base for reward functions.
    
    Rewards are used in RL training (PPO, GRPO) to provide
    feedback on model generations.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Reward function identifier."""
        pass
    
    @abstractmethod
    def compute(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        """
        Compute reward for a single response.
        
        Args:
            prompt: Input prompt
            response: Model's response
            metadata: Additional info (e.g., gold_answer)
            
        Returns:
            RewardOutput with reward value
        """
        pass
    
    def compute_batch(
        self,
        prompts: list[str],
        responses: list[str],
        metadata: Optional[list[dict]] = None,
    ) -> list[RewardOutput]:
        """
        Compute rewards for a batch of responses.
        
        Default implementation calls compute() for each.
        Override for batch-optimized implementations.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            metadata: List of metadata dicts
            
        Returns:
            List of RewardOutput
        """
        if metadata is None:
            metadata = [None] * len(prompts)
        
        return [
            self.compute(p, r, m)
            for p, r, m in zip(prompts, responses, metadata)
        ]
    
    def normalize(
        self,
        rewards: list[float],
        method: str = "zscore",
    ) -> list[float]:
        """
        Normalize rewards.
        
        Args:
            rewards: List of reward values
            method: Normalization method (zscore, minmax, none)
            
        Returns:
            Normalized rewards
        """
        import numpy as np
        
        rewards = np.array(rewards)
        
        if method == "zscore":
            mean = rewards.mean()
            std = rewards.std() + 1e-8
            return ((rewards - mean) / std).tolist()
        
        elif method == "minmax":
            min_r = rewards.min()
            max_r = rewards.max()
            if max_r - min_r < 1e-8:
                return [0.0] * len(rewards)
            return ((rewards - min_r) / (max_r - min_r)).tolist()
        
        else:  # none
            return rewards.tolist()


class BinaryReward(BaseReward):
    """
    Simple binary reward: 1.0 if correct, 0.0 otherwise.
    
    Useful as a base class for outcome-based rewards.
    """
    
    def __init__(self, positive: float = 1.0, negative: float = 0.0):
        """
        Initialize binary reward.
        
        Args:
            positive: Reward for correct answer
            negative: Reward for incorrect answer
        """
        self.positive = positive
        self.negative = negative
    
    @property
    def name(self) -> str:
        return "binary"
    
    @abstractmethod
    def is_correct(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Check if response is correct.
        
        Args:
            prompt: Input prompt
            response: Model's response
            metadata: Additional info (e.g., gold_answer)
            
        Returns:
            True if correct
        """
        pass
    
    def compute(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        """Compute binary reward."""
        correct = self.is_correct(prompt, response, metadata)
        reward = self.positive if correct else self.negative
        
        return RewardOutput(
            reward=reward,
            info={"correct": correct},
        )


class ProcessRewardModel(BaseReward):
    """
    Process Reward Model (PRM) base class.
    
    PRMs provide step-by-step rewards for reasoning chains.
    This is a placeholder for integration with trained PRMs.
    """
    
    @property
    def name(self) -> str:
        return "prm"
    
    @abstractmethod
    def score_steps(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> list[float]:
        """
        Score each reasoning step.
        
        Args:
            prompt: Input prompt
            response: Model's response (with reasoning steps)
            metadata: Additional info
            
        Returns:
            List of scores for each step
        """
        pass
    
    def compute(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        """Compute aggregate reward from step scores."""
        step_scores = self.score_steps(prompt, response, metadata)
        
        if not step_scores:
            return RewardOutput(reward=0.0, info={"step_scores": []})
        
        # Aggregate (can be overridden)
        reward = sum(step_scores) / len(step_scores)
        
        return RewardOutput(
            reward=reward,
            info={"step_scores": step_scores, "num_steps": len(step_scores)},
        )
