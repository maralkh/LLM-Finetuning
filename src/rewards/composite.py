"""Composite rewards for combining multiple reward signals."""

from typing import Optional

from .base import BaseReward, RewardOutput


class WeightedReward(BaseReward):
    """
    Combine multiple rewards with weights.
    
    Final reward = sum(weight_i * reward_i)
    """
    
    def __init__(
        self,
        rewards: list[tuple[BaseReward, float]],
        normalize: bool = False,
    ):
        """
        Initialize weighted reward.
        
        Args:
            rewards: List of (reward_function, weight) tuples
            normalize: If True, normalize weights to sum to 1
        """
        self.rewards = rewards
        
        if normalize:
            total_weight = sum(w for _, w in rewards)
            self.rewards = [(r, w / total_weight) for r, w in rewards]
    
    @property
    def name(self) -> str:
        names = [r.name for r, _ in self.rewards]
        return f"weighted({'+'.join(names)})"
    
    def compute(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        """Compute weighted sum of rewards."""
        total_reward = 0.0
        component_rewards = {}
        
        for reward_fn, weight in self.rewards:
            result = reward_fn.compute(prompt, response, metadata)
            total_reward += weight * result.reward
            component_rewards[reward_fn.name] = {
                "reward": result.reward,
                "weight": weight,
                "weighted": weight * result.reward,
            }
        
        return RewardOutput(
            reward=total_reward,
            info={"components": component_rewards},
        )


class OutcomePlusProcessReward(BaseReward):
    """
    Combine outcome reward with process reward.
    
    Useful for GRPO with both final answer correctness
    and step-by-step reasoning quality.
    """
    
    def __init__(
        self,
        outcome_reward: BaseReward,
        process_reward: BaseReward,
        outcome_weight: float = 0.5,
        process_weight: float = 0.5,
    ):
        """
        Initialize combined reward.
        
        Args:
            outcome_reward: Reward for final answer
            process_reward: Reward for reasoning process
            outcome_weight: Weight for outcome reward
            process_weight: Weight for process reward
        """
        self.outcome_reward = outcome_reward
        self.process_reward = process_reward
        self.outcome_weight = outcome_weight
        self.process_weight = process_weight
    
    @property
    def name(self) -> str:
        return f"outcome_process({self.outcome_reward.name}+{self.process_reward.name})"
    
    def compute(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        """Compute combined reward."""
        outcome_result = self.outcome_reward.compute(prompt, response, metadata)
        process_result = self.process_reward.compute(prompt, response, metadata)
        
        total_reward = (
            self.outcome_weight * outcome_result.reward +
            self.process_weight * process_result.reward
        )
        
        return RewardOutput(
            reward=total_reward,
            info={
                "outcome_reward": outcome_result.reward,
                "process_reward": process_result.reward,
                "outcome_info": outcome_result.info,
                "process_info": process_result.info,
            },
        )


class ThresholdReward(BaseReward):
    """
    Apply threshold to another reward.
    
    Useful for converting continuous rewards to discrete.
    """
    
    def __init__(
        self,
        base_reward: BaseReward,
        threshold: float,
        above_value: float = 1.0,
        below_value: float = 0.0,
    ):
        """
        Initialize threshold reward.
        
        Args:
            base_reward: Underlying reward function
            threshold: Threshold value
            above_value: Reward if base >= threshold
            below_value: Reward if base < threshold
        """
        self.base_reward = base_reward
        self.threshold = threshold
        self.above_value = above_value
        self.below_value = below_value
    
    @property
    def name(self) -> str:
        return f"threshold({self.base_reward.name})"
    
    def compute(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        """Compute thresholded reward."""
        result = self.base_reward.compute(prompt, response, metadata)
        
        if result.reward >= self.threshold:
            reward = self.above_value
        else:
            reward = self.below_value
        
        return RewardOutput(
            reward=reward,
            info={
                "base_reward": result.reward,
                "threshold": self.threshold,
                "above_threshold": result.reward >= self.threshold,
            },
        )


class ClippedReward(BaseReward):
    """
    Clip reward to a range.
    
    Useful for preventing extreme reward values.
    """
    
    def __init__(
        self,
        base_reward: BaseReward,
        min_value: float = -1.0,
        max_value: float = 1.0,
    ):
        """
        Initialize clipped reward.
        
        Args:
            base_reward: Underlying reward function
            min_value: Minimum reward value
            max_value: Maximum reward value
        """
        self.base_reward = base_reward
        self.min_value = min_value
        self.max_value = max_value
    
    @property
    def name(self) -> str:
        return f"clipped({self.base_reward.name})"
    
    def compute(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        """Compute clipped reward."""
        result = self.base_reward.compute(prompt, response, metadata)
        
        clipped = max(self.min_value, min(self.max_value, result.reward))
        
        return RewardOutput(
            reward=clipped,
            info={
                "original_reward": result.reward,
                "clipped": result.reward != clipped,
            },
        )
