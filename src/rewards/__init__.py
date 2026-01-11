"""Reward functions for RL training."""

from .base import (
    BaseReward,
    BinaryReward,
    ProcessRewardModel,
    RewardOutput,
)
from .outcome import (
    MathCorrectnessReward,
    CodeExecutionReward,
    CustomReward,
)
from .composite import (
    WeightedReward,
    OutcomePlusProcessReward,
    ThresholdReward,
    ClippedReward,
)


def load_reward(config: dict) -> BaseReward:
    """
    Load reward function from configuration.
    
    Args:
        config: Dict with 'type' and reward-specific params
        
    Returns:
        Initialized reward function
    """
    reward_type = config.get("type", "math_correctness")
    
    if reward_type == "math_correctness":
        return MathCorrectnessReward(
            positive=config.get("positive", 1.0),
            negative=config.get("negative", 0.0),
            partial_credit=config.get("partial_credit", False),
        )
    elif reward_type == "code_execution":
        return CodeExecutionReward(
            positive=config.get("positive", 1.0),
            negative=config.get("negative", 0.0),
            timeout=config.get("timeout", 5.0),
            partial_credit=config.get("partial_credit", True),
        )
    elif reward_type == "weighted":
        # Expects 'rewards' list with [{type: ..., weight: ...}, ...]
        reward_configs = config.get("rewards", [])
        rewards = [
            (load_reward(rc), rc.get("weight", 1.0))
            for rc in reward_configs
        ]
        return WeightedReward(
            rewards=rewards,
            normalize=config.get("normalize", False),
        )
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


__all__ = [
    # Base
    "BaseReward",
    "BinaryReward",
    "ProcessRewardModel",
    "RewardOutput",
    # Outcome
    "MathCorrectnessReward",
    "CodeExecutionReward",
    "CustomReward",
    # Composite
    "WeightedReward",
    "OutcomePlusProcessReward",
    "ThresholdReward",
    "ClippedReward",
    # Factory
    "load_reward",
]
