"""Training algorithms: SFT, DPO, PPO, GRPO."""

from .base import (
    BaseTrainer,
    TrainerConfig,
    SFTConfig,
    DPOConfig,
    PPOConfig,
    GRPOConfig,
    TrainResult,
    EvalResult,
)
from .sft import SFTTrainer, train_sft
from .dpo import DPOTrainer
from .ppo import PPOTrainer
from .grpo import GRPOTrainer


def load_trainer(
    trainer_type: str,
    model,
    train_dataset,
    config,
    **kwargs,
):
    """
    Load trainer by type.
    
    Args:
        trainer_type: Type of trainer (sft, dpo, ppo, grpo)
        model: Trainable model
        train_dataset: Training dataset
        config: Trainer configuration
        **kwargs: Additional arguments (reward_fn for RL trainers)
        
    Returns:
        Initialized trainer
    """
    if trainer_type == "sft":
        return SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            config=config,
            eval_dataset=kwargs.get("eval_dataset"),
        )
    elif trainer_type == "dpo":
        return DPOTrainer(
            model=model,
            train_dataset=train_dataset,
            config=config,
            eval_dataset=kwargs.get("eval_dataset"),
            reference_model=kwargs.get("reference_model"),
        )
    elif trainer_type == "ppo":
        return PPOTrainer(
            model=model,
            train_dataset=train_dataset,
            reward_fn=kwargs["reward_fn"],
            config=config,
            eval_dataset=kwargs.get("eval_dataset"),
        )
    elif trainer_type == "grpo":
        return GRPOTrainer(
            model=model,
            train_dataset=train_dataset,
            reward_fn=kwargs["reward_fn"],
            config=config,
            eval_dataset=kwargs.get("eval_dataset"),
        )
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")


__all__ = [
    # Base
    "BaseTrainer",
    "TrainerConfig",
    "SFTConfig",
    "DPOConfig",
    "PPOConfig",
    "GRPOConfig",
    "TrainResult",
    "EvalResult",
    # Trainers
    "SFTTrainer",
    "train_sft",
    "DPOTrainer",
    "PPOTrainer",
    "GRPOTrainer",
    # Factory
    "load_trainer",
]
