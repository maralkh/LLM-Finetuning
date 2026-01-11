"""Main training runner - orchestrates model, data, and trainer."""

from typing import Optional
from pathlib import Path

from ..utils import load_yaml
from ..models import load_model_for_training, ModelConfig, LoRAConfig
from ..data import load_dataset
from ..rewards import load_reward
from ..trainers import (
    load_trainer,
    SFTConfig,
    DPOConfig,
    PPOConfig,
    GRPOConfig,
    TrainResult,
)


class TrainingRunner:
    """
    Main runner for training pipelines.
    
    Handles:
    - Loading configuration
    - Initializing components
    - Running training
    - Saving results
    """
    
    def __init__(self, config: dict):
        """
        Initialize runner from config dict.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.model = None
        self.trainer = None
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingRunner":
        """
        Create runner from YAML config file.
        
        Args:
            config_path: Path to YAML config
            
        Returns:
            Initialized TrainingRunner
        """
        config = load_yaml(config_path)
        return cls(config)
    
    def setup(self):
        """Setup all components."""
        print("Setting up training run...")
        
        # Build model config
        model_cfg = self._build_model_config()
        
        # Load model
        print(f"Loading model: {model_cfg.name}")
        self.model = load_model_for_training(model_cfg)
        
        # Load datasets
        print("Loading datasets...")
        train_dataset = load_dataset(self.config["data"])
        
        eval_dataset = None
        if "eval_data" in self.config:
            eval_dataset = load_dataset(self.config["eval_data"])
        
        # Build trainer config
        trainer_type = self.config.get("trainer", {}).get("type", "sft")
        trainer_cfg = self._build_trainer_config(trainer_type)
        
        # Load reward function for RL trainers
        reward_fn = None
        if trainer_type in ["ppo", "grpo"] and "reward" in self.config:
            print("Loading reward function...")
            reward_fn = load_reward(self.config["reward"])
        
        # Create trainer
        print(f"Creating {trainer_type} trainer...")
        trainer_kwargs = {"eval_dataset": eval_dataset}
        
        if reward_fn is not None:
            trainer_kwargs["reward_fn"] = reward_fn
        
        self.trainer = load_trainer(
            trainer_type=trainer_type,
            model=self.model,
            train_dataset=train_dataset,
            config=trainer_cfg,
            **trainer_kwargs,
        )
        
        print("Setup complete!")
    
    def run(self) -> TrainResult:
        """
        Run training.
        
        Returns:
            TrainResult with training summary
        """
        if self.trainer is None:
            self.setup()
        
        return self.trainer.train()
    
    def _build_model_config(self) -> ModelConfig:
        """Build ModelConfig from config dict."""
        model_dict = self.config.get("model", {})
        
        # Build LoRA config if using LoRA
        lora_config = None
        if model_dict.get("use_lora", True):
            lora_dict = model_dict.get("lora", {})
            lora_config = LoRAConfig(
                r=lora_dict.get("r", 16),
                alpha=lora_dict.get("alpha", 32),
                dropout=lora_dict.get("dropout", 0.05),
                target_modules=lora_dict.get("target_modules"),
            )
        
        return ModelConfig(
            name=model_dict["name"],
            dtype=model_dict.get("dtype", "bfloat16"),
            use_lora=model_dict.get("use_lora", True),
            lora=lora_config or LoRAConfig(),
            gradient_checkpointing=model_dict.get("gradient_checkpointing", True),
            resume_from=model_dict.get("resume_from"),
            trust_remote_code=model_dict.get("trust_remote_code", True),
            attn_implementation=model_dict.get("attn_implementation"),
        )
    
    def _build_trainer_config(self, trainer_type: str):
        """Build trainer-specific config."""
        trainer_dict = self.config.get("trainer", {})
        
        # Common config fields
        common = {
            "output_dir": trainer_dict.get("output_dir", "results"),
            "run_name": trainer_dict.get("run_name"),
            "num_epochs": trainer_dict.get("num_epochs", 3),
            "max_steps": trainer_dict.get("max_steps"),
            "batch_size": trainer_dict.get("batch_size", 4),
            "gradient_accumulation_steps": trainer_dict.get("gradient_accumulation_steps", 4),
            "learning_rate": trainer_dict.get("learning_rate", 2e-5),
            "weight_decay": trainer_dict.get("weight_decay", 0.01),
            "max_grad_norm": trainer_dict.get("max_grad_norm", 1.0),
            "warmup_ratio": trainer_dict.get("warmup_ratio", 0.1),
            "warmup_steps": trainer_dict.get("warmup_steps"),
            "lr_scheduler_type": trainer_dict.get("lr_scheduler_type", "cosine"),
            "max_seq_length": trainer_dict.get("max_seq_length", 2048),
            "logging_steps": trainer_dict.get("logging_steps", 10),
            "save_steps": trainer_dict.get("save_steps", 500),
            "eval_steps": trainer_dict.get("eval_steps", 500),
            "seed": trainer_dict.get("seed", 42),
            "resume_from_checkpoint": trainer_dict.get("resume_from_checkpoint"),
        }
        
        if trainer_type == "sft":
            return SFTConfig(
                **common,
                packing=trainer_dict.get("packing", False),
            )
        elif trainer_type == "dpo":
            return DPOConfig(
                **common,
                beta=trainer_dict.get("beta", 0.1),
                label_smoothing=trainer_dict.get("label_smoothing", 0.0),
                loss_type=trainer_dict.get("loss_type", "sigmoid"),
                reference_free=trainer_dict.get("reference_free", False),
            )
        elif trainer_type == "ppo":
            return PPOConfig(
                **common,
                ppo_epochs=trainer_dict.get("ppo_epochs", 4),
                clip_range=trainer_dict.get("clip_range", 0.2),
                kl_coef=trainer_dict.get("kl_coef", 0.05),
                vf_coef=trainer_dict.get("vf_coef", 0.5),
                max_new_tokens=trainer_dict.get("max_new_tokens", 512),
                temperature=trainer_dict.get("temperature", 0.7),
                top_p=trainer_dict.get("top_p", 0.95),
                normalize_rewards=trainer_dict.get("normalize_rewards", True),
                reward_clip=trainer_dict.get("reward_clip"),
            )
        elif trainer_type == "grpo":
            return GRPOConfig(
                **common,
                group_size=trainer_dict.get("group_size", 8),
                kl_coef=trainer_dict.get("kl_coef", 0.05),
                max_new_tokens=trainer_dict.get("max_new_tokens", 512),
                temperature=trainer_dict.get("temperature", 0.8),
                top_p=trainer_dict.get("top_p", 0.95),
                baseline_type=trainer_dict.get("baseline_type", "mean"),
                normalize_rewards=trainer_dict.get("normalize_rewards", True),
            )
        else:
            raise ValueError(f"Unknown trainer type: {trainer_type}")


def run_from_config(config_path: str) -> TrainResult:
    """
    Run training from a YAML config file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        TrainResult
    """
    runner = TrainingRunner.from_yaml(config_path)
    return runner.run()
