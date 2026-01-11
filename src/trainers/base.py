"""Base trainer interface and configuration."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
from pathlib import Path


@dataclass
class TrainerConfig:
    """Base configuration for all trainers."""
    
    # Output
    output_dir: str
    run_name: Optional[str] = None
    
    # Training duration
    num_epochs: int = 3
    max_steps: Optional[int] = None  # Overrides num_epochs if set
    
    # Batch size
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # Optimizer
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Scheduler
    warmup_ratio: float = 0.1
    warmup_steps: Optional[int] = None  # Overrides warmup_ratio if set
    lr_scheduler_type: str = "cosine"  # linear, cosine, constant
    
    # Sequence length
    max_seq_length: int = 2048
    
    # Logging & saving
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: Optional[int] = 3  # Keep only N latest checkpoints
    
    # Evaluation
    eval_steps: Optional[int] = 500
    eval_on_start: bool = False
    
    # Reproducibility
    seed: int = 42
    
    # Resume
    resume_from_checkpoint: Optional[str] = None  # Path or "latest"
    
    @property
    def effective_batch_size(self) -> int:
        """Total batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class SFTConfig(TrainerConfig):
    """Configuration for supervised finetuning."""
    
    # Packing (combine multiple examples into one sequence)
    packing: bool = False
    
    # Dataset
    dataset_text_field: str = "text"  # Field name for formatted text


@dataclass
class DPOConfig(TrainerConfig):
    """Configuration for Direct Preference Optimization."""
    
    # DPO-specific
    beta: float = 0.1  # KL penalty coefficient
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo
    
    # Reference model
    reference_free: bool = False  # If True, don't use reference model
    
    # Preference data generation (if generating on-the-fly)
    generate_preferences: bool = False
    num_generations: int = 4  # Samples per prompt for rejection sampling


@dataclass
class PPOConfig(TrainerConfig):
    """Configuration for Proximal Policy Optimization."""
    
    # PPO-specific
    ppo_epochs: int = 4  # Updates per batch
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    vf_coef: float = 0.5  # Value function coefficient
    
    # KL penalty
    kl_coef: float = 0.05
    target_kl: Optional[float] = None  # Early stop if KL exceeds
    
    # Generation
    num_generations: int = 4  # Responses per prompt
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 512
    
    # Reward normalization
    normalize_rewards: bool = True
    reward_clip: Optional[float] = 10.0


@dataclass
class GRPOConfig(TrainerConfig):
    """
    Configuration for Group Relative Policy Optimization.
    
    GRPO key differences from PPO:
    - No value model (uses group-based advantage)
    - Samples multiple responses per prompt, uses relative rewards
    - More sample-efficient for verifiable tasks
    """
    
    # GRPO-specific
    group_size: int = 8  # Responses per prompt for advantage estimation
    
    # KL penalty
    kl_coef: float = 0.05
    
    # Generation
    temperature: float = 0.8
    top_p: float = 0.95
    max_new_tokens: int = 512
    
    # Reward
    reward_type: str = "outcome"  # outcome, process, composite
    normalize_rewards: bool = True
    
    # Baseline
    baseline_type: str = "mean"  # mean, min, none


@dataclass
class TrainResult:
    """Result of training."""
    output_dir: str
    final_checkpoint: str
    total_steps: int
    final_loss: float
    best_metric: Optional[float] = None
    metrics_history: list = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of evaluation."""
    metrics: dict
    num_examples: int
    predictions: Optional[list] = None


class BaseTrainer(ABC):
    """Abstract base for all trainers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Trainer identifier."""
        pass
    
    @abstractmethod
    def train(self) -> TrainResult:
        """
        Run training.
        
        Returns:
            TrainResult with training summary
        """
        pass
    
    @abstractmethod
    def evaluate(self) -> EvalResult:
        """
        Run evaluation.
        
        Returns:
            EvalResult with metrics
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str, step: int):
        """
        Save training checkpoint.
        
        Args:
            path: Directory to save to
            step: Current training step
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.
        
        Args:
            path: Directory to load from
        """
        pass
    
    def _setup_output_dir(self, config: TrainerConfig) -> Path:
        """Create output directory for training."""
        from ..utils import create_run_dir
        
        if config.run_name:
            run_dir = create_run_dir(config.output_dir, config.run_name)
        else:
            run_dir = Path(config.output_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        
        return run_dir
