"""Supervised Finetuning (SFT) trainer."""

from typing import Optional
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from tqdm import tqdm

from .base import BaseTrainer, SFTConfig, TrainResult, EvalResult
from ..models import TrainableModel, load_model_for_training, ModelConfig, LoRAConfig
from ..data import BaseDataset, DataCollatorForSFT, SFTExample
from ..utils import LocalLogger, ProgressTracker, save_training_config, save_checkpoint_info


class SFTDataset(Dataset):
    """PyTorch dataset wrapper for SFT examples."""
    
    def __init__(self, examples: list[SFTExample]):
        self.examples = examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "prompt": ex.prompt,
            "response": ex.response,
            "text": ex.prompt + ex.response,
        }


class SFTTrainer(BaseTrainer):
    """
    Supervised Finetuning trainer.
    
    Standard causal language model training on prompt-response pairs.
    Supports LoRA and full finetuning.
    """
    
    def __init__(
        self,
        model: TrainableModel,
        train_dataset: BaseDataset,
        config: SFTConfig,
        eval_dataset: Optional[BaseDataset] = None,
    ):
        """
        Initialize SFT trainer.
        
        Args:
            model: Trainable model
            train_dataset: Training dataset
            config: Training configuration
            eval_dataset: Optional evaluation dataset
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        
        # Setup output directory
        self.output_dir = self._setup_output_dir(config)
        
        # Initialize logger
        self.logger = LocalLogger(
            output_dir=str(self.output_dir),
            log_interval=config.logging_steps,
            verbose=True,
        )
        
        # Data collator
        self.collator = DataCollatorForSFT(
            tokenizer=model.tokenizer,
            max_length=config.max_seq_length,
            mask_prompt=True,
        )
        
        # Prepare data
        self._prepare_data()
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
    
    @property
    def name(self) -> str:
        return "sft"
    
    def _prepare_data(self):
        """Prepare data loaders."""
        # Get examples
        train_examples = self.train_dataset.get_sft_examples()
        self.train_data = SFTDataset(train_examples)
        
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=0,
            pin_memory=True,
        )
        
        if self.eval_dataset:
            eval_examples = self.eval_dataset.get_sft_examples()
            self.eval_data = SFTDataset(eval_examples)
            self.eval_loader = DataLoader(
                self.eval_data,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self.collator,
                num_workers=0,
            )
        else:
            self.eval_data = None
            self.eval_loader = None
        
        # Calculate total steps
        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        if self.config.max_steps:
            self.total_steps = self.config.max_steps
        else:
            self.total_steps = steps_per_epoch * self.config.num_epochs
        
        self.logger.log_message(f"Training examples: {len(train_examples)}")
        self.logger.log_message(f"Steps per epoch: {steps_per_epoch}")
        self.logger.log_message(f"Total steps: {self.total_steps}")
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
        
        if self.config.warmup_steps:
            warmup_steps = self.config.warmup_steps
        else:
            warmup_steps = int(self.total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )
        
        self.logger.log_message(f"Warmup steps: {warmup_steps}")
    
    def train(self) -> TrainResult:
        """Run training loop."""
        
        save_training_config(str(self.output_dir), {
            "trainer": "sft",
            "model": str(self.model.config),
            "training": self.config.__dict__,
        })
        
        self.logger.log_message("Starting SFT training...")
        self.model.model.train()
        
        progress = ProgressTracker(self.total_steps)
        accumulated_loss = 0.0
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            self.logger.log_message(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} ---")
            
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")):
                batch = {k: v.to(self.model.model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                
                loss = outputs.loss / self.config.gradient_accumulation_steps
                accumulated_loss += loss.item()
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    progress.update(self.global_step)
                    
                    self.logger.log(self.global_step, {
                        "loss": accumulated_loss,
                        "lr": self.scheduler.get_last_lr()[0],
                    })
                    accumulated_loss = 0.0
                    
                    if self.config.eval_steps and self.global_step % self.config.eval_steps == 0:
                        if self.eval_loader:
                            eval_result = self.evaluate()
                            self.logger.log_eval(self.global_step, eval_result.metrics)
                            self.model.model.train()
                    
                    if self.config.save_steps and self.global_step % self.config.save_steps == 0:
                        checkpoint_path = self.output_dir / f"checkpoint-{self.global_step}"
                        self.save_checkpoint(str(checkpoint_path), self.global_step)
                    
                    if self.config.max_steps and self.global_step >= self.config.max_steps:
                        break
            
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        final_path = self.output_dir / "final"
        self.model.save(str(final_path))
        
        self.logger.log_message(f"\nTraining complete! Model saved to {final_path}")
        self.logger.finish()
        
        return TrainResult(
            output_dir=str(self.output_dir),
            final_checkpoint=str(final_path),
            total_steps=self.global_step,
            final_loss=accumulated_loss,
        )
    
    def evaluate(self) -> EvalResult:
        """Run evaluation."""
        if self.eval_loader is None:
            return EvalResult(metrics={}, num_examples=0)
        
        self.model.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.model.model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return EvalResult(
            metrics={"eval_loss": avg_loss, "perplexity": perplexity},
            num_examples=len(self.eval_data) if self.eval_data else 0,
        )
    
    def save_checkpoint(self, path: str, step: int):
        """Save training checkpoint."""
        checkpoint_path = Path(path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(checkpoint_path))
        
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.current_epoch,
        }, checkpoint_path / "trainer_state.pt")
        
        save_checkpoint_info(str(checkpoint_path), step=step, epoch=self.current_epoch)
        self.logger.log_message(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint_path = Path(path)
        state_path = checkpoint_path / "trainer_state.pt"
        
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.global_step = state["global_step"]
            self.current_epoch = state["epoch"]
            self.logger.log_message(f"Resumed from step {self.global_step}")


def train_sft(
    model_config: dict,
    data_config: dict,
    trainer_config: dict,
    eval_data_config: Optional[dict] = None,
) -> TrainResult:
    """
    Convenience function to run SFT training from configs.
    
    Args:
        model_config: Model configuration dict
        data_config: Training data configuration dict
        trainer_config: Trainer configuration dict
        eval_data_config: Optional eval data configuration dict
        
    Returns:
        TrainResult
    """
    from ..data import load_dataset
    
    lora_config = None
    if model_config.get("use_lora", True):
        lora_dict = model_config.get("lora", {})
        lora_config = LoRAConfig(**lora_dict)
    
    model_cfg = ModelConfig(
        name=model_config["name"],
        dtype=model_config.get("dtype", "bfloat16"),
        use_lora=model_config.get("use_lora", True),
        lora=lora_config or LoRAConfig(),
        gradient_checkpointing=model_config.get("gradient_checkpointing", True),
        resume_from=model_config.get("resume_from"),
    )
    
    sft_cfg = SFTConfig(**trainer_config)
    
    model = load_model_for_training(model_cfg)
    train_dataset = load_dataset(data_config)
    eval_dataset = load_dataset(eval_data_config) if eval_data_config else None
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        config=sft_cfg,
        eval_dataset=eval_dataset,
    )
    
    return trainer.train()
