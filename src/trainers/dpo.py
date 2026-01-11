"""Direct Preference Optimization (DPO) trainer."""

from typing import Optional
from pathlib import Path
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from tqdm import tqdm

from .base import BaseTrainer, DPOConfig, TrainResult, EvalResult
from ..models import TrainableModel, load_model_for_training, ModelConfig, LoRAConfig
from ..data import BasePreferenceDataset, PreferenceExample, DataCollatorForDPO
from ..utils import LocalLogger, ProgressTracker, save_training_config, save_checkpoint_info


class DPODataset(Dataset):
    """PyTorch dataset wrapper for preference examples."""
    
    def __init__(self, examples: list[PreferenceExample]):
        self.examples = examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "prompt": ex.prompt,
            "chosen": ex.chosen,
            "rejected": ex.rejected,
        }


class DPOTrainer(BaseTrainer):
    """
    Direct Preference Optimization trainer.
    
    DPO directly optimizes the policy to prefer chosen over rejected responses,
    without needing a separate reward model.
    
    Reference: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    https://arxiv.org/abs/2305.18290
    """
    
    def __init__(
        self,
        model: TrainableModel,
        train_dataset: BasePreferenceDataset,
        config: DPOConfig,
        eval_dataset: Optional[BasePreferenceDataset] = None,
        reference_model: Optional[TrainableModel] = None,
    ):
        """
        Initialize DPO trainer.
        
        Args:
            model: Policy model to train
            train_dataset: Preference dataset
            config: DPO configuration
            eval_dataset: Optional eval dataset
            reference_model: Reference model (if None, uses frozen copy of model)
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config
        
        # Setup reference model
        if config.reference_free:
            self.ref_model = None
        elif reference_model is not None:
            self.ref_model = reference_model.model
        else:
            # Create frozen copy
            self.ref_model = self._create_reference_model()
        
        # Setup output directory
        self.output_dir = self._setup_output_dir(config)
        
        # Initialize logger
        self.logger = LocalLogger(
            output_dir=str(self.output_dir),
            log_interval=config.logging_steps,
            verbose=True,
        )
        
        # Data collator
        self.collator = DataCollatorForDPO(
            tokenizer=model.tokenizer,
            max_length=config.max_seq_length,
        )
        
        # Prepare data
        self._prepare_data()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
    
    @property
    def name(self) -> str:
        return "dpo"
    
    def _create_reference_model(self):
        """Create frozen reference model."""
        ref_model = copy.deepcopy(self.model.model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model
    
    def _prepare_data(self):
        """Prepare data loaders."""
        train_examples = self.train_dataset.get_examples()
        self.train_data = DPODataset(train_examples)
        
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=0,
            pin_memory=True,
        )
        
        if self.eval_dataset:
            eval_examples = self.eval_dataset.get_examples()
            self.eval_data = DPODataset(eval_examples)
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
        
        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        if self.config.max_steps:
            self.total_steps = self.config.max_steps
        else:
            self.total_steps = steps_per_epoch * self.config.num_epochs
        
        self.logger.log_message(f"Preference pairs: {len(train_examples)}")
        self.logger.log_message(f"Total steps: {self.total_steps}")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
        
        warmup_steps = self.config.warmup_steps or int(self.total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )
    
    def _compute_log_probs(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for sequences."""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits[:, :-1, :]  # Shift for next token prediction
        labels = labels[:, 1:]  # Shift labels
        
        # Compute per-token log probs
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding
        mask = (labels != -100).float()
        token_log_probs = token_log_probs * mask
        
        # Sum log probs
        sequence_log_probs = token_log_probs.sum(dim=-1)
        
        return sequence_log_probs
    
    def _compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute DPO loss.
        
        Args:
            policy_chosen_logps: Policy log probs for chosen
            policy_rejected_logps: Policy log probs for rejected
            ref_chosen_logps: Reference log probs for chosen
            ref_rejected_logps: Reference log probs for rejected
            
        Returns:
            Tuple of (loss, metrics dict)
        """
        # Compute log ratios
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        # DPO loss: -log sigmoid(beta * (log_ratio_policy - log_ratio_ref))
        logits = self.config.beta * (pi_logratios - ref_logratios)
        
        if self.config.loss_type == "sigmoid":
            loss = -F.logsigmoid(logits).mean()
        elif self.config.loss_type == "hinge":
            loss = torch.relu(1 - logits).mean()
        elif self.config.loss_type == "ipo":
            # IPO loss
            loss = (logits - 1 / (2 * self.config.beta)) ** 2
            loss = loss.mean()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Apply label smoothing
        if self.config.label_smoothing > 0:
            smooth_loss = -F.logsigmoid(-logits).mean()
            loss = (1 - self.config.label_smoothing) * loss + self.config.label_smoothing * smooth_loss
        
        # Metrics
        with torch.no_grad():
            chosen_rewards = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = self.config.beta * (policy_rejected_logps - ref_rejected_logps)
            reward_margin = (chosen_rewards - rejected_rewards).mean()
            accuracy = (logits > 0).float().mean()
        
        metrics = {
            "loss": loss.item(),
            "reward_margin": reward_margin.item(),
            "accuracy": accuracy.item(),
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
        }
        
        return loss, metrics
    
    def train(self) -> TrainResult:
        """Run DPO training."""
        save_training_config(str(self.output_dir), {
            "trainer": "dpo",
            "model": str(self.model.config),
            "training": self.config.__dict__,
        })
        
        self.logger.log_message("Starting DPO training...")
        self.model.model.train()
        
        progress = ProgressTracker(self.total_steps)
        accumulated_metrics = {}
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            self.logger.log_message(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} ---")
            
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")):
                # Move to device
                device = self.model.model.device
                chosen_input_ids = batch["chosen_input_ids"].to(device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(device)
                chosen_labels = batch["chosen_labels"].to(device)
                rejected_input_ids = batch["rejected_input_ids"].to(device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(device)
                rejected_labels = batch["rejected_labels"].to(device)
                
                # Compute policy log probs
                policy_chosen_logps = self._compute_log_probs(
                    self.model.model, chosen_input_ids, chosen_attention_mask, chosen_labels
                )
                policy_rejected_logps = self._compute_log_probs(
                    self.model.model, rejected_input_ids, rejected_attention_mask, rejected_labels
                )
                
                # Compute reference log probs
                if self.ref_model is not None:
                    with torch.no_grad():
                        ref_chosen_logps = self._compute_log_probs(
                            self.ref_model, chosen_input_ids, chosen_attention_mask, chosen_labels
                        )
                        ref_rejected_logps = self._compute_log_probs(
                            self.ref_model, rejected_input_ids, rejected_attention_mask, rejected_labels
                        )
                else:
                    # Reference-free: use zeros
                    ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
                    ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
                
                # Compute loss
                loss, metrics = self._compute_dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                )
                
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Accumulate metrics
                for k, v in metrics.items():
                    if k not in accumulated_metrics:
                        accumulated_metrics[k] = 0.0
                    accumulated_metrics[k] += v
                
                # Update weights
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
                    
                    # Average metrics
                    avg_metrics = {
                        k: v / self.config.gradient_accumulation_steps
                        for k, v in accumulated_metrics.items()
                    }
                    avg_metrics["lr"] = self.scheduler.get_last_lr()[0]
                    
                    self.logger.log(self.global_step, avg_metrics)
                    accumulated_metrics = {}
                    
                    # Save checkpoint
                    if self.config.save_steps and self.global_step % self.config.save_steps == 0:
                        checkpoint_path = self.output_dir / f"checkpoint-{self.global_step}"
                        self.save_checkpoint(str(checkpoint_path), self.global_step)
                    
                    if self.config.max_steps and self.global_step >= self.config.max_steps:
                        break
            
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break
        
        # Final save
        final_path = self.output_dir / "final"
        self.model.save(str(final_path))
        
        self.logger.log_message(f"\nTraining complete! Model saved to {final_path}")
        self.logger.finish()
        
        return TrainResult(
            output_dir=str(self.output_dir),
            final_checkpoint=str(final_path),
            total_steps=self.global_step,
            final_loss=metrics.get("loss", 0.0),
        )
    
    def evaluate(self) -> EvalResult:
        """Evaluate on preference dataset."""
        if self.eval_loader is None:
            return EvalResult(metrics={}, num_examples=0)
        
        self.model.model.eval()
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                device = self.model.model.device
                
                policy_chosen_logps = self._compute_log_probs(
                    self.model.model,
                    batch["chosen_input_ids"].to(device),
                    batch["chosen_attention_mask"].to(device),
                    batch["chosen_labels"].to(device),
                )
                policy_rejected_logps = self._compute_log_probs(
                    self.model.model,
                    batch["rejected_input_ids"].to(device),
                    batch["rejected_attention_mask"].to(device),
                    batch["rejected_labels"].to(device),
                )
                
                if self.ref_model is not None:
                    ref_chosen_logps = self._compute_log_probs(
                        self.ref_model,
                        batch["chosen_input_ids"].to(device),
                        batch["chosen_attention_mask"].to(device),
                        batch["chosen_labels"].to(device),
                    )
                    ref_rejected_logps = self._compute_log_probs(
                        self.ref_model,
                        batch["rejected_input_ids"].to(device),
                        batch["rejected_attention_mask"].to(device),
                        batch["rejected_labels"].to(device),
                    )
                else:
                    ref_chosen_logps = torch.zeros_like(policy_chosen_logps)
                    ref_rejected_logps = torch.zeros_like(policy_rejected_logps)
                
                _, metrics = self._compute_dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                )
                
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v
                num_batches += 1
        
        avg_metrics = {f"eval_{k}": v / num_batches for k, v in total_metrics.items()}
        
        self.model.model.train()
        
        return EvalResult(
            metrics=avg_metrics,
            num_examples=len(self.eval_data) if self.eval_data else 0,
        )
    
    def save_checkpoint(self, path: str, step: int):
        """Save checkpoint."""
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
        """Load checkpoint."""
        checkpoint_path = Path(path)
        state_path = checkpoint_path / "trainer_state.pt"
        
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.global_step = state["global_step"]
            self.current_epoch = state["epoch"]
            self.logger.log_message(f"Resumed from step {self.global_step}")
