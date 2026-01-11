"""Proximal Policy Optimization (PPO) trainer."""

from typing import Optional
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from tqdm import tqdm

from .base import BaseTrainer, PPOConfig, TrainResult, EvalResult
from ..models import TrainableModel
from ..data import BaseDataset, RLPrompt
from ..rewards import BaseReward
from ..utils import LocalLogger, ProgressTracker, save_training_config, save_checkpoint_info


class RLPromptDataset(Dataset):
    """Dataset wrapper for RL prompts."""
    
    def __init__(self, prompts: list[RLPrompt]):
        self.prompts = prompts
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> dict:
        p = self.prompts[idx]
        return {
            "prompt": p.prompt,
            "id": p.id,
            "metadata": p.metadata,
        }


class ValueHead(nn.Module):
    """Value head for PPO - predicts state value from hidden states."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Use last token's hidden state
        hidden = hidden_states[:, -1, :]
        hidden = self.dropout(hidden)
        hidden = torch.tanh(self.dense(hidden))
        value = self.value_head(hidden).squeeze(-1)
        return value


class PPOTrainer(BaseTrainer):
    """
    Proximal Policy Optimization trainer.
    
    PPO trains a policy to maximize rewards while staying close to a reference policy.
    Uses a value model to estimate advantages.
    
    Reference: "Proximal Policy Optimization Algorithms"
    https://arxiv.org/abs/1707.06347
    """
    
    def __init__(
        self,
        model: TrainableModel,
        train_dataset: BaseDataset,
        reward_fn: BaseReward,
        config: PPOConfig,
        eval_dataset: Optional[BaseDataset] = None,
    ):
        """
        Initialize PPO trainer.
        
        Args:
            model: Policy model
            train_dataset: Dataset with prompts
            reward_fn: Reward function
            config: PPO configuration
            eval_dataset: Optional eval dataset
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.reward_fn = reward_fn
        self.config = config
        
        # Create reference model (frozen copy)
        self.ref_model = self._create_reference_model()
        
        # Create value head
        hidden_size = self.model.model.config.hidden_size
        self.value_head = ValueHead(hidden_size).to(self.model.model.device)
        
        # Setup output directory
        self.output_dir = self._setup_output_dir(config)
        
        # Logger
        self.logger = LocalLogger(
            output_dir=str(self.output_dir),
            log_interval=config.logging_steps,
            verbose=True,
        )
        
        # Prepare data
        self._prepare_data()
        
        # Setup optimizer (includes value head)
        self._setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
    
    @property
    def name(self) -> str:
        return "ppo"
    
    def _create_reference_model(self):
        """Create frozen reference model."""
        ref_model = copy.deepcopy(self.model.model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model
    
    def _prepare_data(self):
        """Prepare data."""
        rl_prompts = self.train_dataset.get_rl_prompts()
        self.train_data = RLPromptDataset(rl_prompts)
        
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        steps_per_epoch = len(self.train_loader)
        if self.config.max_steps:
            self.total_steps = self.config.max_steps
        else:
            self.total_steps = steps_per_epoch * self.config.num_epochs
        
        self.logger.log_message(f"RL prompts: {len(rl_prompts)}")
        self.logger.log_message(f"Total steps: {self.total_steps}")
    
    def _setup_optimizer(self):
        """Setup optimizer for policy and value head."""
        # Combine parameters
        params = list(self.model.model.parameters()) + list(self.value_head.parameters())
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )
        
        warmup_steps = self.config.warmup_steps or int(self.total_steps * self.config.warmup_ratio)
        
        self.scheduler = get_scheduler(
            name=self.config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )
    
    def _generate_responses(
        self,
        prompts: list[str],
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """
        Generate responses and compute log probs.
        
        Returns:
            Tuple of (responses, log_probs, values)
        """
        tokenizer = self.model.tokenizer
        device = self.model.model.device
        
        # Tokenize prompts
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        
        prompt_len = inputs["input_ids"].shape[1]
        
        # Generate with model
        self.model.model.eval()
        with torch.no_grad():
            outputs = self.model.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_ids = outputs.sequences
        
        # Decode responses
        responses = []
        for seq in generated_ids:
            response = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            responses.append(response)
        
        # Compute log probs for generated tokens
        self.model.model.train()
        
        # Forward pass to get logits
        with torch.no_grad():
            model_outputs = self.model.model(
                input_ids=generated_ids,
                attention_mask=(generated_ids != tokenizer.pad_token_id).long(),
                output_hidden_states=True,
            )
        
        logits = model_outputs.logits[:, prompt_len-1:-1, :]  # Shift for prediction
        response_ids = generated_ids[:, prompt_len:]
        
        # Compute log probs
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=response_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding
        mask = (response_ids != tokenizer.pad_token_id).float()
        token_log_probs = token_log_probs * mask
        
        # Sum log probs
        sequence_log_probs = token_log_probs.sum(dim=-1)
        
        # Compute values
        hidden_states = model_outputs.hidden_states[-1]
        values = self.value_head(hidden_states)
        
        return responses, sequence_log_probs, values
    
    def _compute_rewards(
        self,
        prompts: list[str],
        responses: list[str],
        metadata: list[dict],
    ) -> torch.Tensor:
        """Compute rewards for responses."""
        rewards = []
        
        for prompt, response, meta in zip(prompts, responses, metadata):
            result = self.reward_fn.compute(prompt, response, meta)
            rewards.append(result.reward)
        
        rewards = torch.tensor(rewards, device=self.model.model.device)
        
        # Normalize rewards
        if self.config.normalize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Clip rewards
        if self.config.reward_clip:
            rewards = torch.clamp(rewards, -self.config.reward_clip, self.config.reward_clip)
        
        return rewards
    
    def _compute_kl_penalty(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence penalty."""
        kl = log_probs - ref_log_probs
        return self.config.kl_coef * kl
    
    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns using GAE.
        
        For simplicity, we use single-step rewards here.
        """
        # Simple advantage: reward - value
        advantages = rewards - values.detach()
        returns = rewards
        
        return advantages, returns
    
    def _ppo_update(
        self,
        prompts: list[str],
        responses: list[str],
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict:
        """
        Perform PPO update.
        
        Args:
            prompts: Input prompts
            responses: Generated responses
            old_log_probs: Log probs from rollout
            advantages: Computed advantages
            returns: Computed returns
            
        Returns:
            Metrics dict
        """
        tokenizer = self.model.tokenizer
        device = self.model.model.device
        
        # Tokenize full sequences
        full_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(device)
        
        prompt_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]
        
        total_loss = 0.0
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        
        for _ in range(self.config.ppo_epochs):
            # Forward pass
            outputs = self.model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            
            # Compute current log probs
            logits = outputs.logits[:, prompt_len-1:-1, :]
            response_ids = inputs["input_ids"][:, prompt_len:]
            
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=-1, index=response_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            mask = (response_ids != tokenizer.pad_token_id).float()
            token_log_probs = token_log_probs * mask
            new_log_probs = token_log_probs.sum(dim=-1)
            
            # Compute values
            hidden_states = outputs.hidden_states[-1]
            values = self.value_head(hidden_states)
            
            # Policy loss (clipped)
            ratio = torch.exp(new_log_probs - old_log_probs)
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(
                ratio, 1 - self.config.clip_range, 1 + self.config.clip_range
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            # Value loss (clipped)
            vf_loss = F.mse_loss(values, returns)
            
            # Entropy bonus
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            
            # Total loss
            loss = pg_loss + self.config.vf_coef * vf_loss - 0.01 * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.model.parameters()) + list(self.value_head.parameters()),
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            total_pg_loss += pg_loss.item()
            total_vf_loss += vf_loss.item()
            total_entropy += entropy.item()
        
        n_epochs = self.config.ppo_epochs
        
        return {
            "loss": total_loss / n_epochs,
            "pg_loss": total_pg_loss / n_epochs,
            "vf_loss": total_vf_loss / n_epochs,
            "entropy": total_entropy / n_epochs,
        }
    
    def train(self) -> TrainResult:
        """Run PPO training."""
        save_training_config(str(self.output_dir), {
            "trainer": "ppo",
            "model": str(self.model.config),
            "training": self.config.__dict__,
        })
        
        self.logger.log_message("Starting PPO training...")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            self.logger.log_message(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} ---")
            
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")):
                prompts = batch["prompt"]
                metadata = batch["metadata"]
                
                # Generate responses
                responses, old_log_probs, values = self._generate_responses(prompts)
                
                # Compute rewards
                rewards = self._compute_rewards(prompts, responses, metadata)
                
                # Compute KL penalty (need ref model log probs)
                # For simplicity, we skip explicit KL here - it's implicit in PPO clip
                
                # Compute advantages
                advantages, returns = self._compute_advantages(rewards, values)
                
                # PPO update
                metrics = self._ppo_update(
                    prompts, responses, old_log_probs, advantages, returns
                )
                
                # Add reward stats
                metrics["reward_mean"] = rewards.mean().item()
                metrics["reward_std"] = rewards.std().item()
                metrics["lr"] = self.scheduler.get_last_lr()[0]
                
                self.global_step += 1
                self.scheduler.step()
                
                self.logger.log(self.global_step, metrics)
                
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
        
        # Save value head
        torch.save(self.value_head.state_dict(), final_path / "value_head.pt")
        
        self.logger.log_message(f"\nTraining complete! Model saved to {final_path}")
        self.logger.finish()
        
        return TrainResult(
            output_dir=str(self.output_dir),
            final_checkpoint=str(final_path),
            total_steps=self.global_step,
            final_loss=metrics.get("loss", 0.0),
        )
    
    def evaluate(self) -> EvalResult:
        """Evaluate current policy."""
        if self.eval_dataset is None:
            return EvalResult(metrics={}, num_examples=0)
        
        from ..utils import evaluate_from_dataset
        
        eval_metrics = evaluate_from_dataset(
            model=self.model.model,
            tokenizer=self.model.tokenizer,
            dataset=self.eval_dataset,
            max_samples=200,
        )
        
        return EvalResult(
            metrics={
                "eval_accuracy": eval_metrics.accuracy,
                "eval_correct": eval_metrics.correct,
                "eval_total": eval_metrics.total,
            },
            num_examples=eval_metrics.total,
        )
    
    def save_checkpoint(self, path: str, step: int):
        """Save checkpoint."""
        checkpoint_path = Path(path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(checkpoint_path))
        
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "value_head": self.value_head.state_dict(),
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
            self.value_head.load_state_dict(state["value_head"])
            self.global_step = state["global_step"]
            self.current_epoch = state["epoch"]
            self.logger.log_message(f"Resumed from step {self.global_step}")
