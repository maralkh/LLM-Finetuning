"""Group Relative Policy Optimization (GRPO) trainer."""

from typing import Optional
from pathlib import Path
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from tqdm import tqdm

from .base import BaseTrainer, GRPOConfig, TrainResult, EvalResult
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


class GRPOTrainer(BaseTrainer):
    """
    Group Relative Policy Optimization trainer.
    
    GRPO is simpler than PPO:
    - No value model needed
    - Samples multiple responses per prompt
    - Uses relative rewards within group for advantage estimation
    - More sample-efficient for verifiable tasks (math, code)
    
    Key insight: For binary rewards (correct/incorrect), the advantage
    of a correct response is naturally higher than incorrect ones
    within the same group.
    
    Reference: DeepSeek-R1 paper (2024)
    """
    
    def __init__(
        self,
        model: TrainableModel,
        train_dataset: BaseDataset,
        reward_fn: BaseReward,
        config: GRPOConfig,
        eval_dataset: Optional[BaseDataset] = None,
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            model: Policy model
            train_dataset: Dataset with prompts
            reward_fn: Reward function
            config: GRPO configuration
            eval_dataset: Optional eval dataset
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.reward_fn = reward_fn
        self.config = config
        
        # Create reference model (frozen copy for KL penalty)
        self.ref_model = self._create_reference_model()
        
        # Setup output
        self.output_dir = self._setup_output_dir(config)
        
        # Logger
        self.logger = LocalLogger(
            output_dir=str(self.output_dir),
            log_interval=config.logging_steps,
            verbose=True,
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
        return "grpo"
    
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
        
        # Batch size is 1 prompt, but we generate group_size responses
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        steps_per_epoch = len(self.train_loader) // self.config.gradient_accumulation_steps
        if self.config.max_steps:
            self.total_steps = self.config.max_steps
        else:
            self.total_steps = steps_per_epoch * self.config.num_epochs
        
        self.logger.log_message(f"RL prompts: {len(rl_prompts)}")
        self.logger.log_message(f"Group size: {self.config.group_size}")
        self.logger.log_message(f"Total steps: {self.total_steps}")
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
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
    
    def _generate_group_responses(
        self,
        prompt: str,
        group_size: int,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """
        Generate a group of responses for one prompt.
        
        Returns:
            Tuple of (responses, log_probs, response_ids)
        """
        tokenizer = self.model.tokenizer
        device = self.model.model.device
        
        # Tokenize prompt
        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        
        prompt_len = inputs["input_ids"].shape[1]
        
        # Generate group_size responses
        self.model.model.eval()
        with torch.no_grad():
            outputs = self.model.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                num_return_sequences=group_size,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode responses
        responses = []
        for seq in outputs:
            response = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            responses.append(response)
        
        # Compute log probs for each response
        self.model.model.train()
        
        # Pad all sequences to same length
        max_len = max(len(seq) for seq in outputs)
        padded_outputs = torch.full(
            (group_size, max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        for i, seq in enumerate(outputs):
            padded_outputs[i, :len(seq)] = seq
        
        attention_mask = (padded_outputs != tokenizer.pad_token_id).long()
        
        # Forward pass
        with torch.no_grad():
            model_outputs = self.model.model(
                input_ids=padded_outputs,
                attention_mask=attention_mask,
            )
        
        logits = model_outputs.logits[:, prompt_len-1:-1, :]
        response_ids = padded_outputs[:, prompt_len:]
        
        # Compute log probs
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=response_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding
        response_mask = (response_ids != tokenizer.pad_token_id).float()
        token_log_probs = token_log_probs * response_mask
        
        # Per-token average log prob (length normalized)
        lengths = response_mask.sum(dim=-1).clamp(min=1)
        sequence_log_probs = token_log_probs.sum(dim=-1) / lengths
        
        return responses, sequence_log_probs, padded_outputs
    
    def _compute_group_rewards(
        self,
        prompt: str,
        responses: list[str],
        metadata: dict,
    ) -> torch.Tensor:
        """Compute rewards for a group of responses."""
        rewards = []
        
        for response in responses:
            result = self.reward_fn.compute(prompt, response, metadata)
            rewards.append(result.reward)
        
        rewards = torch.tensor(rewards, device=self.model.model.device)
        return rewards
    
    def _compute_group_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute advantages using group-relative baseline.
        
        For GRPO, we use the group mean as baseline:
        A_i = r_i - mean(r_group)
        
        This naturally gives positive advantages to above-average
        responses and negative to below-average.
        """
        if self.config.baseline_type == "mean":
            baseline = rewards.mean()
        elif self.config.baseline_type == "min":
            baseline = rewards.min()
        elif self.config.baseline_type == "none":
            baseline = 0.0
        else:
            baseline = rewards.mean()
        
        advantages = rewards - baseline
        
        # Optionally normalize advantages
        if self.config.normalize_rewards and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def _compute_grpo_loss(
        self,
        prompt: str,
        responses: list[str],
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute GRPO loss.
        
        Loss = -E[advantage * log_prob] + kl_coef * KL(policy || ref)
        
        Args:
            prompt: Input prompt
            responses: Generated responses
            old_log_probs: Log probs from generation
            advantages: Computed advantages
            response_ids: Response token IDs
            
        Returns:
            Tuple of (loss, metrics dict)
        """
        tokenizer = self.model.tokenizer
        device = self.model.model.device
        group_size = len(responses)
        
        # Tokenize full sequences
        full_texts = [prompt + r for r in responses]
        inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(device)
        
        prompt_inputs = tokenizer([prompt], return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]
        
        # Forward pass - policy
        outputs = self.model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        
        # Compute new log probs
        logits = outputs.logits[:, prompt_len-1:-1, :]
        response_tokens = inputs["input_ids"][:, prompt_len:]
        
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=response_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        response_mask = (response_tokens != tokenizer.pad_token_id).float()
        token_log_probs = token_log_probs * response_mask
        
        lengths = response_mask.sum(dim=-1).clamp(min=1)
        new_log_probs = token_log_probs.sum(dim=-1) / lengths
        
        # Compute reference log probs
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_token_log_probs = torch.gather(
                ref_log_probs, dim=-1, index=response_tokens.unsqueeze(-1)
            ).squeeze(-1)
            ref_token_log_probs = ref_token_log_probs * response_mask
            ref_sequence_log_probs = ref_token_log_probs.sum(dim=-1) / lengths
        
        # Policy gradient loss
        # Weight log probs by advantages
        pg_loss = -(advantages * new_log_probs).mean()
        
        # KL penalty
        kl = new_log_probs - ref_sequence_log_probs
        kl_loss = self.config.kl_coef * kl.mean()
        
        # Total loss
        loss = pg_loss + kl_loss
        
        # Metrics
        with torch.no_grad():
            reward_mean = advantages.mean() + (advantages - advantages).mean()  # Use original rewards
            accuracy = (advantages > 0).float().mean()
        
        metrics = {
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl_loss": kl_loss.item(),
            "kl": kl.mean().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "log_prob_mean": new_log_probs.mean().item(),
        }
        
        return loss, metrics
    
    def train(self) -> TrainResult:
        """Run GRPO training."""
        save_training_config(str(self.output_dir), {
            "trainer": "grpo",
            "model": str(self.model.config),
            "training": self.config.__dict__,
            "reward": self.reward_fn.name,
        })
        
        self.logger.log_message("Starting GRPO training...")
        self.logger.log_message(f"Reward function: {self.reward_fn.name}")
        
        accumulated_loss = 0.0
        accumulated_metrics = {}
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            self.logger.log_message(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} ---")
            
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")):
                # Process each prompt in batch
                batch_losses = []
                batch_metrics = {}
                
                prompts = batch["prompt"]
                metadata_list = batch["metadata"]
                
                for prompt, metadata in zip(prompts, metadata_list):
                    # Generate group of responses
                    responses, old_log_probs, response_ids = self._generate_group_responses(
                        prompt, self.config.group_size
                    )
                    
                    # Compute rewards
                    rewards = self._compute_group_rewards(prompt, responses, metadata)
                    
                    # Compute group-relative advantages
                    advantages = self._compute_group_advantages(rewards)
                    
                    # Compute GRPO loss
                    loss, metrics = self._compute_grpo_loss(
                        prompt, responses, old_log_probs, advantages, response_ids
                    )
                    
                    batch_losses.append(loss)
                    
                    # Track reward stats
                    metrics["reward_mean"] = rewards.mean().item()
                    metrics["reward_max"] = rewards.max().item()
                    metrics["reward_min"] = rewards.min().item()
                    metrics["num_correct"] = (rewards > 0.5).sum().item()
                    
                    for k, v in metrics.items():
                        if k not in batch_metrics:
                            batch_metrics[k] = []
                        batch_metrics[k].append(v)
                
                # Average loss across batch
                total_loss = sum(batch_losses) / len(batch_losses)
                total_loss = total_loss / self.config.gradient_accumulation_steps
                
                accumulated_loss += total_loss.item()
                
                # Backward
                total_loss.backward()
                
                # Accumulate metrics
                for k, v in batch_metrics.items():
                    if k not in accumulated_metrics:
                        accumulated_metrics[k] = []
                    accumulated_metrics[k].extend(v)
                
                # Update
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Average metrics
                    avg_metrics = {
                        k: sum(v) / len(v) for k, v in accumulated_metrics.items()
                    }
                    avg_metrics["lr"] = self.scheduler.get_last_lr()[0]
                    
                    self.logger.log(self.global_step, avg_metrics)
                    
                    accumulated_loss = 0.0
                    accumulated_metrics = {}
                    
                    # Evaluation
                    if self.config.eval_steps and self.global_step % self.config.eval_steps == 0:
                        if self.eval_dataset:
                            eval_result = self.evaluate()
                            self.logger.log_eval(self.global_step, eval_result.metrics)
                    
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
            final_loss=accumulated_loss,
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
            temperature=0.0,  # Greedy for evaluation
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
