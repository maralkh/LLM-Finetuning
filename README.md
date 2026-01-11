# LLM Finetuning Framework

A simple, flexible framework for finetuning 1-4B parameter LLMs with SFT and RL methods.

## Features

- **SFT**: Standard supervised finetuning on prompt-response pairs
- **DPO**: Direct Preference Optimization for preference learning
- **PPO**: Proximal Policy Optimization with value model
- **GRPO**: Group Relative Policy Optimization (value-free, efficient for verifiable tasks)
- **LoRA**: Parameter-efficient finetuning by default
- **Math & Code**: Built-in support for math and code datasets with reward functions

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### SFT Training

```bash
# Train on GSM8K
python scripts/train.py --config config/sft/gsm8k.yaml

# Train on code
python scripts/train.py --config config/sft/code.yaml

# With overrides
python scripts/train.py --config config/sft/gsm8k.yaml \
    --override trainer.num_epochs=1 \
    --override trainer.batch_size=2
```

### RL Training (after SFT)

```bash
# GRPO with math correctness reward
python scripts/train.py --config config/grpo/math_outcome.yaml

# GRPO with code execution reward
python scripts/train.py --config config/grpo/code_execution.yaml

# PPO
python scripts/train.py --config config/ppo/math.yaml

# DPO (requires preference data)
python scripts/train.py --config config/dpo/gsm8k.yaml
```

## Training Pipeline

The recommended training pipeline:

```
1. SFT → 2. RL (GRPO/PPO) or DPO
```

**For math:**
```bash
# Step 1: SFT on GSM8K
python scripts/train.py --config config/sft/gsm8k.yaml

# Step 2: GRPO with math correctness
python scripts/train.py --config config/grpo/math_outcome.yaml
```

**For code:**
```bash
# Step 1: SFT on CodeAlpaca
python scripts/train.py --config config/sft/code.yaml

# Step 2: GRPO with code execution
python scripts/train.py --config config/grpo/code_execution.yaml
```

## Configuration

All training is driven by YAML config files. See `config/` for examples.

### Model Configuration

```yaml
model:
  name: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  dtype: bfloat16
  use_lora: true
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
  gradient_checkpointing: true
  resume_from: null  # Path to continue from checkpoint
```

### Data Configuration

```yaml
data:
  name: gsm8k  # gsm8k, math, metamathqa, codealpaca, magicoder, humaneval, mbpp
  split: train
  template: math_cot  # Prompt template
  max_samples: null
```

### Trainer Configuration

```yaml
trainer:
  type: sft  # sft, dpo, ppo, grpo
  output_dir: results
  run_name: my_experiment
  
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-5
  
  logging_steps: 10
  save_steps: 500
  eval_steps: 500
```

### Reward Configuration (for RL)

```yaml
reward:
  type: math_correctness  # or code_execution
  positive: 1.0
  negative: 0.0
  partial_credit: false  # for code execution
```

## Project Structure

```
llm-finetuning/
├── config/
│   ├── sft/           # SFT configurations
│   │   ├── gsm8k.yaml
│   │   └── code.yaml
│   ├── dpo/           # DPO configurations
│   │   └── gsm8k.yaml
│   ├── ppo/           # PPO configurations
│   │   └── math.yaml
│   └── grpo/          # GRPO configurations
│       ├── math_outcome.yaml
│       └── code_execution.yaml
│
├── src/
│   ├── models/        # Model loading & LoRA
│   ├── data/          # Datasets & templates
│   ├── trainers/      # Training algorithms
│   ├── rewards/       # Reward functions for RL
│   ├── utils/         # Logging, checkpointing
│   └── runners/       # Training orchestration
│
├── scripts/
│   └── train.py       # Main entry point
│
├── results/           # Training outputs
└── requirements.txt
```

## Supported Components

### Datasets

| Name | Type | Description |
|------|------|-------------|
| `gsm8k` | Math | Grade school math (7.5K train) |
| `math` | Math | Competition math (12.5K train) |
| `metamathqa` | Math | Augmented math (395K) |
| `codealpaca` | Code | Code instructions (20K) |
| `magicoder` | Code | High-quality code (75K) |
| `humaneval` | Code | Code eval benchmark (164) |
| `mbpp` | Code | Basic Python (1K) |

### Trainers

| Type | Description | When to Use |
|------|-------------|-------------|
| `sft` | Supervised finetuning | First step, foundation |
| `dpo` | Direct Preference Optimization | When you have preference data |
| `ppo` | Proximal Policy Optimization | Complex reward shaping |
| `grpo` | Group Relative Policy Optimization | Verifiable tasks (math, code) |

### Reward Functions

| Type | Description | Use With |
|------|-------------|----------|
| `math_correctness` | Final answer correctness | Math tasks |
| `code_execution` | Test case execution | Code tasks |
| `weighted` | Combine multiple rewards | Complex setups |

## Training Tips

1. **Start with SFT**: Always do SFT before RL for better starting point
2. **LoRA**: Use LoRA for memory efficiency (enabled by default)
3. **Batch size**: Use gradient accumulation for effective larger batches
4. **Learning rate**: 
   - SFT: ~2e-5
   - DPO: ~5e-7
   - PPO/GRPO: ~1e-6
5. **GRPO group_size**: 8 is a good default, increase for harder tasks
6. **Temperature**: 0.7-0.8 for generation during RL

## Checkpoints

Checkpoints are saved to `{output_dir}/{run_name}_{timestamp}/`:

```
checkpoint-500/
├── adapter_model.safetensors  # LoRA weights
├── adapter_config.json
├── tokenizer.json
├── trainer_state.pt           # Optimizer, scheduler
└── checkpoint_info.json       # Step, metrics
```

### Resuming Training

```yaml
model:
  resume_from: results/sft/gsm8k_sft_20240101/checkpoint-500

# Or use latest
trainer:
  resume_from_checkpoint: latest
```

## Algorithm Details

### SFT (Supervised Finetuning)
Standard language modeling on prompt-response pairs. Masks prompt tokens to only train on response.

### DPO (Direct Preference Optimization)
Learns from preference pairs (chosen vs rejected) without a reward model:
```
Loss = -log σ(β * (log π(chosen) - log π(rejected) - log π_ref(chosen) + log π_ref(rejected)))
```

### PPO (Proximal Policy Optimization)
Classic RL algorithm with:
- Value model for advantage estimation
- KL penalty to stay close to reference
- Clipped policy updates

### GRPO (Group Relative Policy Optimization)
Simplified RL for verifiable tasks:
- No value model needed
- Samples N responses per prompt
- Uses group mean as baseline
- More sample-efficient for tasks with clear correctness signal

## License

MIT
