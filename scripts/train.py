#!/usr/bin/env python3
"""
Main entry point for training.

Usage:
    python scripts/train.py --config config/sft/gsm8k.yaml
    python scripts/train.py --config config/grpo/math_outcome.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runners import TrainingRunner, run_from_config
from src.utils import load_yaml


def main():
    parser = argparse.ArgumentParser(
        description="Train LLM with SFT or RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="Override config values (format: key=value)",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_yaml(args.config)
    
    # Apply overrides
    if args.override:
        for override in args.override:
            if "=" not in override:
                print(f"Warning: Invalid override format: {override}")
                continue
            
            key, value = override.split("=", 1)
            keys = key.split(".")
            
            # Navigate to nested key
            target = config
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            
            # Try to parse value
            try:
                # Try as int
                value = int(value)
            except ValueError:
                try:
                    # Try as float
                    value = float(value)
                except ValueError:
                    # Try as bool
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    # Otherwise keep as string
            
            target[keys[-1]] = value
            print(f"Override: {key} = {value}")
    
    # Print config summary
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Model: {config.get('model', {}).get('name', 'N/A')}")
    print(f"Trainer: {config.get('trainer', {}).get('type', 'sft')}")
    print(f"Dataset: {config.get('data', {}).get('name', 'N/A')}")
    print(f"Output: {config.get('trainer', {}).get('output_dir', 'results')}")
    print("=" * 60 + "\n")
    
    # Run training
    runner = TrainingRunner(config)
    result = runner.run()
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Output directory: {result.output_dir}")
    print(f"Final checkpoint: {result.final_checkpoint}")
    print(f"Total steps: {result.total_steps}")
    print(f"Final loss: {result.final_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
