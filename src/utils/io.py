"""Utilities for configuration, checkpointing, and file I/O."""

import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, asdict, is_dataclass


def load_yaml(path: str) -> dict:
    """Load configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str):
    """Save configuration to YAML file."""
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path: str) -> dict:
    """Load data from JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2):
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=_json_serializer)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for dataclasses and special types."""
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""
    step: int
    epoch: Optional[float] = None
    loss: Optional[float] = None
    metrics: Optional[dict] = None
    timestamp: Optional[str] = None


def create_run_dir(base_dir: str, run_name: str) -> Path:
    """
    Create a timestamped run directory.
    
    Args:
        base_dir: Base output directory
        run_name: Name for this run
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint_info(
    checkpoint_dir: str,
    step: int,
    epoch: Optional[float] = None,
    loss: Optional[float] = None,
    metrics: Optional[dict] = None,
):
    """
    Save checkpoint metadata.
    
    Args:
        checkpoint_dir: Directory where checkpoint is saved
        step: Training step
        epoch: Current epoch (optional)
        loss: Current loss (optional)
        metrics: Additional metrics (optional)
    """
    info = CheckpointInfo(
        step=step,
        epoch=epoch,
        loss=loss,
        metrics=metrics,
        timestamp=datetime.now().isoformat(),
    )
    
    info_path = Path(checkpoint_dir) / "checkpoint_info.json"
    save_json(asdict(info), str(info_path))


def load_checkpoint_info(checkpoint_dir: str) -> Optional[CheckpointInfo]:
    """
    Load checkpoint metadata.
    
    Args:
        checkpoint_dir: Directory containing checkpoint
        
    Returns:
        CheckpointInfo or None if not found
    """
    info_path = Path(checkpoint_dir) / "checkpoint_info.json"
    
    if not info_path.exists():
        return None
    
    data = load_json(str(info_path))
    return CheckpointInfo(**data)


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in output directory.
    
    Looks for directories named 'checkpoint-*' and returns the one
    with the highest step number.
    
    Args:
        output_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None
    """
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return None
    
    checkpoints = list(output_path.glob("checkpoint-*"))
    
    if not checkpoints:
        return None
    
    # Sort by step number
    def get_step(p: Path) -> int:
        try:
            return int(p.name.split("-")[-1])
        except ValueError:
            return 0
    
    latest = max(checkpoints, key=get_step)
    return str(latest)


def save_training_config(
    output_dir: str,
    config: dict,
    filename: str = "training_config.yaml",
):
    """
    Save training configuration to output directory.
    
    Args:
        output_dir: Output directory
        config: Configuration dictionary
        filename: Config filename
    """
    config_path = Path(output_dir) / filename
    save_yaml(config, str(config_path))
    print(f"Training config saved to {config_path}")


def load_training_config(
    output_dir: str,
    filename: str = "training_config.yaml",
) -> Optional[dict]:
    """
    Load training configuration from output directory.
    
    Args:
        output_dir: Output directory
        filename: Config filename
        
    Returns:
        Configuration dict or None
    """
    config_path = Path(output_dir) / filename
    
    if not config_path.exists():
        return None
    
    return load_yaml(str(config_path))


def save_training_state(
    output_dir: str,
    state: dict,
    filename: str = "training_state.json",
):
    """
    Save training state (for resuming).
    
    Args:
        output_dir: Output directory
        state: State dictionary
        filename: State filename
    """
    state_path = Path(output_dir) / filename
    save_json(state, str(state_path))


def load_training_state(
    output_dir: str,
    filename: str = "training_state.json",
) -> Optional[dict]:
    """
    Load training state.
    
    Args:
        output_dir: Output directory
        filename: State filename
        
    Returns:
        State dict or None
    """
    state_path = Path(output_dir) / filename
    
    if not state_path.exists():
        return None
    
    return load_json(str(state_path))
