"""Local logging utilities for training."""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class LogEntry:
    """Single log entry."""
    step: int
    timestamp: str
    metrics: dict
    phase: str = "train"  # train, eval, etc.


class LocalLogger:
    """
    Simple local logger for training metrics.
    
    Logs to:
    - Console (real-time)
    - JSON file (all metrics)
    - Text file (human-readable summary)
    """
    
    def __init__(
        self,
        output_dir: str,
        log_interval: int = 10,
        verbose: bool = True,
    ):
        """
        Initialize logger.
        
        Args:
            output_dir: Directory for log files
            log_interval: Steps between console logs
            verbose: Whether to print to console
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_interval = log_interval
        self.verbose = verbose
        
        # Log files
        self.json_log_path = self.output_dir / "training_log.jsonl"
        self.text_log_path = self.output_dir / "training_log.txt"
        
        # Metrics history (for averaging)
        self._metrics_buffer = {}
        self._last_log_step = 0
        
        # Initialize log files
        self._init_log_files()
    
    def _init_log_files(self):
        """Initialize log files with header."""
        with open(self.text_log_path, "w") as f:
            f.write(f"Training started at {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")
    
    def log(
        self,
        step: int,
        metrics: dict,
        phase: str = "train",
        force: bool = False,
    ):
        """
        Log metrics.
        
        Args:
            step: Current training step
            metrics: Dictionary of metrics
            phase: train, eval, etc.
            force: Force logging regardless of interval
        """
        # Add to buffer
        for key, value in metrics.items():
            if key not in self._metrics_buffer:
                self._metrics_buffer[key] = []
            self._metrics_buffer[key].append(value)
        
        # Check if should log
        should_log = force or (step - self._last_log_step >= self.log_interval)
        
        if not should_log:
            return
        
        # Average buffered metrics
        avg_metrics = {}
        for key, values in self._metrics_buffer.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        # Create entry
        entry = LogEntry(
            step=step,
            timestamp=datetime.now().isoformat(),
            metrics=avg_metrics,
            phase=phase,
        )
        
        # Write to JSON log
        with open(self.json_log_path, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
        
        # Write to text log
        self._write_text_log(entry)
        
        # Print to console
        if self.verbose:
            self._print_console(entry)
        
        # Clear buffer
        self._metrics_buffer = {}
        self._last_log_step = step
    
    def log_eval(self, step: int, metrics: dict):
        """Log evaluation metrics."""
        self.log(step, metrics, phase="eval", force=True)
    
    def _write_text_log(self, entry: LogEntry):
        """Write entry to text log file."""
        with open(self.text_log_path, "a") as f:
            f.write(f"[{entry.phase.upper()}] Step {entry.step}\n")
            for key, value in entry.metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    def _print_console(self, entry: LogEntry):
        """Print entry to console."""
        metrics_str = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in entry.metrics.items()
        )
        
        phase_prefix = f"[{entry.phase.upper()}]" if entry.phase != "train" else ""
        print(f"Step {entry.step} {phase_prefix} {metrics_str}")
    
    def log_message(self, message: str):
        """Log a text message."""
        timestamp = datetime.now().isoformat()
        
        # Write to text log
        with open(self.text_log_path, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        
        # Print to console
        if self.verbose:
            print(message)
    
    def log_config(self, config: dict):
        """Log training configuration."""
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        # Also write to text log
        with open(self.text_log_path, "a") as f:
            f.write("Configuration:\n")
            f.write(json.dumps(config, indent=2, default=str))
            f.write("\n\n")
    
    def get_history(self, phase: str = "train") -> list[LogEntry]:
        """
        Load log history from file.
        
        Args:
            phase: Filter by phase (train, eval, etc.)
            
        Returns:
            List of LogEntry
        """
        entries = []
        
        if not self.json_log_path.exists():
            return entries
        
        with open(self.json_log_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entry = LogEntry(**data)
                    if phase is None or entry.phase == phase:
                        entries.append(entry)
        
        return entries
    
    def finish(self):
        """Finalize logging."""
        with open(self.text_log_path, "a") as f:
            f.write("=" * 60 + "\n")
            f.write(f"Training finished at {datetime.now().isoformat()}\n")
        
        if self.verbose:
            print(f"\nLogs saved to {self.output_dir}")


class ProgressTracker:
    """Track training progress with ETA estimation."""
    
    def __init__(self, total_steps: int):
        """
        Initialize tracker.
        
        Args:
            total_steps: Total number of training steps
        """
        self.total_steps = total_steps
        self.start_time = datetime.now()
        self.current_step = 0
    
    def update(self, step: int):
        """Update current step."""
        self.current_step = step
    
    @property
    def progress(self) -> float:
        """Progress as fraction (0-1)."""
        return self.current_step / self.total_steps if self.total_steps > 0 else 0
    
    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated time remaining in seconds."""
        if self.current_step == 0:
            return None
        
        rate = self.current_step / self.elapsed_seconds
        remaining_steps = self.total_steps - self.current_step
        return remaining_steps / rate if rate > 0 else None
    
    def format_eta(self) -> str:
        """Format ETA as human-readable string."""
        eta = self.eta_seconds
        if eta is None:
            return "calculating..."
        
        hours, remainder = divmod(int(eta), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def get_status(self) -> str:
        """Get formatted status string."""
        pct = self.progress * 100
        return f"{self.current_step}/{self.total_steps} ({pct:.1f}%) | ETA: {self.format_eta()}"
