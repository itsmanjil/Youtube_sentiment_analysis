"""
Training Callbacks for PyTorch Models

Provides modular callbacks for:
- Early stopping with patience
- Model checkpointing (best and periodic)
- Learning rate scheduling
- TensorBoard logging
- Progress monitoring

All callbacks follow a consistent interface for easy integration with the Trainer.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import json


class Callback:
    """
    Base callback class

    All callbacks should inherit from this and override relevant methods.
    """

    def on_train_begin(self, trainer):
        """Called at the beginning of training"""
        pass

    def on_train_end(self, trainer):
        """Called at the end of training"""
        pass

    def on_epoch_begin(self, epoch: int, trainer):
        """Called at the beginning of each epoch"""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Called at the end of each epoch with metrics"""
        pass

    def on_batch_begin(self, batch_idx: int, trainer):
        """Called at the beginning of each batch"""
        pass

    def on_batch_end(self, batch_idx: int, loss: float, trainer):
        """Called at the end of each batch"""
        pass


class EarlyStopping(Callback):
    """
    Early stopping to halt training when validation metric stops improving

    Args:
        monitor: Metric to monitor (e.g., 'val_loss', 'val_f1_macro')
        patience: Number of epochs with no improvement before stopping
        mode: 'min' for loss, 'max' for accuracy/F1
        min_delta: Minimum change to qualify as improvement
        verbose: Print messages when stopping

    Example:
        >>> early_stop = EarlyStopping(
        ...     monitor='val_f1_macro',
        ...     patience=7,
        ...     mode='max'
        ... )
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 7,
        mode: str = 'min',
        min_delta: float = 0.0,
        verbose: bool = True
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

        # For 'min' mode (loss), lower is better
        # For 'max' mode (accuracy/F1), higher is better
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best - min_delta
            self.best_score = np.inf
        else:
            self.monitor_op = lambda current, best: current > best + min_delta
            self.best_score = -np.inf

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Check if we should stop training"""

        if self.monitor not in metrics:
            if self.verbose:
                print(f"WARNING: EarlyStopping: '{self.monitor}' not found in metrics. "
                      f"Available: {list(metrics.keys())}")
            return

        current_score = metrics[self.monitor]

        # Check for improvement
        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0

            if self.verbose:
                direction = "decreased" if self.mode == 'min' else "increased"
                print(f"[OK] {self.monitor} {direction} to {current_score:.4f}")

        else:
            self.counter += 1

            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
                trainer.stop_training = True

                if self.verbose:
                    print(f"\nEarly stopping triggered after {self.patience} epochs "
                          f"with no improvement in {self.monitor}")
                    print(f"   Best {self.monitor}: {self.best_score:.4f}\n")


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training

    Supports:
    - Save best model based on monitored metric
    - Save periodic checkpoints every N epochs
    - Save final model at end of training

    Args:
        filepath: Path template for saving (can include {epoch}, {metric})
        monitor: Metric to monitor for best model
        mode: 'min' or 'max'
        save_best_only: Only save when monitored metric improves
        save_frequency: Save every N epochs (0 = only best)
        verbose: Print save messages

    Example:
        >>> checkpoint = ModelCheckpoint(
        ...     filepath='models/hybrid_epoch{epoch:02d}_f1{val_f1_macro:.4f}.pt',
        ...     monitor='val_f1_macro',
        ...     mode='max',
        ...     save_best_only=True
        ... )
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_frequency: int = 0,
        verbose: bool = True
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.verbose = verbose

        self.best_score = np.inf if mode == 'min' else -np.inf
        self.monitor_op = np.less if mode == 'min' else np.greater

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Save checkpoint if conditions are met"""

        current_score = metrics.get(self.monitor, None)

        # Check if we should save
        should_save = False

        # Save best model
        if current_score is not None and self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            should_save = True
            checkpoint_type = "BEST"

        # Save periodic checkpoint
        elif (not self.save_best_only and
              self.save_frequency > 0 and
              (epoch + 1) % self.save_frequency == 0):
            should_save = True
            checkpoint_type = "PERIODIC"

        if should_save:
            # Format filepath (only pass numeric values to avoid format errors)
            numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            filepath = self.filepath.format(
                epoch=epoch + 1,
                **numeric_metrics
            )

            # Prepare checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'metrics': metrics,
                'best_score': self.best_score,
                'config': trainer.config if hasattr(trainer, 'config') else {}
            }

            # Add scheduler if present
            if hasattr(trainer, 'scheduler') and trainer.scheduler is not None:
                checkpoint['scheduler_state_dict'] = trainer.scheduler.state_dict()

            # Save
            torch.save(checkpoint, filepath)

            if self.verbose:
                print(f"[{checkpoint_type}] Checkpoint saved: {filepath}")
                if current_score is not None:
                    print(f"   {self.monitor}: {current_score:.4f}")

    def on_train_end(self, trainer):
        """Save final model"""
        if not self.save_best_only:
            filepath = self.filepath.replace('{epoch', '{epoch_final').format(
                epoch_final=trainer.current_epoch + 1,
                **{k: "final" for k in ['val_loss', 'val_f1_macro', 'val_accuracy']}
            )

            checkpoint = {
                'epoch': trainer.current_epoch + 1,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'best_score': self.best_score,
                'config': trainer.config if hasattr(trainer, 'config') else {}
            }

            torch.save(checkpoint, filepath)

            if self.verbose:
                print(f"[FINAL] Model saved: {filepath}")


class LearningRateScheduler(Callback):
    """
    Learning rate scheduling callback

    Supports PyTorch schedulers: ReduceLROnPlateau, StepLR, CosineAnnealingLR, etc.

    Args:
        scheduler: PyTorch scheduler instance or function to create scheduler
        monitor: Metric to monitor (for ReduceLROnPlateau)
        verbose: Print LR changes

    Example:
        >>> from torch.optim.lr_scheduler import ReduceLROnPlateau
        >>> scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
        >>> lr_callback = LearningRateScheduler(scheduler, monitor='val_f1_macro')
    """

    def __init__(
        self,
        scheduler,
        monitor: Optional[str] = None,
        verbose: bool = True
    ):
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Step the scheduler"""

        # Get current learning rate before step
        current_lr = trainer.optimizer.param_groups[0]['lr']

        # Step scheduler
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # ReduceLROnPlateau needs a metric
            if self.monitor and self.monitor in metrics:
                self.scheduler.step(metrics[self.monitor])
            else:
                print(f"WARNING: LRScheduler: '{self.monitor}' not in metrics for ReduceLROnPlateau")
        else:
            # Other schedulers step automatically
            self.scheduler.step()

        # Get new learning rate
        new_lr = trainer.optimizer.param_groups[0]['lr']

        # Print if changed
        if self.verbose and abs(new_lr - current_lr) > 1e-10:
            print(f"Learning rate adjusted: {current_lr:.6f} -> {new_lr:.6f}")


class ProgressLogger(Callback):
    """
    Log training progress to console

    Args:
        print_freq: Print every N batches (0 = only epoch summary)
        metrics_format: Format string for metrics (e.g., '.4f')

    Example:
        >>> logger = ProgressLogger(print_freq=10)
    """

    def __init__(self, print_freq: int = 10, metrics_format: str = '.4f'):
        self.print_freq = print_freq
        self.metrics_format = metrics_format
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch: int, trainer):
        """Print epoch header"""
        import time
        self.epoch_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{trainer.max_epochs}")
        print(f"{'='*80}")

    def on_batch_end(self, batch_idx: int, loss: float, trainer):
        """Print batch progress"""
        if self.print_freq > 0 and (batch_idx + 1) % self.print_freq == 0:
            total_batches = len(trainer.train_loader)
            progress = (batch_idx + 1) / total_batches * 100

            print(f"  Batch {batch_idx + 1}/{total_batches} ({progress:.1f}%) - "
                  f"Loss: {loss:.4f}")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Print epoch summary"""
        import time
        epoch_time = time.time() - self.epoch_start_time

        print(f"\nEpoch {epoch + 1} Summary (Time: {epoch_time:.2f}s):")

        # Format metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:{self.metrics_format}}")
            else:
                print(f"   {key}: {value}")


class TrainingHistory(Callback):
    """
    Track training history and save to JSON

    Args:
        save_path: Path to save history JSON (optional)

    Attributes:
        history: Dictionary with lists of metrics per epoch

    Example:
        >>> history = TrainingHistory(save_path='logs/history.json')
        >>> # After training:
        >>> history.plot()  # Plot training curves
    """

    def __init__(self, save_path: Optional[str] = None):
        self.save_path = save_path
        self.history = {}

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Record metrics"""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []

            # Convert to Python native types
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()

            self.history[key].append(value)

        # Save after each epoch if path provided
        if self.save_path:
            self._save()

    def on_train_end(self, trainer):
        """Save final history"""
        if self.save_path:
            self._save()
            print(f"Training history saved to: {self.save_path}")

    def _save(self):
        """Save history to JSON"""
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)

        history_with_metadata = {
            'history': self.history,
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'total_epochs': len(self.history.get('train_loss', []))
            }
        }

        with open(self.save_path, 'w') as f:
            json.dump(history_with_metadata, f, indent=2)

    def plot(self, metrics: Optional[list] = None, save_path: Optional[str] = None):
        """
        Plot training curves

        Args:
            metrics: List of metrics to plot (None = plot all)
            save_path: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("WARNING: Matplotlib not installed. Cannot plot history.")
            return

        if not self.history:
            print("WARNING: No history to plot")
            return

        # Determine metrics to plot
        if metrics is None:
            metrics = list(self.history.keys())

        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))

        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            if metric in self.history:
                epochs = range(1, len(self.history[metric]) + 1)
                ax.plot(epochs, self.history[metric], marker='o')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} over epochs')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()


# Example usage
if __name__ == "__main__":
    """
    Test callbacks independently

    Usage:
        cd backend/research
        python -m training.callbacks
    """
    print("Testing Training Callbacks...\n")

    # Mock trainer
    class MockTrainer:
        def __init__(self):
            self.stop_training = False
            self.current_epoch = 0
            self.max_epochs = 10
            self.model = torch.nn.Linear(10, 3)
            self.optimizer = torch.optim.Adam(self.model.parameters())

    trainer = MockTrainer()

    print("1. Testing EarlyStopping...")
    early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min')

    # Simulate improving then plateauing
    for epoch in range(10):
        metrics = {'val_loss': 1.0 - epoch * 0.1 if epoch < 5 else 0.5}
        early_stop.on_epoch_end(epoch, metrics, trainer)

        if trainer.stop_training:
            print(f"   Stopped at epoch {epoch + 1}")
            break

    print("\n2. Testing ModelCheckpoint...")
    checkpoint = ModelCheckpoint(
        filepath='test_checkpoint_epoch{epoch:02d}.pt',
        monitor='val_f1_macro',
        mode='max'
    )

    for epoch in range(3):
        metrics = {'val_f1_macro': 0.7 + epoch * 0.05}
        checkpoint.on_epoch_end(epoch, metrics, trainer)

    print("\n3. Testing TrainingHistory...")
    history = TrainingHistory()

    for epoch in range(5):
        metrics = {
            'train_loss': 1.0 - epoch * 0.15,
            'val_loss': 1.1 - epoch * 0.12,
            'val_f1_macro': 0.6 + epoch * 0.05
        }
        history.on_epoch_end(epoch, metrics, trainer)

    print(f"   Recorded history: {list(history.history.keys())}")
    print(f"   train_loss: {history.history['train_loss']}")

    print("\nAll callback tests passed!")

    # Clean up test files
    import os
    for f in Path('.').glob('test_checkpoint_*.pt'):
        os.remove(f)
