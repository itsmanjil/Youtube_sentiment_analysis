"""
Comprehensive Trainer for Hybrid CNN-BiLSTM-Attention Model

Provides complete training pipeline with:
- Training and validation loops
- TensorBoard logging
- Callback system (early stopping, checkpointing, LR scheduling)
- Gradient clipping and mixed precision training
- Comprehensive metrics tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception as exc:  # TensorBoard can fail due to protobuf version mismatches.
    SummaryWriter = None
    _TENSORBOARD_IMPORT_ERROR = exc
else:
    _TENSORBOARD_IMPORT_ERROR = None
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import time

from .evaluator import SentimentEvaluator, evaluate_model
from .callbacks import Callback


class _NoOpSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_scalars(self, *args, **kwargs):
        return None

    def close(self):
        return None


class HybridDLTrainer:
    """
    Comprehensive trainer for sentiment analysis models

    Features:
    - Training and validation loops with automatic metrics computation
    - TensorBoard logging for all metrics and learning curves
    - Callback system for modularity (early stopping, checkpointing, etc.)
    - Gradient clipping to prevent exploding gradients
    - Optional mixed precision training for faster training
    - Attention weight visualization (specific to hybrid model)

    Args:
        model: PyTorch model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader (optional)
        criterion: Loss function (default: CrossEntropyLoss)
        optimizer: Optimizer (required)
        scheduler: Learning rate scheduler (optional)
        device: Device to train on ('cuda', 'cpu', or 'mps')
        max_epochs: Maximum number of epochs
        gradient_clip: Maximum gradient norm (0 = no clipping)
        tensorboard_log_dir: TensorBoard log directory
        mixed_precision: Use mixed precision training (requires CUDA)
        callbacks: List of callback objects
        config: Configuration dict to save with checkpoints

    Example:
        >>> trainer = HybridDLTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     optimizer=optimizer,
        ...     device='cuda',
        ...     max_epochs=50,
        ...     callbacks=[early_stop, checkpoint, lr_scheduler]
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cuda',
        max_epochs: int = 50,
        gradient_clip: float = 5.0,
        tensorboard_log_dir: str = './runs',
        mixed_precision: bool = False,
        callbacks: Optional[List[Callback]] = None,
        config: Optional[Dict] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.gradient_clip = gradient_clip
        self.config = config or {}
        self.callbacks = callbacks or []

        # Device setup
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Loss function
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(tensorboard_log_dir) / f'hybrid_dl_{timestamp}'
        if SummaryWriter is None:
            self.writer = _NoOpSummaryWriter()
            print("WARNING: TensorBoard logging disabled.")
            print(f"         Import error: {_TENSORBOARD_IMPORT_ERROR}\n")
        else:
            self.writer = SummaryWriter(log_dir=str(log_dir))
            print(f"TensorBoard logging to: {log_dir}")
            print(f"   Run: tensorboard --logdir={tensorboard_log_dir}\n")

        # Mixed precision training
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision training enabled\n")

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.stop_training = False
        self.training_history = []

        # Evaluators
        self.train_evaluator = SentimentEvaluator()
        self.val_evaluator = SentimentEvaluator()

    def train(self):
        """
        Main training loop

        Trains the model for max_epochs, calling callbacks at appropriate times.
        """
        print("="*80)
        print("STARTING TRAINING")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Batches per epoch: {len(self.train_loader)}")
        print(f"Max epochs: {self.max_epochs}")
        print("="*80 + "\n")

        # Callbacks: training begin
        for callback in self.callbacks:
            callback.on_train_begin(self)

        try:
            for epoch in range(self.max_epochs):
                self.current_epoch = epoch

                # Callbacks: epoch begin
                for callback in self.callbacks:
                    callback.on_epoch_begin(epoch, self)

                # Training phase
                train_metrics = self._train_epoch(epoch)

                # Validation phase
                if self.val_loader is not None:
                    val_metrics = self._validate_epoch(epoch)
                else:
                    val_metrics = {}

                # Combine metrics
                all_metrics = {
                    **{f'train_{k}': v for k, v in train_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }
                self.training_history.append({'epoch': epoch, **all_metrics})

                # Log to TensorBoard
                self._log_metrics(epoch, all_metrics)

                # Callbacks: epoch end
                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, all_metrics, self)

                # Check for early stopping
                if self.stop_training:
                    print(f"\nTraining stopped early at epoch {epoch + 1}")
                    break

        except KeyboardInterrupt:
            print("\nWARNING: Training interrupted by user")

        finally:
            # Callbacks: training end
            for callback in self.callbacks:
                callback.on_train_end(self)

            self.writer.close()

            print("\n" + "="*80)
            print("TRAINING COMPLETE")
            print("="*80 + "\n")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.train_evaluator.reset()

        epoch_start_time = time.time()
        running_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            # Callbacks: batch begin
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx, self)

            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch.get('lengths', None)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass (with mixed precision if enabled)
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    if lengths is not None:
                        logits, attention_weights = self.model(input_ids, lengths)
                    else:
                        logits, attention_weights = self.model(input_ids)

                    loss = self.criterion(logits, labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Standard training (no mixed precision)
                if lengths is not None:
                    logits, attention_weights = self.model(input_ids, lengths)
                else:
                    logits, attention_weights = self.model(input_ids)

                loss = self.criterion(logits, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )

                # Optimizer step
                self.optimizer.step()

            # Track metrics
            batch_loss = loss.item()
            running_loss += batch_loss
            self.train_evaluator.update(logits, labels, batch_loss)

            # Log batch metrics to TensorBoard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('batch/loss', batch_loss, self.global_step)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('batch/learning_rate', current_lr, self.global_step)

            self.global_step += 1

            # Callbacks: batch end
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, batch_loss, self)

        # Compute epoch metrics
        train_metrics = self.train_evaluator.compute()
        train_metrics['epoch_time'] = time.time() - epoch_start_time

        return train_metrics

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Validate for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        self.val_evaluator.reset()

        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch.get('lengths', None)

                # Forward pass
                if lengths is not None:
                    logits, attention_weights = self.model(input_ids, lengths)
                else:
                    logits, attention_weights = self.model(input_ids)

                loss = self.criterion(logits, labels)

                # Track metrics
                self.val_evaluator.update(logits, labels, loss.item())

        # Compute validation metrics
        val_metrics = self.val_evaluator.compute()

        return val_metrics

    def _log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics to TensorBoard

        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics to log
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                # Determine metric group for organization
                if key.startswith('train_'):
                    group = 'train'
                    metric_name = key[6:]  # Remove 'train_' prefix
                elif key.startswith('val_'):
                    group = 'validation'
                    metric_name = key[4:]  # Remove 'val_' prefix
                else:
                    group = 'other'
                    metric_name = key

                self.writer.add_scalar(f'{group}/{metric_name}', value, epoch)

        # Log combined train/val curves for key metrics
        if 'train_f1_macro' in metrics and 'val_f1_macro' in metrics:
            self.writer.add_scalars('comparison/f1_macro', {
                'train': metrics['train_f1_macro'],
                'val': metrics['val_f1_macro']
            }, epoch)

        if 'train_avg_loss' in metrics and 'val_avg_loss' in metrics:
            self.writer.add_scalars('comparison/loss', {
                'train': metrics['train_avg_loss'],
                'val': metrics['val_avg_loss']
            }, epoch)

    def evaluate_on_test(
        self,
        test_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on test set

        Args:
            test_loader: Test DataLoader
            verbose: Print evaluation summary

        Returns:
            Dictionary with test metrics
        """
        return evaluate_model(
            self.model,
            test_loader,
            self.criterion,
            self.device,
            verbose=verbose
        )

    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        metrics: Optional[Dict] = None
    ):
        """
        Save training checkpoint

        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch
            metrics: Optional metrics to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics or {}
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)

        print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True):
        """
        Load training checkpoint

        Args:
            filepath: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)

        print(f"Checkpoint loaded: {filepath}")
        print(f"   Resuming from epoch {self.current_epoch + 1}")


# Example usage
if __name__ == "__main__":
    """
    Test trainer with mock model and data

    Usage:
        cd backend/research
        python -m training.trainer
    """
    print("Testing HybridDLTrainer...\n")

    # Mock model
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 128)
            self.classifier = nn.Linear(128, 3)

        def forward(self, input_ids, lengths=None):
            embedded = self.embedding(input_ids)
            pooled = embedded.mean(dim=1)
            logits = self.classifier(pooled)
            return logits, None  # Return None for attention weights

    # Mock data
    from torch.utils.data import TensorDataset, DataLoader

    train_data = TensorDataset(
        torch.randint(0, 1000, (200, 50)),  # input_ids
        torch.randint(0, 3, (200,))  # labels
    )

    val_data = TensorDataset(
        torch.randint(0, 1000, (50, 50)),
        torch.randint(0, 3, (50,))
    )

    def collate_fn(batch):
        input_ids = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return {
            'input_ids': input_ids,
            'labels': labels,
            'lengths': torch.full((len(batch),), 50)
        }

    train_loader = DataLoader(train_data, batch_size=32, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=32, collate_fn=collate_fn)

    # Create model and optimizer
    model = MockModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create callbacks
    from .callbacks import EarlyStopping, ModelCheckpoint, ProgressLogger

    callbacks = [
        ProgressLogger(print_freq=2),
        EarlyStopping(monitor='val_f1_macro', patience=3, mode='max'),
        ModelCheckpoint(
            filepath='test_checkpoint.pt',
            monitor='val_f1_macro',
            mode='max'
        )
    ]

    # Create trainer
    trainer = HybridDLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device='cpu',
        max_epochs=5,
        callbacks=callbacks,
        tensorboard_log_dir='./test_runs'
    )

    # Train
    trainer.train()

    print("\nTrainer test complete!")

    # Clean up
    import shutil
    Path('test_checkpoint.pt').unlink(missing_ok=True)
    if Path('test_runs').exists():
        shutil.rmtree('test_runs')
