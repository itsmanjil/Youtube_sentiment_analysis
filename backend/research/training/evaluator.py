"""
Comprehensive Evaluation Module for Sentiment Analysis Models

Provides thesis-level metrics computation during training:
- Accuracy, Precision, Recall, F1 (macro/micro/per-class)
- Cohen's Kappa
- Confusion Matrix
- Loss tracking

Integrates with existing evaluation_framework.py for consistency.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    classification_report
)
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class SentimentEvaluator:
    """
    Comprehensive evaluator for sentiment analysis models

    Computes all thesis-level metrics following the patterns from
    evaluation_framework.py for consistency across experiments.

    Args:
        num_classes: Number of sentiment classes (default: 3)
        class_names: List of class names (default: ['Negative', 'Neutral', 'Positive'])

    Example:
        >>> evaluator = SentimentEvaluator()
        >>> for batch in dataloader:
        ...     predictions = model(batch['input_ids'])
        ...     evaluator.update(predictions, batch['labels'])
        >>> metrics = evaluator.compute()
        >>> evaluator.reset()
    """

    def __init__(
        self,
        num_classes: int = 3,
        class_names: Optional[List[str]] = None
    ):
        self.num_classes = num_classes

        if class_names is None:
            self.class_names = ['Negative', 'Neutral', 'Positive']
        else:
            self.class_names = class_names

        self.reset()

    def reset(self):
        """Reset all accumulated predictions and labels"""
        self.all_predictions = []
        self.all_labels = []
        self.all_losses = []

    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        loss: Optional[float] = None
    ):
        """
        Accumulate predictions and labels for batch

        Args:
            predictions: Model predictions (logits or probabilities) [batch_size, num_classes]
            labels: True labels [batch_size]
            loss: Optional loss value for this batch
        """
        # Convert logits to predicted classes
        if predictions.dim() == 2:
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions

        # Move to CPU and convert to numpy
        pred_classes = pred_classes.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        self.all_predictions.extend(pred_classes)
        self.all_labels.extend(labels)

        if loss is not None:
            self.all_losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated predictions

        Returns:
            Dictionary with all metrics matching evaluation_framework.py format:
            - accuracy
            - precision_macro, recall_macro, f1_macro
            - precision_micro, recall_micro, f1_micro
            - cohen_kappa
            - per_class_f1_{class_name}
            - avg_loss (if losses were tracked)
        """
        if not self.all_predictions:
            return {}

        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)

        metrics = {}

        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Macro-averaged metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics['precision_macro'] = precision_macro
        metrics['recall_macro'] = recall_macro
        metrics['f1_macro'] = f1_macro

        # Micro-averaged metrics
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )
        metrics['precision_micro'] = precision_micro
        metrics['recall_micro'] = recall_micro
        metrics['f1_micro'] = f1_micro

        # Cohen's Kappa (inter-rater agreement)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # Per-class F1 scores
        _, _, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        for class_idx, class_name in enumerate(self.class_names):
            if class_idx < len(f1_per_class):
                metrics[f'f1_{class_name.lower()}'] = f1_per_class[class_idx]

        # Average loss
        if self.all_losses:
            metrics['avg_loss'] = np.mean(self.all_losses)

        return metrics

    def get_confusion_matrix(self, normalize: bool = False) -> np.ndarray:
        """
        Get confusion matrix

        Args:
            normalize: If True, normalize by row (true label)

        Returns:
            Confusion matrix [num_classes, num_classes]
        """
        if not self.all_predictions:
            return np.zeros((self.num_classes, self.num_classes))

        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        return cm

    def get_classification_report(self) -> str:
        """
        Get detailed classification report

        Returns:
            Formatted classification report string
        """
        if not self.all_predictions:
            return "No predictions to report"

        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)

        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4,
            zero_division=0
        )

    def print_summary(self):
        """Print comprehensive evaluation summary"""
        metrics = self.compute()

        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)

        # Main metrics
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:       {metrics.get('accuracy', 0):.4f}")
        print(f"  F1-Macro:       {metrics.get('f1_macro', 0):.4f}")
        print(f"  Precision-Macro: {metrics.get('precision_macro', 0):.4f}")
        print(f"  Recall-Macro:   {metrics.get('recall_macro', 0):.4f}")
        print(f"  Cohen's Kappa:  {metrics.get('cohen_kappa', 0):.4f}")

        # Per-class metrics
        print(f"\nPer-Class F1 Scores:")
        for class_name in self.class_names:
            key = f'f1_{class_name.lower()}'
            if key in metrics:
                print(f"  {class_name:10s}: {metrics[key]:.4f}")

        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = self.get_confusion_matrix()
        print(f"{'':12s}", end='')
        for name in self.class_names:
            print(f"{name:12s}", end='')
        print()

        for i, name in enumerate(self.class_names):
            print(f"{name:12s}", end='')
            for j in range(self.num_classes):
                print(f"{cm[i, j]:12.0f}", end='')
            print()

        # Classification report
        print(f"\nDetailed Classification Report:")
        print(self.get_classification_report())

        print("="*80 + "\n")


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 3,
    class_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on a dataset

    Convenience function for one-time evaluation without managing evaluator state.

    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader with evaluation data
        criterion: Loss function
        device: Device to run evaluation on
        num_classes: Number of classes
        class_names: List of class names
        verbose: Print evaluation summary

    Returns:
        Dictionary with all computed metrics

    Example:
        >>> metrics = evaluate_model(
        ...     model, val_loader, criterion, device,
        ...     verbose=True
        ... )
        >>> print(f"Validation F1: {metrics['f1_macro']:.4f}")
    """
    model.eval()
    evaluator = SentimentEvaluator(num_classes, class_names)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch.get('lengths', None)

            # Forward pass
            if lengths is not None:
                logits, _ = model(input_ids, lengths)
            else:
                logits, _ = model(input_ids)

            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            # Update evaluator
            evaluator.update(logits, labels, loss.item())

    # Compute metrics
    metrics = evaluator.compute()
    metrics['avg_loss'] = total_loss / num_batches if num_batches > 0 else 0.0

    # Print summary if requested
    if verbose:
        evaluator.print_summary()

    return metrics


class MetricsTracker:
    """
    Track metrics across multiple epochs

    Useful for monitoring training progress and generating plots.

    Example:
        >>> tracker = MetricsTracker()
        >>> for epoch in range(num_epochs):
        ...     train_metrics = evaluate_model(model, train_loader, ...)
        ...     val_metrics = evaluate_model(model, val_loader, ...)
        ...     tracker.add_epoch('train', train_metrics)
        ...     tracker.add_epoch('val', val_metrics)
        >>> tracker.plot(['f1_macro', 'accuracy'])
    """

    def __init__(self):
        self.metrics = defaultdict(lambda: defaultdict(list))

    def add_epoch(self, split: str, metrics: Dict[str, float]):
        """
        Add metrics for an epoch

        Args:
            split: 'train', 'val', or 'test'
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            self.metrics[split][key].append(float(value))

    def get_metric(self, split: str, metric: str) -> List[float]:
        """Get history of a specific metric"""
        return self.metrics[split].get(metric, [])

    def get_best_epoch(self, split: str, metric: str, mode: str = 'max') -> Tuple[int, float]:
        """
        Find epoch with best metric value

        Args:
            split: 'train', 'val', or 'test'
            metric: Metric name
            mode: 'max' or 'min'

        Returns:
            Tuple of (best_epoch, best_value)
        """
        values = self.get_metric(split, metric)

        if not values:
            return -1, float('inf') if mode == 'min' else float('-inf')

        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)

        return best_idx, values[best_idx]

    def print_summary(self):
        """Print summary of tracked metrics"""
        print("\n" + "="*80)
        print("METRICS TRACKING SUMMARY")
        print("="*80)

        for split in self.metrics:
            print(f"\n{split.upper()}:")

            for metric, values in self.metrics[split].items():
                if values:
                    final = values[-1]
                    best_epoch, best_value = self.get_best_epoch(
                        split, metric,
                        mode='min' if 'loss' in metric else 'max'
                    )

                    print(f"  {metric:20s}: Final={final:.4f}, "
                          f"Best={best_value:.4f} (Epoch {best_epoch + 1})")

        print("="*80 + "\n")


# Example usage
if __name__ == "__main__":
    """
    Test evaluator with mock data

    Usage:
        cd backend/research
        python -m training.evaluator
    """
    print("Testing SentimentEvaluator...\n")

    # Mock predictions and labels
    np.random.seed(42)

    # Simulate 100 samples, 3 classes
    predictions = torch.tensor(np.random.rand(100, 3))
    labels = torch.tensor(np.random.randint(0, 3, 100))

    print("1. Testing SentimentEvaluator...")
    evaluator = SentimentEvaluator()

    # Update with batches
    for i in range(0, 100, 32):
        batch_preds = predictions[i:i+32]
        batch_labels = labels[i:i+32]
        evaluator.update(batch_preds, batch_labels, loss=0.5)

    # Compute metrics
    metrics = evaluator.compute()
    print(f"   Computed {len(metrics)} metrics")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   F1-Macro: {metrics['f1_macro']:.4f}")

    # Print full summary
    evaluator.print_summary()

    print("\n2. Testing MetricsTracker...")
    tracker = MetricsTracker()

    # Simulate 5 epochs
    for epoch in range(5):
        train_metrics = {
            'accuracy': 0.6 + epoch * 0.05,
            'f1_macro': 0.55 + epoch * 0.06,
            'avg_loss': 1.0 - epoch * 0.15
        }
        val_metrics = {
            'accuracy': 0.58 + epoch * 0.04,
            'f1_macro': 0.53 + epoch * 0.05,
            'avg_loss': 1.1 - epoch * 0.12
        }

        tracker.add_epoch('train', train_metrics)
        tracker.add_epoch('val', val_metrics)

    tracker.print_summary()

    # Find best epoch
    best_epoch, best_f1 = tracker.get_best_epoch('val', 'f1_macro', mode='max')
    print(f"\n   Best validation F1: {best_f1:.4f} at epoch {best_epoch + 1}")

    print("\n✅ All evaluator tests passed!")
