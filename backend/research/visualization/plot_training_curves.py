"""
Training Curve Visualization for Hybrid CNN-BiLSTM Model

Generates publication-quality plots of:
- Training/validation loss curves
- Training/validation accuracy curves  
- Training/validation F1-macro curves
- Per-class F1 scores
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def load_training_history(history_path):
    """Load training history from JSON file."""
    with open(history_path, 'r') as f:
        data = json.load(f)
    return data['history']


def plot_metric(ax, history, train_key, val_key, ylabel, title):
    """Plot a single metric with train/val curves."""
    epochs = range(1, len(history[train_key]) + 1)
    
    # Plot training and validation
    ax.plot(epochs, history[train_key], 'o-', label='Training', linewidth=2, markersize=6)
    ax.plot(epochs, history[val_key], 's-', label='Validation', linewidth=2, markersize=6)
    
    # Mark best validation epoch
    best_epoch = np.argmax(history[val_key]) + 1
    best_val = max(history[val_key])
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.3, label=f'Best (Epoch {best_epoch})')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add best value annotation
    ax.annotate(f'Best: {best_val:.4f}', 
                xy=(best_epoch, best_val), 
                xytext=(10, 10), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))


def plot_training_curves(history_path, output_dir):
    """Generate comprehensive training curve plots."""
    history = load_training_history(history_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Progress - Hybrid CNN-BiLSTM Model', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    plot_metric(axes[0, 0], history, 'train_avg_loss', 'val_avg_loss', 
                'Loss', 'Training and Validation Loss')
    
    # Plot 2: Accuracy
    plot_metric(axes[0, 1], history, 'train_accuracy', 'val_accuracy',
                'Accuracy', 'Training and Validation Accuracy')
    
    # Plot 3: F1-Macro
    plot_metric(axes[1, 0], history, 'train_f1_macro', 'val_f1_macro',
                'F1-Macro Score', 'Training and Validation F1-Macro')
    
    # Plot 4: Per-Class F1 (Validation only)
    epochs = range(1, len(history['val_f1_negative']) + 1)
    axes[1, 1].plot(epochs, history['val_f1_negative'], 'o-', label='Negative', linewidth=2)
    axes[1, 1].plot(epochs, history['val_f1_neutral'], 's-', label='Neutral', linewidth=2)
    axes[1, 1].plot(epochs, history['val_f1_positive'], '^-', label='Positive', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Per-Class F1 Scores (Validation)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved training curves to: {output_path}')
    
    plt.close()
    
    # Create separate high-res plots for each metric
    metrics = [
        ('train_avg_loss', 'val_avg_loss', 'Loss', 'loss_curve.png'),
        ('train_accuracy', 'val_accuracy', 'Accuracy', 'accuracy_curve.png'),
        ('train_f1_macro', 'val_f1_macro', 'F1-Macro', 'f1_curve.png'),
    ]
    
    for train_key, val_key, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_metric(ax, history, train_key, val_key, ylabel, 
                   f'{ylabel} - Training vs Validation')
        plt.tight_layout()
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved {ylabel} curve to: {output_path}')
        plt.close()
    
    # Summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    best_epoch = np.argmax(history['val_f1_macro']) + 1
    print(f"\nBest Epoch: {best_epoch}")
    print(f"  Val F1-Macro: {history['val_f1_macro'][best_epoch-1]:.4f}")
    print(f"  Val Accuracy: {history['val_accuracy'][best_epoch-1]:.4f}")
    print(f"  Val Loss:     {history['val_avg_loss'][best_epoch-1]:.4f}")
    
    print(f"\nFinal Epoch: {len(history['val_f1_macro'])}")
    print(f"  Val F1-Macro: {history['val_f1_macro'][-1]:.4f}")
    print(f"  Val Accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"  Val Loss:     {history['val_avg_loss'][-1]:.4f}")
    
    print("\nPer-Class F1 (Best Epoch):")
    print(f"  Negative: {history['val_f1_negative'][best_epoch-1]:.4f}")
    print(f"  Neutral:  {history['val_f1_neutral'][best_epoch-1]:.4f}")
    print(f"  Positive: {history['val_f1_positive'][best_epoch-1]:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training curves')
    parser.add_argument('--history', required=True, help='Path to training_history.json')
    parser.add_argument('--output', default='./plots', help='Output directory for plots')
    args = parser.parse_args()
    
    plot_training_curves(args.history, args.output)
