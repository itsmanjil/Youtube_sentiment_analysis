"""
Confusion Matrix Visualization for Sentiment Analysis Models

Generates confusion matrices for:
- Individual models (LogReg, SVM, Hybrid, Ensemble)
- Comparative visualization of all models
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.sentiment import get_sentiment_engine
import pandas as pd


def plot_confusion_matrix(cm, labels, title, output_path, normalize=True):
    """Plot a single confusion matrix."""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        vmax = 1.0
    else:
        fmt = 'd'
        vmax = None
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=vmax, cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved confusion matrix to: {output_path}')
    plt.close()


def generate_confusion_matrices(test_csv, models, output_dir):
    """Generate confusion matrices for all models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    df = pd.read_csv(test_csv)
    texts = df['text'].tolist()[:1500]  # Sample for speed
    
    # Normalize labels
    label_map = {'Positive': 'Positive', 'Negative': 'Negative', 'Neutral': 'Neutral',
                 'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'}
    true_labels = [label_map.get(l, 'Neutral') for l in df['label'].tolist()[:1500]]
    
    labels_order = ['Negative', 'Neutral', 'Positive']
    
    print(f"Generating confusion matrices for {len(texts)} samples...")
    print(f"Models: {', '.join(models)}\n")
    
    all_cms = {}
    
    for model_name in models:
        print(f"Processing {model_name}...")
        
        try:
            # Load model
            if model_name == 'ensemble':
                # Use PSO weights
                try:
                    with open('results/pso_ensemble_weights.json') as f:
                        weights = json.load(f)['weights']
                    engine = get_sentiment_engine('ensemble',
                                                 base_models=['logreg', 'svm', 'tfidf'],
                                                 weights=weights)
                except:
                    engine = get_sentiment_engine('ensemble')
            else:
                engine = get_sentiment_engine(model_name)
            
            # Get predictions
            predictions = []
            for i, text in enumerate(texts):
                if i % 300 == 0:
                    print(f"  {i}/{len(texts)}...")
                result = engine.analyze(text[:1000])  # Truncate long texts
                predictions.append(result.label)
            
            # Generate confusion matrix
            cm = confusion_matrix(true_labels, predictions, labels=labels_order)
            all_cms[model_name] = cm
            
            # Plot individual confusion matrix
            plot_confusion_matrix(
                cm, labels_order, 
                f'Confusion Matrix - {model_name.upper()}',
                output_dir / f'confusion_matrix_{model_name}.png',
                normalize=True
            )
            
            # Also save raw counts
            plot_confusion_matrix(
                cm, labels_order,
                f'Confusion Matrix (Counts) - {model_name.upper()}',
                output_dir / f'confusion_matrix_{model_name}_counts.png',
                normalize=False
            )
            
        except Exception as e:
            print(f"  Error processing {model_name}: {e}")
            continue
    
    # Create comparison plot (2x2 grid)
    if len(all_cms) >= 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold')
        
        for idx, (model_name, cm) in enumerate(list(all_cms.items())[:4]):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Normalize
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                       xticklabels=labels_order, yticklabels=labels_order,
                       vmin=0, vmax=1.0, ax=ax, cbar_kws={'label': 'Proportion'})
            
            ax.set_title(model_name.upper(), fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        comparison_path = output_dir / 'confusion_matrix_comparison.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f'\nSaved comparison plot to: {comparison_path}')
        plt.close()
    
    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate confusion matrices')
    parser.add_argument('--test_csv', required=True, help='Path to test CSV')
    parser.add_argument('--models', default='logreg,svm,hybrid_dl,ensemble',
                       help='Comma-separated model names')
    parser.add_argument('--output', default='./plots', help='Output directory')
    args = parser.parse_args()
    
    models = [m.strip() for m in args.models.split(',')]
    generate_confusion_matrices(args.test_csv, models, args.output)
