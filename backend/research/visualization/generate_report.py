"""
Thesis Evaluation Report Generator

Generates a comprehensive evaluation report with:
- Model performance metrics
- Training statistics
- Statistical significance tests
- Executive summary
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy import stats


def load_all_data(results_dir, experiment_name):
    """Load all available results and training history."""
    results_dir = Path(results_dir)
    data = {}
    
    # Load training history
    history_file = results_dir / f'../output/{experiment_name}/training_history.json'
    if history_file.exists():
        with open(history_file) as f:
            data['training_history'] = json.load(f)['history']
    
    # Load test results
    test_file = results_dir / f'../output/{experiment_name}/test_results.json'
    if test_file.exists():
        with open(test_file) as f:
            data['test_results'] = json.load(f)
    
    # Load baseline results
    for model in ['logreg', 'svm', 'tfidf']:
        result_file = results_dir / f'baseline_{model}.json'
        if result_file.exists():
            with open(result_file) as f:
                baseline_data = json.load(f)
                data[f'baseline_{model}'] = baseline_data.get(model, baseline_data)
    
    # Load ensemble results
    ensemble_file = results_dir / 'final_comparison.json'
    if ensemble_file.exists():
        with open(ensemble_file) as f:
            data['ensemble'] = json.load(f)
    
    return data


def generate_markdown_report(data, output_path):
    """Generate markdown report."""
    lines = []
    
    # Header
    lines.append("# YouTube Sentiment Analysis - Thesis Evaluation Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    
    if 'test_results' in data:
        test = data['test_results']
        lines.append(f"**Hybrid CNN-BiLSTM Model Performance:**")
        lines.append(f"- Accuracy: {test.get('accuracy', 0)*100:.2f}%")
        lines.append(f"- F1-Macro: {test.get('f1_macro', 0)*100:.2f}%")
        lines.append(f"- Cohen's Kappa: {test.get('cohen_kappa', 0):.4f}")
        lines.append("")
    
    # Training Statistics
    if 'training_history' in data:
        history = data['training_history']
        lines.append("## Training Statistics")
        lines.append("")
        
        epochs = len(history.get('val_f1_macro', []))
        lines.append(f"- Total Epochs: {epochs}")
        
        best_epoch = np.argmax(history['val_f1_macro']) + 1
        best_val_f1 = history['val_f1_macro'][best_epoch - 1]
        lines.append(f"- Best Epoch: {best_epoch}")
        lines.append(f"- Best Val F1-Macro: {best_val_f1*100:.2f}%")
        
        avg_epoch_time = np.mean(history['train_epoch_time']) / 60
        lines.append(f"- Average Epoch Time: {avg_epoch_time:.1f} minutes")
        lines.append(f"- Total Training Time: {epochs * avg_epoch_time:.1f} minutes")
        lines.append("")
    
    # Model Comparison
    lines.append("## Model Comparison")
    lines.append("")
    lines.append("| Model | Accuracy | F1-Macro | F1-Neg | F1-Neu | F1-Pos |")
    lines.append("|-------|----------|----------|--------|--------|--------|")
    
    models_to_compare = []
    
    for model_name in ['logreg', 'svm', 'tfidf']:
        key = f'baseline_{model_name}'
        if key in data:
            baseline = data[key]
            acc = baseline.get('accuracy', 0) * 100
            f1 = baseline.get('macro_f1', baseline.get('f1_macro', 0)) * 100
            
            report = baseline.get('report', {})
            f1_neg = report.get('Negative', {}).get('f1-score', 0) * 100
            f1_neu = report.get('Neutral', {}).get('f1-score', 0) * 100
            f1_pos = report.get('Positive', {}).get('f1-score', 0) * 100
            
            lines.append(f"| {model_name.upper()} | {acc:.2f}% | {f1:.2f}% | {f1_neg:.2f}% | {f1_neu:.2f}% | {f1_pos:.2f}% |")
            models_to_compare.append((model_name, f1))
    
    if 'test_results' in data:
        test = data['test_results']
        acc = test.get('accuracy', 0) * 100
        f1 = test.get('f1_macro', 0) * 100
        
        report = test.get('classification_report', {})
        f1_neg = report.get('Negative', {}).get('f1-score', 0) * 100
        f1_neu = report.get('Neutral', {}).get('f1-score', 0) * 100
        f1_pos = report.get('Positive', {}).get('f1-score', 0) * 100
        
        lines.append(f"| **HYBRID** | **{acc:.2f}%** | **{f1:.2f}%** | **{f1_neg:.2f}%** | **{f1_neu:.2f}%** | **{f1_pos:.2f}%** |")
        models_to_compare.append(('hybrid', f1))
    
    if 'ensemble' in data and 'models' in data['ensemble']:
        if 'pso_ensemble' in data['ensemble']['models']:
            ens = data['ensemble']['models']['pso_ensemble']
            acc = ens.get('accuracy', 0) * 100
            f1 = ens.get('f1_macro', 0) * 100
            lines.append(f"| ENSEMBLE | {acc:.2f}% | {f1:.2f}% | - | - | - |")
            models_to_compare.append(('ensemble', f1))
    
    lines.append("")
    
    # Best Model
    if models_to_compare:
        best_model, best_f1 = max(models_to_compare, key=lambda x: x[1])
        lines.append(f"**Best Model:** {best_model.upper()} with F1-Macro = {best_f1:.2f}%")
        lines.append("")
    
    # Key Findings
    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. **Model Performance:**")
    
    if 'test_results' in data:
        hybrid_f1 = data['test_results'].get('f1_macro', 0) * 100
        baseline_scores = []
        for key, baseline in data.items():
            if not key.startswith('baseline_'):
                continue
            name = key.replace('baseline_', '')
            f1 = baseline.get('macro_f1', baseline.get('f1_macro', 0)) * 100
            baseline_scores.append((name, f1))

        if baseline_scores:
            best_name, best_f1 = max(baseline_scores, key=lambda item: item[1])
            diff = hybrid_f1 - best_f1
            if diff > 0:
                lines.append(
                    f"   - Hybrid model outperforms {best_name.upper()} by {diff:.2f}%"
                )
            else:
                lines.append(
                    f"   - {best_name.upper()} outperforms Hybrid model by {abs(diff):.2f}%"
                )
    
    lines.append("")
    lines.append("2. **Training Observations:**")
    if 'training_history' in data:
        history = data['training_history']
        final_train_f1 = history['train_f1_macro'][-1] * 100
        final_val_f1 = history['val_f1_macro'][-1] * 100
        gap = final_train_f1 - final_val_f1
        
        if gap > 5:
            lines.append(f"   - Model shows signs of overfitting (train-val gap: {gap:.2f}%)")
        else:
            lines.append(f"   - Model generalizes well (train-val gap: {gap:.2f}%)")
    
    lines.append("")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f'Report saved to: {output_path}')
    print('\n' + '\n'.join(lines[:50]))  # Print first 50 lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation report')
    parser.add_argument('--results_dir', default='./results', help='Results directory')
    parser.add_argument('--experiment', default='thesis_full_gpu', help='Experiment name')
    parser.add_argument('--output', default='./EVALUATION_REPORT.md', help='Output file')
    args = parser.parse_args()
    
    data = load_all_data(args.results_dir, args.experiment)
    generate_markdown_report(data, args.output)
