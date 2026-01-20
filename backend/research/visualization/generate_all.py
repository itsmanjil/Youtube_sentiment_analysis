#!/usr/bin/env python
"""
Master script to generate all visualizations and reports

Runs all visualization scripts in sequence:
1. Training curves
2. Confusion matrices  
3. Model comparisons
4. Comprehensive report
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_script(script_name, args_list):
    """Run a visualization script."""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f'ERROR: Script not found: {script_path}')
        return False
    
    cmd = [sys.executable, str(script_path)] + args_list
    print(f"\nRunning: {' '.join(cmd)}")
    print('-' * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print('SUCCESS')
        return True
    except subprocess.CalledProcessError as e:
        print(f'ERROR: {e}')
        return False


def generate_all_visualizations(experiment_name='thesis_full_gpu', output_dir='./plots'):
    """Generate all visualizations."""
    print('='*60)
    print('THESIS VISUALIZATION GENERATOR')
    print('='*60)
    
    base_dir = Path(__file__).resolve().parents[2]
    results_dir = base_dir / 'results'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_count = 0
    
    # 1. Training curves
    print("\n[1/4] Generating training curves...")
    history_file = base_dir / 'output' / experiment_name / 'training_history.json'
    if history_file.exists():
        total_count += 1
        if run_script('plot_training_curves.py', [
            '--history', str(history_file),
            '--output', output_dir
        ]):
            success_count += 1
    else:
        print(f'WARNING: Training history not found: {history_file}')
    
    # 2. Confusion matrices
    print("\n[2/4] Generating confusion matrices...")
    test_file = base_dir / 'data' / 'test.csv'
    if test_file.exists():
        total_count += 1
        if run_script('plot_confusion_matrix.py', [
            '--test_csv', str(test_file),
            '--models', 'logreg,svm,hybrid_dl',
            '--output', output_dir
        ]):
            success_count += 1
    else:
        print(f'WARNING: Test data not found: {test_file}')
    
    # 3. Model comparisons
    print("\n[3/4] Generating model comparisons...")
    total_count += 1
    if run_script('plot_model_comparison.py', [
        '--results_dir', str(results_dir),
        '--output', output_dir
    ]):
        success_count += 1
    
    # 4. Evaluation report
    print("\n[4/4] Generating evaluation report...")
    total_count += 1
    report_output = base_dir / 'EVALUATION_REPORT.md'
    if run_script('generate_report.py', [
        '--results_dir', str(results_dir),
        '--experiment', experiment_name,
        '--output', str(report_output)
    ]):
        success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f'SUMMARY: {success_count}/{total_count} tasks completed successfully')
    print('='*60)
    print(f"Outputs saved to: {output_path.resolve()}")
    
    if success_count == total_count:
        print("\nAll visualizations generated successfully!")
        return 0
    else:
        print(f"\nWARNING: {total_count - success_count} task(s) failed")
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate all visualizations')
    parser.add_argument('--experiment', default='thesis_full_gpu', 
                       help='Experiment name (default: thesis_full_gpu)')
    parser.add_argument('--output', default='./plots',
                       help='Output directory (default: ./plots)')
    args = parser.parse_args()
    
    sys.exit(generate_all_visualizations(args.experiment, args.output))
