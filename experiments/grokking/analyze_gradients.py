import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime
import torch


def load_data(log_dir):
    """Load gradient and Hessian data from log directory."""
    # Load parameter info
    with open(os.path.join(log_dir, 'param_info.pkl'), 'rb') as f:
        param_info = pickle.load(f)
    
    # Load final gradients
    grad_path = os.path.join(log_dir, 'gradients_final.pkl')
    with open(grad_path, 'rb') as f:
        gradient_history = pickle.load(f)
    
    # Load final Hessians if available
    hessian_history = None
    hess_path = os.path.join(log_dir, 'hessians_final.pkl')
    if os.path.exists(hess_path):
        with open(hess_path, 'rb') as f:
            hessian_history = pickle.load(f)
    
    # Load training metrics
    metrics_path = os.path.join(log_dir, 'training_metrics.pkl')
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    
    return param_info, gradient_history, hessian_history, metrics


def compute_gradient_stats(gradient_history, param_name):
    """Compute statistics for gradients of a specific parameter."""
    steps = []
    norms = []
    means = []
    stds = []
    maxs = []
    mins = []
    
    for entry in gradient_history:
        if param_name in entry['gradients']:
            grad = entry['gradients'][param_name]
            steps.append(entry['step'])
            norms.append(np.linalg.norm(grad.flatten()))
            means.append(np.mean(grad))
            stds.append(np.std(grad))
            maxs.append(np.max(grad))
            mins.append(np.min(grad))
    
    return {
        'steps': np.array(steps),
        'norms': np.array(norms),
        'means': np.array(means),
        'stds': np.array(stds),
        'maxs': np.array(maxs),
        'mins': np.array(mins)
    }


def compute_hessian_stats(hessian_history, param_name):
    """Compute statistics for Hessian diagonal of a specific parameter."""
    steps = []
    norms = []
    means = []
    stds = []
    maxs = []
    mins = []
    
    for entry in hessian_history:
        if param_name in entry['hessian_diagonal']:
            hess = entry['hessian_diagonal'][param_name]
            steps.append(entry['step'])
            norms.append(np.linalg.norm(hess.flatten()))
            means.append(np.mean(hess))
            stds.append(np.std(hess))
            maxs.append(np.max(hess))
            mins.append(np.min(hess))
    
    return {
        'steps': np.array(steps),
        'norms': np.array(norms),
        'means': np.array(means),
        'stds': np.array(stds),
        'maxs': np.array(maxs),
        'mins': np.array(mins)
    }


def plot_gradient_analysis(log_dir, save_plots=True):
    """Create comprehensive plots for gradient and Hessian analysis."""
    print(f"Loading data from {log_dir}...")
    param_info, gradient_history, hessian_history, metrics = load_data(log_dir)
    
    # Create output directory for plots
    if save_plots:
        plot_dir = os.path.join(log_dir, 'analysis_plots')
        os.makedirs(plot_dir, exist_ok=True)
    
    # Get parameter names
    param_names = list(param_info.keys())
    
    # 1. Plot gradient norms for all parameters
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Gradient Norm Evolution During Training', fontsize=16)
    
    for ax, param_name in zip(axes.flatten(), param_names[:4]):  # Plot first 4 parameters
        stats = compute_gradient_stats(gradient_history, param_name)
        ax.plot(stats['steps'], stats['norms'], label=param_name, alpha=0.8)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Gradient Norm')
        ax.set_title(f'{param_name}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'gradient_norms.png'), dpi=150)
    
    # 2. Plot combined gradient statistics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Aggregate gradient norms
    total_grad_norms = []
    steps = []
    for entry in gradient_history:
        total_norm = 0
        for param_name, grad in entry['gradients'].items():
            total_norm += np.linalg.norm(grad.flatten())**2
        total_grad_norms.append(np.sqrt(total_norm))
        steps.append(entry['step'])
    
    # Plot total gradient norm with grokking phases
    ax1.plot(steps, total_grad_norms, label='Total Gradient Norm', color='blue', alpha=0.8)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Total Gradient Norm')
    ax1.set_title('Total Gradient Norm vs Training Progress')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add test accuracy on secondary axis
    ax1_acc = ax1.twinx()
    ax1_acc.plot(metrics['steps'], metrics['test_accs'], label='Test Accuracy', color='red', alpha=0.6)
    ax1_acc.set_ylabel('Test Accuracy', color='red')
    ax1_acc.tick_params(axis='y', labelcolor='red')
    ax1_acc.set_ylim(-0.05, 1.05)
    
    # Plot gradient variance
    param_variances = []
    for entry in gradient_history:
        variances = []
        for param_name, grad in entry['gradients'].items():
            variances.append(np.var(grad.flatten()))
        param_variances.append(np.mean(variances))
    
    ax2.plot(steps, param_variances, label='Mean Gradient Variance', color='green', alpha=0.8)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Mean Gradient Variance')
    ax2.set_title('Gradient Variance vs Training Progress')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'gradient_statistics.png'), dpi=150)
    
    # 3. Plot Hessian diagonal statistics if available
    if hessian_history:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hessian Diagonal Evolution During Training', fontsize=16)
        
        for ax, param_name in zip(axes.flatten(), param_names[:4]):
            stats = compute_hessian_stats(hessian_history, param_name)
            ax.plot(stats['steps'], stats['norms'], label=f'{param_name} norm', alpha=0.8)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Hessian Diagonal Norm')
            ax.set_title(f'{param_name}')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(plot_dir, 'hessian_norms.png'), dpi=150)
    
    # 4. Create phase analysis plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    # Identify grokking phases based on test accuracy
    test_accs = np.array(metrics['test_accs'])
    train_accs = np.array(metrics['train_accs'])
    steps_array = np.array(metrics['steps'])
    
    # Phase 1: Memorization (train acc high, test acc low)
    # Phase 2: Grokking transition (test acc increasing rapidly)
    # Phase 3: Generalization (both accs high)
    
    memorization_end = np.where(train_accs > 0.95)[0][0] if np.any(train_accs > 0.95) else 0
    grokking_start = np.where(test_accs > 0.1)[0][0] if np.any(test_accs > 0.1) else len(test_accs)
    grokking_end = np.where(test_accs > 0.9)[0][0] if np.any(test_accs > 0.9) else len(test_accs)
    
    # Plot 1: Accuracies with phases
    ax = axes[0]
    ax.plot(steps_array, train_accs, label='Train Accuracy', alpha=0.8)
    ax.plot(steps_array, test_accs, label='Test Accuracy', alpha=0.8)
    
    # Shade phases
    if memorization_end > 0:
        ax.axvspan(0, steps_array[memorization_end], alpha=0.2, color='red', label='Memorization')
    if grokking_start < grokking_end:
        ax.axvspan(steps_array[grokking_start], steps_array[grokking_end], alpha=0.2, color='yellow', label='Grokking')
    if grokking_end < len(steps_array):
        ax.axvspan(steps_array[grokking_end], steps_array[-1], alpha=0.2, color='green', label='Generalization')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Phases')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient norms with phases
    ax = axes[1]
    ax.plot(steps, total_grad_norms, color='blue', alpha=0.8)
    
    # Shade phases
    if memorization_end > 0:
        ax.axvspan(0, steps_array[memorization_end], alpha=0.2, color='red')
    if grokking_start < grokking_end:
        ax.axvspan(steps_array[grokking_start], steps_array[grokking_end], alpha=0.2, color='yellow')
    if grokking_end < len(steps_array):
        ax.axvspan(steps_array[grokking_end], steps_array[-1], alpha=0.2, color='green')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Total Gradient Norm')
    ax.set_title('Gradient Behavior Across Phases')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss curves with phases
    ax = axes[2]
    ax.plot(steps_array, metrics['train_losses'], label='Train Loss', alpha=0.8)
    ax.plot(steps_array, metrics['test_losses'], label='Test Loss', alpha=0.8)
    
    # Shade phases
    if memorization_end > 0:
        ax.axvspan(0, steps_array[memorization_end], alpha=0.2, color='red')
    if grokking_start < grokking_end:
        ax.axvspan(steps_array[grokking_start], steps_array[grokking_end], alpha=0.2, color='yellow')
    if grokking_end < len(steps_array):
        ax.axvspan(steps_array[grokking_end], steps_array[-1], alpha=0.2, color='green')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Evolution Across Phases')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(plot_dir, 'phase_analysis.png'), dpi=150)
    
    # Print summary statistics
    print("\n=== Gradient Analysis Summary ===")
    print(f"Total training steps: {len(gradient_history)}")
    print(f"Parameters tracked: {len(param_names)}")
    print(f"\nGrokking phases identified:")
    print(f"  Memorization: steps 0 - {steps_array[memorization_end] if memorization_end > 0 else 'N/A'}")
    print(f"  Grokking: steps {steps_array[grokking_start] if grokking_start < len(steps_array) else 'N/A'} - {steps_array[grokking_end] if grokking_end < len(steps_array) else 'N/A'}")
    print(f"  Generalization: steps {steps_array[grokking_end] if grokking_end < len(steps_array) else 'N/A'} - {steps_array[-1]}")
    
    print(f"\nFinal gradient norm: {total_grad_norms[-1]:.6f}")
    print(f"Peak gradient norm: {max(total_grad_norms):.6f} at step {steps[np.argmax(total_grad_norms)]}")
    print(f"Minimum gradient norm: {min(total_grad_norms):.6f} at step {steps[np.argmin(total_grad_norms)]}")
    
    if hessian_history:
        print(f"\nHessian diagonal data available: {len(hessian_history)} snapshots")
    
    if save_plots:
        print(f"\nPlots saved to: {plot_dir}")
    
    plt.show()
    
    return param_info, gradient_history, hessian_history, metrics


def main():
    parser = argparse.ArgumentParser(description='Analyze gradients and Hessians from grokking experiment')
    parser.add_argument('log_dir', type=str, help='Directory containing the gradient/Hessian logs')
    parser.add_argument('--no_save', action='store_true', help='Do not save plots')
    
    args = parser.parse_args()
    
    plot_gradient_analysis(args.log_dir, save_plots=not args.no_save)


if __name__ == '__main__':
    main()