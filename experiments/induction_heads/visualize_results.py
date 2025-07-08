import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import argparse
import os
from matplotlib.patches import Rectangle
import seaborn as sns


def load_results(results_path):
    """Load training results from pickle file."""
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results


def plot_training_curves(history, output_dir):
    """Plot training and test metrics over time."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    steps = history['steps']
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(steps, history['train_loss'], label='Train Loss', alpha=0.8)
    ax.plot(steps, history['test_loss'], label='Test Loss', alpha=0.8)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Accuracy curves
    ax = axes[0, 1]
    ax.plot(steps, history['train_accuracy'], label='Train Accuracy', alpha=0.8)
    ax.plot(steps, history['test_accuracy'], label='Test Accuracy', alpha=0.8)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Overall Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Pattern vs Random accuracy
    ax = axes[1, 0]
    ax.plot(steps, history['pattern_accuracy'], label='Pattern Completion', color='green', linewidth=2)
    ax.plot(steps, history['random_accuracy'], label='Random Tokens', color='orange', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Induction vs Random Token Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Learning rate
    ax = axes[1, 1]
    ax.plot(steps, history['learning_rates'], color='purple', alpha=0.8)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_eigenvalue_evolution(eigen_history, history, output_dir):
    """Plot eigenvalue evolution and phase transitions."""
    if not eigen_history:
        print("No eigenvalue history found")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Extract data
    steps = [h['step'] for h in eigen_history]
    pattern_acc = [h['accuracy'] for h in eigen_history]
    
    # Plot 1: Top eigenvalues over time
    ax = axes[0]
    n_eigenvalues = min(5, len(eigen_history[0]['top_eigenvalues']))
    
    eigenvalue_traces = [[] for _ in range(n_eigenvalues)]
    for h in eigen_history:
        for i in range(min(n_eigenvalues, len(h['top_eigenvalues']))):
            eigenvalue_traces[i].append(h['top_eigenvalues'][i])
    
    for i, trace in enumerate(eigenvalue_traces):
        ax.plot(steps, trace, label=f'λ_{i+1}', alpha=0.8)
    
    # Add pattern accuracy on secondary axis
    ax2 = ax.twinx()
    ax2.plot(steps, pattern_acc, 'k--', alpha=0.5, label='Pattern Accuracy')
    ax2.set_ylabel('Pattern Accuracy', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Eigenvalue Magnitude')
    ax.set_title('Top Hessian Eigenvalues During Training')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Spectral norm and gradient norm
    ax = axes[1]
    spectral_norms = [h['spectral_norm'] for h in eigen_history]
    grad_norms = [h['grad_norm'] for h in eigen_history]
    
    ax.plot(steps, spectral_norms, label='Spectral Norm (Top Eigenvalue)', color='red', alpha=0.8)
    ax.plot(steps, grad_norms, label='Gradient Norm', color='blue', alpha=0.8)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Magnitude')
    ax.set_title('Spectral Norm vs Gradient Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Gradient alignment with top eigenvector
    ax = axes[2]
    alignments = [abs(h['top_alignment']) for h in eigen_history]
    
    ax.plot(steps, alignments, color='purple', alpha=0.8)
    ax.fill_between(steps, 0, alignments, alpha=0.3, color='purple')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('|cos(∇L, v₁)|')
    ax.set_title('Gradient Alignment with Top Eigenvector')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eigenvalue_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_analysis(eigen_history, output_dir):
    """Plot attention head metrics to visualize induction head formation."""
    if not eigen_history or 'attention_info' not in eigen_history[0]:
        print("No attention analysis found")
        return
    
    # Extract attention metrics
    steps = []
    qk_eigenvalues = {f'layer_{i}': [] for i in range(2)}  # Assuming 2 layers
    ov_copy_ratios = {f'layer_{i}': [] for i in range(2)}
    
    for h in eigen_history:
        if 'attention_info' in h and h['attention_info']:
            steps.append(h['step'])
            for layer_key in h['attention_info']:
                if layer_key in qk_eigenvalues:
                    qk_eigenvalues[layer_key].append(
                        h['attention_info'][layer_key]['qk_top_eigenvalue']
                    )
                    ov_copy_ratios[layer_key].append(
                        h['attention_info'][layer_key]['ov_copy_ratio']
                    )
    
    if not steps:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot QK eigenvalues
    ax = axes[0]
    for layer_key in sorted(qk_eigenvalues.keys()):
        if qk_eigenvalues[layer_key]:
            ax.plot(steps, qk_eigenvalues[layer_key], 
                   label=f'{layer_key} QK', alpha=0.8)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Top QK Eigenvalue')
    ax.set_title('QK Circuit Formation (Pattern Matching)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot OV copy ratios
    ax = axes[1]
    for layer_key in sorted(ov_copy_ratios.keys()):
        if ov_copy_ratios[layer_key]:
            ax.plot(steps, ov_copy_ratios[layer_key], 
                   label=f'{layer_key} OV', alpha=0.8)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('OV Copy Ratio')
    ax.set_title('OV Circuit Formation (Copying Behavior)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attention_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_phase_transition_summary(results, output_dir):
    """Create a summary plot highlighting the phase transition."""
    history = results['history']
    eigen_history = results['eigen_history']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    steps = history['steps']
    pattern_acc = history['pattern_accuracy']
    
    # Plot pattern accuracy
    ax.plot(steps, pattern_acc, 'b-', linewidth=2, label='Pattern Completion Accuracy')
    
    # Detect phase transition
    if len(pattern_acc) > 20:
        # Simple detection: find steepest increase
        diffs = np.diff(pattern_acc)
        window = 10
        smoothed_diffs = np.convolve(diffs, np.ones(window)/window, mode='valid')
        
        if len(smoothed_diffs) > 0:
            max_idx = np.argmax(smoothed_diffs) + window//2
            transition_step = steps[max_idx]
            
            # Highlight phase transition
            ax.axvline(transition_step, color='red', linestyle='--', 
                      linewidth=2, label='Phase Transition')
            
            # Add shaded regions
            ax.axvspan(0, transition_step, alpha=0.1, color='gray', 
                      label='Pre-transition')
            ax.axvspan(transition_step, steps[-1], alpha=0.1, color='green', 
                      label='Post-transition')
    
    # Add annotations
    ax.set_xlabel('Training Steps', fontsize=14)
    ax.set_ylabel('Pattern Completion Accuracy', fontsize=14)
    ax.set_title('Induction Head Formation Phase Transition', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add text box with key insights
    if eigen_history:
        final_pattern_acc = pattern_acc[-1]
        initial_pattern_acc = pattern_acc[0]
        improvement = final_pattern_acc - initial_pattern_acc
        
        textstr = f'Initial Accuracy: {initial_pattern_acc:.1%}\n'
        textstr += f'Final Accuracy: {final_pattern_acc:.1%}\n'
        textstr += f'Improvement: {improvement:.1%}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.7, 0.2, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_transition_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def visualize_attention_heatmap(model_path, data_path, output_dir, layer_idx=0, seq_idx=0):
    """Visualize attention patterns as heatmaps."""
    # This would require loading the model and data
    # Placeholder for now - would be implemented if model is available
    pass


def main():
    parser = argparse.ArgumentParser(description='Visualize induction head training results')
    parser.add_argument('results_dir', type=str, help='Directory containing results.pkl')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory for plots (default: same as results_dir)')
    
    args = parser.parse_args()
    
    # Load results
    results_path = os.path.join(args.results_dir, 'results.pkl')
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    results = load_results(results_path)
    
    # Set output directory
    output_dir = args.output_dir or args.results_dir
    
    print("Generating visualizations...")
    
    # Generate all plots
    print("- Training curves...")
    plot_training_curves(results['history'], output_dir)
    
    print("- Eigenvalue evolution...")
    plot_eigenvalue_evolution(results['eigen_history'], results['history'], output_dir)
    
    print("- Attention analysis...")
    plot_attention_analysis(results['eigen_history'], output_dir)
    
    print("- Phase transition summary...")
    plot_phase_transition_summary(results, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == '__main__':
    main()