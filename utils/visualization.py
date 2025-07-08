import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


def plot_training_metrics(
    metrics: Dict[str, List[float]], 
    save_path: Optional[str] = None,
    title: str = "Training Metrics"
) -> None:
    """Plot training metrics like loss and accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    if 'train_loss' in metrics:
        axes[0].plot(metrics['train_loss'], label='Train Loss')
    if 'test_loss' in metrics:
        axes[0].plot(metrics['test_loss'], label='Test Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss over Training')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    # Plot accuracy
    if 'train_acc' in metrics:
        axes[1].plot(metrics['train_acc'], label='Train Accuracy')
    if 'test_acc' in metrics:
        axes[1].plot(metrics['test_acc'], label='Test Accuracy')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy over Training')
    axes[1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_eigenvalue_evolution(
    eigenvalues: np.ndarray,
    steps: List[int],
    save_path: Optional[str] = None,
    top_k: int = 10
) -> None:
    """Plot evolution of top eigenvalues over training."""
    plt.figure(figsize=(10, 6))
    
    for i in range(min(top_k, eigenvalues.shape[1])):
        plt.plot(steps, eigenvalues[:, i], label=f'λ_{i+1}')
    
    plt.xlabel('Training Step')
    plt.ylabel('Eigenvalue')
    plt.title(f'Evolution of Top {top_k} Eigenvalues')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_phase_transition(
    metrics: Dict[str, np.ndarray],
    transition_point: Optional[int] = None,
    save_path: Optional[str] = None
) -> None:
    """Visualize phase transition in training dynamics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    steps = metrics.get('steps', range(len(next(iter(metrics.values())))))
    
    # Plot various metrics
    metric_pairs = [
        ('train_loss', 'test_loss', 'Loss', axes[0, 0]),
        ('train_acc', 'test_acc', 'Accuracy', axes[0, 1]),
        ('gradient_norm', None, 'Gradient Norm', axes[1, 0]),
        ('weight_norm', None, 'Weight Norm', axes[1, 1])
    ]
    
    for train_key, test_key, title, ax in metric_pairs:
        if train_key in metrics:
            ax.plot(steps, metrics[train_key], label=f'Train {title}')
        if test_key and test_key in metrics:
            ax.plot(steps, metrics[test_key], label=f'Test {title}')
        
        if transition_point:
            ax.axvline(transition_point, color='red', linestyle='--', 
                      label='Phase Transition')
        
        ax.set_xlabel('Step')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Phase Transition Analysis')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()