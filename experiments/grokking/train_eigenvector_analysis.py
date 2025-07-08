import argparse
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import create_data_loader, generate_modular_addition_data
from model import ModularAdditionTransformer
from simple_model import SimpleModularAdditionMLP


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_accuracy(model, data_loader, device):
    """Compute accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def compute_hessian_vector_product(loss, params, vector):
    """Compute Hessian-vector product H @ v."""
    # First compute gradients
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    
    # Compute grad @ vector
    grad_dot_v = sum((g * v).sum() for g, v in zip(grads, vector))
    
    # Compute Hessian-vector product
    Hv = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
    
    return Hv


def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors (list of tensors)."""
    # Flatten and concatenate
    v1_flat = torch.cat([t.flatten() for t in v1])
    v2_flat = torch.cat([t.flatten() for t in v2])
    
    # Compute cosine similarity
    dot_product = (v1_flat * v2_flat).sum()
    norm_v1 = torch.norm(v1_flat)
    norm_v2 = torch.norm(v2_flat)
    
    if norm_v1 > 0 and norm_v2 > 0:
        cos_sim = dot_product / (norm_v1 * norm_v2)
        return cos_sim.item()
    else:
        return 0.0


def analyze_gradient_eigenvector(model, loss, device):
    """Analyze how close gradient is to being an eigenvector of Hessian."""
    params = list(model.parameters())
    
    # Get gradients
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    
    # Compute Hessian-vector product H @ grad
    try:
        Hg = compute_hessian_vector_product(loss, params, grads)
        
        # Compute cosine similarity between grad and H @ grad
        cos_sim = compute_cosine_similarity(grads, Hg)
        
        # Compute eigenvalue estimate (Rayleigh quotient)
        grad_flat = torch.cat([g.flatten() for g in grads])
        Hg_flat = torch.cat([h.flatten() for h in Hg])
        eigenvalue_estimate = (grad_flat * Hg_flat).sum() / (grad_flat * grad_flat).sum()
        
        return cos_sim, eigenvalue_estimate.item()
    except Exception as e:
        print(f"Warning: Eigenvector analysis failed: {str(e)[:50]}")
        return None, None


def train_with_eigenvector_analysis(args):
    """Main training function with gradient eigenvector analysis."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Generate data
    print(f"Generating modular addition data with p={args.p}")
    train_data, test_data = generate_modular_addition_data(
        p=args.p,
        train_frac=args.train_frac,
        seed=args.seed
    )
    
    # Create data loaders
    train_loader = create_data_loader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = create_data_loader(test_data, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train size: {len(train_data[0])}, Test size: {len(test_data[0])}")
    
    # Initialize model - use simple MLP to avoid attention issues
    if args.use_simple_model:
        model = SimpleModularAdditionMLP(p=args.p, hidden_size=args.d_model).to(device)
    else:
        model = ModularAdditionTransformer(
            p=args.p,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )
    
    # Initialize logging
    eigenvector_history = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"grokking_eigen_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {args.num_steps} steps")
    print(f"Computing cos(grad, H@grad) every {args.eigen_log_interval} steps")
    
    step = 0
    
    with tqdm(total=args.num_steps) as pbar:
        while step < args.num_steps:
            for inputs, labels in train_loader:
                if step >= args.num_steps:
                    break
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                model.train()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Eigenvector analysis
                if step % args.eigen_log_interval == 0:
                    # Recompute for clean graph
                    outputs_analysis = model(inputs)
                    loss_analysis = criterion(outputs_analysis, labels)
                    
                    cos_sim, eigenvalue = analyze_gradient_eigenvector(model, loss_analysis, device)
                    
                    if cos_sim is not None:
                        eigenvector_history.append({
                            'step': step,
                            'cosine_similarity': cos_sim,
                            'eigenvalue_estimate': eigenvalue,
                            'loss': loss.item()
                        })
                        
                        if step == 0 or step % 1000 == 0:
                            print(f"\nStep {step}: cos(grad, H@grad) = {cos_sim:.4f}, eigenvalue ≈ {eigenvalue:.4f}")
                
                # Optimizer step
                optimizer.step()
                
                # Log metrics periodically
                if step % args.log_interval == 0:
                    # Compute accuracies
                    train_acc = compute_accuracy(model, train_loader, device)
                    test_acc = compute_accuracy(model, test_loader, device)
                    
                    # Compute test loss
                    model.eval()
                    test_loss = 0
                    with torch.no_grad():
                        for test_inputs, test_labels in test_loader:
                            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                            test_outputs = model(test_inputs)
                            test_loss += criterion(test_outputs, test_labels).item()
                    test_loss /= len(test_loader)
                    
                    # Store metrics
                    train_losses.append(loss.item())
                    test_losses.append(test_loss)
                    train_accs.append(train_acc)
                    test_accs.append(test_acc)
                    
                    pbar.set_postfix({
                        'train_acc': f'{train_acc:.3f}',
                        'test_acc': f'{test_acc:.3f}',
                        'cos_sim': f'{eigenvector_history[-1]["cosine_similarity"]:.3f}' if eigenvector_history else 'N/A'
                    })
                
                step += 1
                pbar.update(1)
    
    # Save results
    print("\nSaving results...")
    
    # Save eigenvector analysis
    eigen_path = os.path.join(output_dir, 'eigenvector_analysis.pkl')
    with open(eigen_path, 'wb') as f:
        pickle.dump(eigenvector_history, f)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'training_metrics.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump({
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'steps': list(range(0, len(train_losses) * args.log_interval, args.log_interval))
        }, f)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot accuracies
    steps = np.arange(len(train_accs)) * args.log_interval
    ax = axes[0, 0]
    ax.plot(steps, train_accs, label='Train', alpha=0.8)
    ax.plot(steps, test_accs, label='Test', alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy during Grokking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot cosine similarity
    eigen_steps = [e['step'] for e in eigenvector_history]
    cos_sims = [e['cosine_similarity'] for e in eigenvector_history]
    
    ax = axes[0, 1]
    ax.plot(eigen_steps, cos_sims, color='red', alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('cos(grad, H@grad)')
    ax.set_title('Gradient-Hessian Eigenvector Alignment')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.3, label='Perfect alignment')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Plot eigenvalue estimates
    eigenvalues = [e['eigenvalue_estimate'] for e in eigenvector_history]
    
    ax = axes[1, 0]
    ax.plot(eigen_steps, eigenvalues, color='green', alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Eigenvalue estimate')
    ax.set_title('Rayleigh Quotient (grad^T H grad / ||grad||²)')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('symlog')
    
    # Combined plot with phases
    ax = axes[1, 1]
    
    # Identify phases
    test_accs_array = np.array(test_accs)
    train_accs_array = np.array(train_accs)
    
    memorization_end = np.where(train_accs_array > 0.95)[0][0] if np.any(train_accs_array > 0.95) else 0
    grokking_start = np.where(test_accs_array > 0.1)[0][0] if np.any(test_accs_array > 0.1) else len(test_accs)
    grokking_end = np.where(test_accs_array > 0.9)[0][0] if np.any(test_accs_array > 0.9) else len(test_accs)
    
    # Plot cosine similarity with phase shading
    ax.plot(eigen_steps, cos_sims, color='red', alpha=0.8, label='cos(grad, H@grad)')
    
    # Shade phases
    if memorization_end > 0:
        ax.axvspan(0, steps[memorization_end], alpha=0.2, color='red', label='Memorization')
    if grokking_start < grokking_end:
        ax.axvspan(steps[grokking_start], steps[grokking_end], alpha=0.2, color='yellow', label='Grokking')
    if grokking_end < len(steps):
        ax.axvspan(steps[grokking_end], steps[-1], alpha=0.2, color='green', label='Generalization')
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Eigenvector Alignment Across Grokking Phases')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'eigenvector_analysis.png')
    plt.savefig(plot_path, dpi=150)
    
    # Print summary
    print(f"\n=== Eigenvector Analysis Summary ===")
    print(f"Total measurements: {len(eigenvector_history)}")
    
    # Compute statistics per phase
    if eigenvector_history:
        # Memorization phase
        mem_cos = [e['cosine_similarity'] for e in eigenvector_history if e['step'] <= steps[memorization_end]]
        if mem_cos:
            print(f"\nMemorization phase: mean cos(grad, H@grad) = {np.mean(mem_cos):.4f}")
        
        # Grokking phase
        grok_cos = [e['cosine_similarity'] for e in eigenvector_history 
                    if steps[grokking_start] <= e['step'] <= steps[grokking_end]]
        if grok_cos:
            print(f"Grokking phase: mean cos(grad, H@grad) = {np.mean(grok_cos):.4f}")
        
        # Generalization phase
        gen_cos = [e['cosine_similarity'] for e in eigenvector_history if e['step'] >= steps[grokking_end]]
        if gen_cos:
            print(f"Generalization phase: mean cos(grad, H@grad) = {np.mean(gen_cos):.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Grokking with gradient eigenvector analysis')
    
    # Data parameters
    parser.add_argument('--p', type=int, default=97, help='Prime modulus')
    parser.add_argument('--train_frac', type=float, default=0.3, help='Fraction of data for training')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--num_steps', type=int, default=50000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1.0, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.98, help='Adam beta2')
    
    # Analysis parameters
    parser.add_argument('--eigen_log_interval', type=int, default=100, help='Eigenvector analysis interval')
    parser.add_argument('--use_simple_model', action='store_true', default=True, help='Use simple MLP instead of transformer')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=100, help='Metrics logging interval')
    
    args = parser.parse_args()
    
    # Run training
    train_with_eigenvector_analysis(args)


if __name__ == '__main__':
    main()