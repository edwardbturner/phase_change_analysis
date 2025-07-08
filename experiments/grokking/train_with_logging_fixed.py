import argparse
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from data import create_data_loader, generate_modular_addition_data
from model import ModularAdditionTransformer


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


def compute_hessian_diagonal_hutchinson(loss, model, num_samples=5):
    """Hutchinson trace estimator for Hessian diagonal - works with transformers."""
    hess_diag = {}
    
    # First compute gradients
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    
    for (name, param), grad in zip(model.named_parameters(), grads):
        if param.requires_grad and grad is not None:
            diag_estimate = torch.zeros_like(param)
            
            # Skip very large parameters to save memory/time
            if param.numel() > 100000:
                # For large parameters, sample a subset
                n_samples = min(num_samples, 3)
            else:
                n_samples = num_samples
            
            for _ in range(n_samples):
                # Random vector with Rademacher distribution
                z = torch.randint_like(param, high=2, dtype=param.dtype, device=param.device).float()
                z[z == 0] = -1
                
                try:
                    # Compute Hessian-vector product
                    hz = torch.autograd.grad(grad, param, grad_outputs=z, retain_graph=True)[0]
                    
                    # Update diagonal estimate
                    diag_estimate += z * hz
                except RuntimeError as e:
                    # Skip if gradient computation fails
                    print(f"Warning: Skipping Hessian for {name}: {str(e)[:50]}...")
                    diag_estimate = torch.zeros_like(param)
                    break
            
            hess_diag[name] = (diag_estimate / n_samples).detach()
    
    return hess_diag


def get_param_groups_info(model):
    """Get information about parameter groups for logging."""
    param_info = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_info[name] = {
                'shape': list(param.shape),
                'numel': param.numel()
            }
    return param_info


def train_grokking_with_logging(args):
    """Main training function with gradient and Hessian logging."""
    # Set device and optimize for H100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        # Enable TF32 for H100 optimization
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory fraction to allow for gradient/hessian storage
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Set random seed
    set_seed(args.seed)
    
    # Generate data
    print(f"Generating modular addition data with p={args.p}")
    train_data, test_data = generate_modular_addition_data(
        p=args.p,
        train_frac=args.train_frac,
        seed=args.seed
    )
    
    # Create data loaders with larger batch size for H100
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    train_loader = create_data_loader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = create_data_loader(test_data, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train size: {len(train_data[0])}, Test size: {len(test_data[0])}")
    print(f"Effective batch size: {effective_batch_size}")
    
    # Initialize model
    model = ModularAdditionTransformer(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    
    # Get parameter info
    param_info = get_param_groups_info(model)
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
    
    # Mixed precision training for H100
    scaler = GradScaler() if args.use_amp else None
    
    # Initialize logging structures
    gradient_history = []
    hessian_history = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"grokking_logs_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save parameter info
    with open(os.path.join(output_dir, 'param_info.pkl'), 'wb') as f:
        pickle.dump(param_info, f)
    
    # Training loop
    print(f"Starting training for {args.num_steps} steps")
    print(f"Logging gradients and Hessians{'with mixed precision' if args.use_amp else ''}")
    
    step = 0
    epoch = 0
    accumulation_counter = 0
    
    with tqdm(total=args.num_steps) as pbar:
        while step < args.num_steps:
            epoch += 1
            for inputs, labels in train_loader:
                if step >= args.num_steps:
                    break
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass with mixed precision
                if args.use_amp and scaler is not None:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / args.gradient_accumulation_steps
                    
                    # Backward pass
                    scaler.scale(loss).backward()
                else:
                    model.train()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss = loss / args.gradient_accumulation_steps
                    loss.backward()
                
                accumulation_counter += 1
                
                # Perform optimizer step after accumulating gradients
                if accumulation_counter % args.gradient_accumulation_steps == 0:
                    # Log gradients before optimizer step
                    current_gradients = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if args.use_amp and scaler is not None:
                                # Unscale gradients for logging
                                inv_scale = 1.0 / scaler.get_scale()
                                current_gradients[name] = (param.grad * inv_scale).detach().cpu().numpy().copy()
                            else:
                                current_gradients[name] = param.grad.detach().cpu().numpy().copy()
                    
                    gradient_history.append({
                        'step': step,
                        'gradients': current_gradients
                    })
                    
                    # Compute and log Hessian diagonal if requested
                    if args.log_hessian and step % args.hessian_log_interval == 0:
                        # Disable mixed precision for Hessian computation
                        model.train()
                        with torch.cuda.amp.autocast(enabled=False):
                            outputs_hess = model(inputs)
                            loss_hess = criterion(outputs_hess, labels)
                        
                        try:
                            hessian_diag = compute_hessian_diagonal_hutchinson(
                                loss_hess, model, num_samples=args.hessian_samples
                            )
                            
                            # Convert to numpy for storage
                            current_hessian = {}
                            for name, h_diag in hessian_diag.items():
                                current_hessian[name] = h_diag.cpu().numpy().copy()
                            
                            hessian_history.append({
                                'step': step,
                                'hessian_diagonal': current_hessian
                            })
                            
                            if step == 0:
                                print(f"\nHessian computation successful! Tracking {len(current_hessian)} parameters")
                        except Exception as e:
                            if step == 0:
                                print(f"\nWarning: Hessian computation failed: {str(e)[:100]}")
                    
                    # Optimizer step
                    if args.use_amp and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    
                    # Save checkpoint periodically
                    if step % args.save_interval == 0 and step > 0:
                        # Save gradients
                        grad_path = os.path.join(output_dir, f'gradients_step_{step}.pkl')
                        with open(grad_path, 'wb') as f:
                            pickle.dump(gradient_history, f)
                        
                        # Save Hessians
                        if args.log_hessian and hessian_history:
                            hess_path = os.path.join(output_dir, f'hessians_step_{step}.pkl')
                            with open(hess_path, 'wb') as f:
                                pickle.dump(hessian_history, f)
                        
                        # Clear old data to save memory
                        if args.clear_old_logs:
                            gradient_history = gradient_history[-args.save_interval:]
                            if args.log_hessian:
                                hessian_history = hessian_history[-(args.save_interval // args.hessian_log_interval):]
                
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
                    train_losses.append(loss.item() * args.gradient_accumulation_steps)
                    test_losses.append(test_loss)
                    train_accs.append(train_acc)
                    test_accs.append(test_acc)
                    
                    pbar.set_postfix({
                        'train_loss': f'{loss.item() * args.gradient_accumulation_steps:.4f}',
                        'test_loss': f'{test_loss:.4f}',
                        'train_acc': f'{train_acc:.3f}',
                        'test_acc': f'{test_acc:.3f}'
                    })
                
                step += 1
                pbar.update(1)
    
    # Save final data
    print("\nSaving final gradient and Hessian logs...")
    
    # Save all gradients
    final_grad_path = os.path.join(output_dir, 'gradients_final.pkl')
    with open(final_grad_path, 'wb') as f:
        pickle.dump(gradient_history, f)
    
    # Save all Hessians
    if args.log_hessian:
        final_hess_path = os.path.join(output_dir, 'hessians_final.pkl')
        with open(final_hess_path, 'wb') as f:
            pickle.dump(hessian_history, f)
    
    # Save training metrics
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
    steps = np.arange(len(train_losses)) * args.log_interval
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot losses
    ax1.plot(steps, train_losses, label='Train Loss', alpha=0.8)
    ax1.plot(steps, test_losses, label='Test Loss', alpha=0.8)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Grokking: Loss Curves')
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(steps, train_accs, label='Train Accuracy', alpha=0.8)
    ax2.plot(steps, test_accs, label='Test Accuracy', alpha=0.8)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Grokking: Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'grokking_results.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    
    # Save final checkpoint
    checkpoint_path = os.path.join(output_dir, 'final_checkpoint.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'args': args
    }, checkpoint_path)
    print(f"Final checkpoint saved to {checkpoint_path}")
    
    # Print final metrics
    print(f"\nFinal metrics:")
    print(f"Train accuracy: {train_accs[-1]:.3f}")
    print(f"Test accuracy: {test_accs[-1]:.3f}")
    print(f"Train loss: {train_losses[-1]:.4f}")
    print(f"Test loss: {test_losses[-1]:.4f}")
    
    if args.log_hessian and hessian_history:
        print(f"\nHessian diagonal logged: {len(hessian_history)} snapshots")
    
    print(f"\nAll logs saved to: {output_dir}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Reproduce grokking with gradient/Hessian logging')
    
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
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size (increased for H100)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1.0, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.98, help='Adam beta2')
    
    # H100 optimization parameters
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use mixed precision training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    
    # Logging parameters
    parser.add_argument('--log_hessian', action='store_true', default=True, help='Log Hessian diagonal')
    parser.add_argument('--hessian_log_interval', type=int, default=500, help='Hessian logging interval')
    parser.add_argument('--hessian_samples', type=int, default=3, help='Number of samples for Hutchinson estimator')
    parser.add_argument('--save_interval', type=int, default=5000, help='Save checkpoint interval')
    parser.add_argument('--clear_old_logs', action='store_true', default=True, help='Clear old logs to save memory')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    
    args = parser.parse_args()
    
    # Run training
    train_grokking_with_logging(args)


if __name__ == '__main__':
    main()