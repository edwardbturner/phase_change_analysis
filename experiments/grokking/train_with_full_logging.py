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


def compute_hessian_diagonal_simple(loss, model):
    """Simple but robust Hessian diagonal computation."""
    hess_diag = {}
    
    # Get gradients
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    
    for (name, param), grad in zip(model.named_parameters(), grads):
        if param.requires_grad and grad is not None:
            # For small parameters, compute exact diagonal
            if param.numel() < 1000:
                diag = []
                grad_flat = grad.view(-1)
                for i in range(grad_flat.shape[0]):
                    try:
                        grad2 = torch.autograd.grad(grad_flat[i], param, retain_graph=True)[0]
                        diag.append(grad2.view(-1)[i].item())
                    except:
                        diag.append(0.0)
                hess_diag[name] = np.array(diag).reshape(param.shape)
            else:
                # For large parameters, use stochastic estimation
                z = torch.randn_like(param)
                z_norm = z / torch.sqrt((z**2).sum())
                try:
                    hz = torch.autograd.grad(grad, param, grad_outputs=z_norm, retain_graph=True)[0]
                    hess_diag[name] = (hz * z_norm).detach().cpu().numpy()
                except:
                    hess_diag[name] = np.zeros_like(param.detach().cpu().numpy())
    
    return hess_diag


def train_grokking_with_logging(args):
    """Main training function with gradient and Hessian logging."""
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
    
    # Initialize model
    model = ModularAdditionTransformer(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    
    # Get parameter info
    param_info = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_info[name] = {
                'shape': list(param.shape),
                'numel': param.numel()
            }
    
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
    print(f"Logging gradients every step, Hessians every {args.hessian_log_interval} steps")
    
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
                
                # Log gradients BEFORE optimizer step
                current_gradients = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        current_gradients[name] = param.grad.detach().cpu().numpy().copy()
                
                gradient_history.append({
                    'step': step,
                    'gradients': current_gradients
                })
                
                # Compute and log Hessian diagonal if requested
                if args.log_hessian and step % args.hessian_log_interval == 0:
                    # Recompute forward pass for clean computation graph
                    outputs_hess = model(inputs)
                    loss_hess = criterion(outputs_hess, labels)
                    
                    try:
                        hessian_diag = compute_hessian_diagonal_simple(loss_hess, model)
                        
                        hessian_history.append({
                            'step': step,
                            'hessian_diagonal': hessian_diag
                        })
                        
                        if step == 0:
                            print(f"\nHessian computation successful! Tracking {len(hessian_diag)} parameters")
                            for name in list(hessian_diag.keys())[:3]:
                                print(f"  {name}: shape={hessian_diag[name].shape}")
                    except Exception as e:
                        print(f"\nWarning: Hessian computation failed at step {step}: {str(e)[:100]}")
                
                # Optimizer step
                optimizer.step()
                
                # Save checkpoint periodically
                if step > 0 and step % args.save_interval == 0:
                    # Save gradients
                    print(f"\nSaving checkpoint at step {step}...")
                    grad_path = os.path.join(output_dir, f'gradients_step_{step}.pkl')
                    with open(grad_path, 'wb') as f:
                        pickle.dump(gradient_history[-args.save_interval:], f)
                    
                    # Save Hessians
                    if args.log_hessian and hessian_history:
                        hess_path = os.path.join(output_dir, f'hessians_step_{step}.pkl')
                        with open(hess_path, 'wb') as f:
                            recent_hessians = [h for h in hessian_history if h['step'] > step - args.save_interval]
                            pickle.dump(recent_hessians, f)
                
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
                        'train_loss': f'{loss.item():.4f}',
                        'test_loss': f'{test_loss:.4f}',
                        'train_acc': f'{train_acc:.3f}',
                        'test_acc': f'{test_acc:.3f}'
                    })
                
                step += 1
                pbar.update(1)
    
    # Save final data
    print("\nSaving final logs...")
    
    # Save all gradients
    final_grad_path = os.path.join(output_dir, 'gradients_final.pkl')
    with open(final_grad_path, 'wb') as f:
        pickle.dump(gradient_history, f)
    print(f"Saved {len(gradient_history)} gradient snapshots")
    
    # Save all Hessians
    if args.log_hessian and hessian_history:
        final_hess_path = os.path.join(output_dir, 'hessians_final.pkl')
        with open(final_hess_path, 'wb') as f:
            pickle.dump(hessian_history, f)
        print(f"Saved {len(hessian_history)} Hessian snapshots")
    
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
    
    # Save final checkpoint
    checkpoint_path = os.path.join(output_dir, 'final_checkpoint.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'args': args
    }, checkpoint_path)
    
    # Print final metrics
    print(f"\nFinal metrics:")
    print(f"Train accuracy: {train_accs[-1]:.3f}")
    print(f"Test accuracy: {test_accs[-1]:.3f}")
    print(f"Train loss: {train_losses[-1]:.4f}")
    print(f"Test loss: {test_losses[-1]:.4f}")
    print(f"\nAll logs saved to: {output_dir}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Grokking with gradient/Hessian logging')
    
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
    
    # Logging parameters
    parser.add_argument('--log_hessian', action='store_true', default=True, help='Log Hessian diagonal')
    parser.add_argument('--hessian_log_interval', type=int, default=500, help='Hessian logging interval')
    parser.add_argument('--save_interval', type=int, default=5000, help='Save checkpoint interval')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    
    args = parser.parse_args()
    
    # Run training
    train_grokking_with_logging(args)


if __name__ == '__main__':
    main()