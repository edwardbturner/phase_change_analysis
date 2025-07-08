import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from simple_model import SimpleModularAdditionMLP
from data import generate_modular_addition_data, create_data_loader


def compute_cos_grad_hgrad_and_norms(model, inputs, labels, criterion):
    """Compute cos(grad, H@grad) and gradient norm."""
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    
    # Compute gradient norm
    grad_norm = torch.sqrt(sum((g * g).sum() for g in grads)).item()
    
    grad_norm_sq = sum((g * g).sum() for g in grads)
    
    try:
        Hgrad = torch.autograd.grad(grad_norm_sq, params, retain_graph=True)
        
        grad_vec = torch.cat([g.flatten() for g in grads])
        Hgrad_vec = torch.cat([h.flatten() for h in Hgrad])
        
        cos_sim = torch.nn.functional.cosine_similarity(
            grad_vec.unsqueeze(0), 
            Hgrad_vec.unsqueeze(0)
        ).item()
        
        return cos_sim, grad_norm
    except:
        return None, grad_norm


def compute_weight_stats(model):
    """Compute weight norm squared and Gini coefficient."""
    all_weights = []
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only weights, not biases
            weights = param.detach().cpu().numpy().flatten()
            all_weights.extend(weights)
    
    all_weights = np.array(all_weights)
    
    # Total weight norm squared
    total_norm_squared = np.sum(all_weights ** 2)
    
    # Gini coefficient (sparsity measure)
    abs_weights = np.abs(all_weights)
    sorted_weights = np.sort(abs_weights)
    n = len(sorted_weights)
    if n > 0 and np.sum(sorted_weights) > 0:
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
    else:
        gini = 0
    
    return {
        'total_norm_squared': total_norm_squared,
        'gini': gini,
    }


def smooth_array(arr, window=10):
    """Apply moving average smoothing."""
    if len(arr) < window:
        return arr
    smoothed = np.convolve(arr, np.ones(window)/window, mode='valid')
    # Pad the beginning to maintain array length
    padding = np.full(window-1, smoothed[0])
    return np.concatenate([padding, smoothed])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(42)
    
    # Setup
    p = 97
    model = SimpleModularAdditionMLP(p=p, hidden_size=256).to(device)
    
    # Data
    train_data, test_data = generate_modular_addition_data(p=p, train_frac=0.3, seed=42)
    train_loader = create_data_loader(train_data, batch_size=512, shuffle=True)
    test_loader = create_data_loader(test_data, batch_size=512, shuffle=False)
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    
    # Logging
    metrics = {
        'steps': [],
        'cos_sims': [],
        'grad_norms': [],
        'test_accs': [],
        'train_accs': [],
        'weight_norms_squared': [],
        'gini_coeffs': [],
        'test_losses': [],
        'train_losses': []
    }
    
    print("Running cleanup analysis with gradient norm tracking (50k steps)...")
    print("This will take approximately 30-60 minutes on a good GPU\n")
    
    step = 0
    total_steps = 50000
    log_interval = 200
    
    pbar = tqdm(total=total_steps, desc='Training')
    
    while step < total_steps:
        for inputs, labels in train_loader:
            if step >= total_steps:
                break
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Standard training
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log metrics
            if step % log_interval == 0:
                # Cosine similarity and gradient norm
                cos_sim, grad_norm = compute_cos_grad_hgrad_and_norms(model, inputs, labels, criterion)
                
                # Weight statistics
                weight_stats = compute_weight_stats(model)
                
                # Accuracies and losses
                model.eval()
                with torch.no_grad():
                    # Train metrics
                    train_pred = outputs.argmax(dim=1)
                    train_acc = (train_pred == labels).float().mean().item()
                    train_loss = loss.item()
                    
                    # Test metrics
                    test_correct = 0
                    test_total = 0
                    test_loss_sum = 0
                    for test_inputs, test_labels in test_loader:
                        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                        test_outputs = model(test_inputs)
                        test_loss_sum += criterion(test_outputs, test_labels).item()
                        test_correct += (test_outputs.argmax(dim=1) == test_labels).sum().item()
                        test_total += test_labels.size(0)
                    test_acc = test_correct / test_total
                    test_loss = test_loss_sum / len(test_loader)
                
                # Store metrics
                metrics['steps'].append(step)
                metrics['cos_sims'].append(cos_sim if cos_sim is not None else 0)
                metrics['grad_norms'].append(grad_norm)
                metrics['test_accs'].append(test_acc)
                metrics['train_accs'].append(train_acc)
                metrics['weight_norms_squared'].append(weight_stats['total_norm_squared'])
                metrics['gini_coeffs'].append(weight_stats['gini'])
                metrics['test_losses'].append(test_loss)
                metrics['train_losses'].append(train_loss)
                
                if step % 2000 == 0:
                    pbar.set_postfix({
                        'test_acc': f'{test_acc:.3f}',
                        'cos': f'{cos_sim:.3f}' if cos_sim else 'N/A',
                        'grad_norm': f'{grad_norm:.2e}',
                        'W²': f'{weight_stats["total_norm_squared"]:.0f}'
                    })
            
            step += 1
            pbar.update(1)
    
    pbar.close()
    
    # Save metrics
    with open('cleanup_metrics_with_gradnorm.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    # Create enhanced analysis plots
    create_enhanced_plots(metrics)
    
    # Print enhanced analysis
    print_enhanced_analysis(metrics)


def create_enhanced_plots(metrics):
    """Create enhanced plots including gradient norm."""
    fig = plt.figure(figsize=(16, 24))
    
    # Convert to arrays
    steps = np.array(metrics['steps'])
    test_accs = np.array(metrics['test_accs'])
    cos_sims = np.array(metrics['cos_sims'])
    grad_norms = np.array(metrics['grad_norms'])
    weight_norms = np.array(metrics['weight_norms_squared'])
    gini_coeffs = np.array(metrics['gini_coeffs'])
    
    # Smooth gradient norms
    grad_norms_smooth = smooth_array(grad_norms, window=10)
    
    # Identify key phases
    grokking_start = None
    grokking_end = None
    if np.any(test_accs > 0.1):
        grokking_start = steps[np.where(test_accs > 0.1)[0][0]]
    if np.any(test_accs > 0.9):
        grokking_end = steps[np.where(test_accs > 0.9)[0][0]]
    
    # Find cosine similarity drop
    cos_drop_step = None
    if len(cos_sims) > 10:
        for i in range(len(cos_sims) - 5):
            if cos_sims[i] > 0.7 and cos_sims[i+5] < 0.3:
                cos_drop_step = steps[i+2]
                break
    
    # Plot 1: Test Accuracy
    ax1 = plt.subplot(6, 1, 1)
    ax1.plot(steps, test_accs, 'b-', linewidth=2, label='Test')
    ax1.plot(steps, metrics['train_accs'], 'g--', linewidth=1.5, label='Train', alpha=0.7)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Enhanced Cleanup Phase Analysis: With Gradient Norm Tracking', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot 2: Cosine Similarity
    ax2 = plt.subplot(6, 1, 2, sharex=ax1)
    ax2.plot(steps, cos_sims, 'r-', linewidth=2, label='cos(grad, H@grad)')
    ax2.set_ylabel('cos(grad, H@grad)', fontsize=12)
    ax2.set_title('Gradient-Hessian Eigenvector Alignment', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    
    # Plot 3: Gradient Norm (with smoothing)
    ax3 = plt.subplot(6, 1, 3, sharex=ax1)
    ax3.plot(steps, grad_norms, 'orange', linewidth=1, alpha=0.3, label='Raw')
    ax3.plot(steps, grad_norms_smooth, 'darkorange', linewidth=2, label='Smoothed')
    ax3.set_ylabel('Gradient Norm', fontsize=12)
    ax3.set_title('Gradient Norm Evolution (Key for Interpreting Cosine Drop)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    ax3.legend()
    
    # Plot 4: Weight Norm Squared
    ax4 = plt.subplot(6, 1, 4, sharex=ax1)
    ax4.plot(steps, weight_norms, 'g-', linewidth=2)
    ax4.set_ylabel('Σ||W||²', fontsize=12)
    ax4.set_title('Total Weight Norm Squared', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Plot 5: Gini Coefficient
    ax5 = plt.subplot(6, 1, 5, sharex=ax1)
    ax5.plot(steps, gini_coeffs, 'm-', linewidth=2)
    ax5.set_ylabel('Gini Coefficient', fontsize=12)
    ax5.set_title('Weight Distribution Sparsity', fontsize=14)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Key relationship - cos vs grad norm
    ax6 = plt.subplot(6, 1, 6, sharex=ax1)
    ax6_twin = ax6.twinx()
    
    line1 = ax6.plot(steps, cos_sims, 'r-', linewidth=2, label='cos(grad, H@grad)')
    line2 = ax6_twin.plot(steps, grad_norms_smooth, 'darkorange', linewidth=2, label='Grad Norm (smoothed)')
    
    ax6.set_ylabel('cos(grad, H@grad)', fontsize=12, color='red')
    ax6_twin.set_ylabel('Gradient Norm', fontsize=12, color='darkorange')
    ax6_twin.set_yscale('log')
    ax6.set_xlabel('Training Steps', fontsize=12)
    ax6.set_title('Critical Relationship: Cosine vs Gradient Norm', fontsize=14)
    ax6.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6.legend(lines, labels, loc='upper right')
    
    # Mark phases on all plots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        if grokking_start:
            ax.axvline(x=grokking_start, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        if grokking_end:
            ax.axvline(x=grokking_end, color='green', linestyle='--', linewidth=1, alpha=0.5)
        if cos_drop_step:
            ax.axvline(x=cos_drop_step, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add cleanup annotation
    if cos_drop_step:
        ax2.text(cos_drop_step, 0.5, 'CLEANUP', rotation=90, va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cleanup_analysis_with_gradnorm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n📊 Enhanced plot saved to: cleanup_analysis_with_gradnorm.png")


def print_enhanced_analysis(metrics):
    """Print enhanced analysis including gradient norm insights."""
    steps = np.array(metrics['steps'])
    test_accs = np.array(metrics['test_accs'])
    cos_sims = np.array(metrics['cos_sims'])
    grad_norms = np.array(metrics['grad_norms'])
    weight_norms = np.array(metrics['weight_norms_squared'])
    gini_coeffs = np.array(metrics['gini_coeffs'])
    
    print(f"\n{'='*80}")
    print("ENHANCED CLEANUP PHASE ANALYSIS - WITH GRADIENT NORM TRACKING")
    print(f"{'='*80}")
    
    # Find key transitions
    grokking_end = None
    cos_drop_step = None
    cos_drop_idx = None
    
    if np.any(test_accs > 0.9):
        grokking_end = steps[np.where(test_accs > 0.9)[0][0]]
    
    # Find cosine drop
    if len(cos_sims) > 10:
        for i in range(len(cos_sims) - 5):
            if cos_sims[i] > 0.7 and cos_sims[i+5] < 0.3:
                cos_drop_step = steps[i+2]
                cos_drop_idx = i+2
                break
    
    print(f"\n📅 TIMELINE:")
    print(f"   Grokking complete: step {grokking_end if grokking_end else 'N/A'}")
    print(f"   cos(grad,H@grad) drop: step {cos_drop_step if cos_drop_step else 'N/A'}")
    
    if cos_drop_step:
        print(f"\n🔍 CRITICAL ANALYSIS AT CLEANUP PHASE (step {cos_drop_step}):")
        
        # Get indices around the drop
        before_idx = max(0, cos_drop_idx - 3)
        after_idx = min(len(steps) - 1, cos_drop_idx + 3)
        
        print(f"\n   📊 METRICS BEFORE → AFTER:")
        print(f"   cos(grad,H@grad): {cos_sims[before_idx]:.3f} → {cos_sims[after_idx]:.3f}")
        print(f"   Gradient norm: {grad_norms[before_idx]:.2e} → {grad_norms[after_idx]:.2e}")
        print(f"   Weight norm²: {weight_norms[before_idx]:.1f} → {weight_norms[after_idx]:.1f}")
        print(f"   Gini coeff: {gini_coeffs[before_idx]:.3f} → {gini_coeffs[after_idx]:.3f}")
        
        # Key insight: Is cos drop due to grad norm going to zero?
        grad_ratio = grad_norms[after_idx] / grad_norms[before_idx]
        print(f"\n   🎯 KEY INSIGHT:")
        print(f"   Gradient norm change: {grad_ratio:.3f}x")
        
        if grad_ratio < 0.1:
            print(f"   ⚠️  CAUTION: Gradient norm drops significantly!")
            print(f"       The cos(grad,H@grad) drop might be due to vanishing gradients,")
            print(f"       not necessarily a change in the Hessian structure.")
        elif grad_ratio > 0.5:
            print(f"   ✅ GOOD: Gradient norm remains substantial.")
            print(f"       The cos(grad,H@grad) drop likely reflects real changes")
            print(f"       in the optimization landscape, not just vanishing gradients.")
        else:
            print(f"   🤔 UNCLEAR: Moderate gradient norm change.")
            print(f"       Could be a combination of both effects.")
        
        # Check other metrics
        weight_decreased = weight_norms[after_idx] < weight_norms[before_idx]
        gini_changed = abs(gini_coeffs[after_idx] - gini_coeffs[before_idx]) > 0.01
        
        print(f"\n   📈 NANDA'S CLEANUP SIGNATURES:")
        print(f"   Weight norm decrease: {'YES ✓' if weight_decreased else 'NO ✗'}")
        print(f"   Gini coefficient change: {'YES ✓' if gini_changed else 'NO ✗'}")
        
        # Final assessment
        print(f"\n   🏁 FINAL ASSESSMENT:")
        if grad_ratio > 0.5 and weight_decreased:
            print(f"   ✅ Strong evidence for cleanup phase!")
            print(f"      - Substantial gradients rule out numerical issues")
            print(f"      - Weight norm decrease confirms circuit pruning")
        elif grad_ratio < 0.1:
            print(f"   ❌ Likely NOT cleanup phase")
            print(f"      - Vanishing gradients explain the cosine drop")
            print(f"      - May just be reaching a local minimum")
        else:
            print(f"   ⚠️  Mixed evidence - needs further investigation")
    
    print(f"\n💾 Data saved to: cleanup_metrics_with_gradnorm.pkl")


if __name__ == "__main__":
    main()