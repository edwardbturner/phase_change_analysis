import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from simple_model import SimpleModularAdditionMLP
from data import generate_modular_addition_data, create_data_loader


def compute_cos_grad_hgrad(model, inputs, labels, criterion):
    """Compute cos(grad, H@grad) using autograd."""
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    
    grad_norm_sq = sum((g * g).sum() for g in grads)
    
    try:
        Hgrad = torch.autograd.grad(grad_norm_sq, params, retain_graph=True)
        
        grad_vec = torch.cat([g.flatten() for g in grads])
        Hgrad_vec = torch.cat([h.flatten() for h in Hgrad])
        
        cos_sim = torch.nn.functional.cosine_similarity(
            grad_vec.unsqueeze(0), 
            Hgrad_vec.unsqueeze(0)
        ).item()
        
        return cos_sim
    except:
        return None


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
    
    # Sparsity (fraction of small weights)
    sparsity = np.mean(np.abs(all_weights) < 0.01)
    
    return {
        'total_norm_squared': total_norm_squared,
        'gini': gini,
        'sparsity': sparsity
    }


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
        'test_accs': [],
        'train_accs': [],
        'weight_norms_squared': [],
        'gini_coeffs': [],
        'sparsities': [],
        'test_losses': [],
        'train_losses': []
    }
    
    print("Running full cleanup phase analysis (50k steps)...")
    print("Tracking: cos(grad,H@grad), weight norms, Gini coefficient")
    print("This will take approximately 30-60 minutes on a good GPU\n")
    
    step = 0
    total_steps = 50000
    log_interval = 200  # Log every 200 steps
    save_interval = 5000  # Save checkpoint every 5k steps
    
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
                # Cosine similarity
                cos_sim = compute_cos_grad_hgrad(model, inputs, labels, criterion)
                
                # Weight statistics
                weight_stats = compute_weight_stats(model)
                
                # Accuracies and losses
                model.eval()
                with torch.no_grad():
                    # Train metrics
                    train_pred = outputs.argmax(dim=1)
                    train_acc = (train_pred == labels).float().mean().item()
                    train_loss = loss.item()
                    
                    # Test metrics (on full test set for accuracy)
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
                metrics['test_accs'].append(test_acc)
                metrics['train_accs'].append(train_acc)
                metrics['weight_norms_squared'].append(weight_stats['total_norm_squared'])
                metrics['gini_coeffs'].append(weight_stats['gini'])
                metrics['sparsities'].append(weight_stats['sparsity'])
                metrics['test_losses'].append(test_loss)
                metrics['train_losses'].append(train_loss)
                
                if step % 2000 == 0:
                    pbar.set_postfix({
                        'test_acc': f'{test_acc:.3f}',
                        'cos': f'{cos_sim:.3f}' if cos_sim else 'N/A',
                        'W²': f'{weight_stats["total_norm_squared"]:.0f}',
                        'gini': f'{weight_stats["gini"]:.3f}'
                    })
            
            # Save checkpoint
            if step % save_interval == 0 and step > 0:
                with open(f'cleanup_metrics_checkpoint_{step}.pkl', 'wb') as f:
                    pickle.dump(metrics, f)
                print(f"\nCheckpoint saved at step {step}")
            
            step += 1
            pbar.update(1)
    
    pbar.close()
    
    # Save final metrics
    with open('cleanup_metrics_final.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    # Create comprehensive analysis plots
    create_analysis_plots(metrics)
    
    # Print analysis
    print_analysis(metrics)


def create_analysis_plots(metrics):
    """Create comprehensive analysis plots."""
    fig = plt.figure(figsize=(16, 20))
    
    # Convert to arrays
    steps = np.array(metrics['steps'])
    test_accs = np.array(metrics['test_accs'])
    cos_sims = np.array(metrics['cos_sims'])
    weight_norms = np.array(metrics['weight_norms_squared'])
    gini_coeffs = np.array(metrics['gini_coeffs'])
    
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
    ax1 = plt.subplot(5, 1, 1)
    ax1.plot(steps, test_accs, 'b-', linewidth=2, label='Test')
    ax1.plot(steps, metrics['train_accs'], 'g--', linewidth=1.5, label='Train', alpha=0.7)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Cleanup Phase Analysis: Full 50k Steps', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-0.05, 1.05)
    
    # Mark phases
    if grokking_start:
        ax1.axvline(x=grokking_start, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(grokking_start, 0.5, 'Grokking starts', rotation=90, va='bottom')
    if grokking_end:
        ax1.axvline(x=grokking_end, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(grokking_end, 0.5, 'Grokking complete', rotation=90, va='bottom')
    if cos_drop_step:
        ax1.axvline(x=cos_drop_step, color='purple', linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(cos_drop_step, 0.5, 'CLEANUP', rotation=90, va='bottom', fontweight='bold')
    
    # Plot 2: Cosine Similarity
    ax2 = plt.subplot(5, 1, 2, sharex=ax1)
    ax2.plot(steps, cos_sims, 'r-', linewidth=2)
    ax2.scatter(steps, cos_sims, c='red', s=10, alpha=0.3)
    ax2.set_ylabel('cos(grad, H@grad)', fontsize=12)
    ax2.set_title('Gradient-Hessian Eigenvector Alignment', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    if cos_drop_step:
        ax2.axvline(x=cos_drop_step, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    
    # Plot 3: Weight Norm Squared
    ax3 = plt.subplot(5, 1, 3, sharex=ax1)
    ax3.plot(steps, weight_norms, 'g-', linewidth=2)
    ax3.set_ylabel('Σ||W||²', fontsize=12)
    ax3.set_title('Total Weight Norm Squared (Nanda\'s Metric)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    if cos_drop_step:
        ax3.axvline(x=cos_drop_step, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    
    # Plot 4: Gini Coefficient
    ax4 = plt.subplot(5, 1, 4, sharex=ax1)
    ax4.plot(steps, gini_coeffs, 'm-', linewidth=2)
    ax4.set_ylabel('Gini Coefficient', fontsize=12)
    ax4.set_title('Weight Distribution Sparsity (Nanda\'s Metric)', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    if cos_drop_step:
        ax4.axvline(x=cos_drop_step, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    
    # Plot 5: All metrics normalized
    ax5 = plt.subplot(5, 1, 5, sharex=ax1)
    
    # Normalize for comparison
    def normalize(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    
    ax5.plot(steps, normalize(test_accs), 'b-', linewidth=2, label='Test Acc', alpha=0.8)
    ax5.plot(steps, normalize(cos_sims), 'r-', linewidth=2, label='cos(g,Hg)', alpha=0.8)
    ax5.plot(steps, 1 - normalize(weight_norms), 'g-', linewidth=2, label='1 - ||W||²', alpha=0.8)
    ax5.plot(steps, normalize(gini_coeffs), 'm-', linewidth=2, label='Gini', alpha=0.8)
    
    ax5.set_xlabel('Training Steps', fontsize=12)
    ax5.set_ylabel('Normalized Value', fontsize=12)
    ax5.set_title('All Metrics Normalized - Cleanup Phase Alignment Check', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='best')
    
    if cos_drop_step:
        ax5.axvline(x=cos_drop_step, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('cleanup_phase_analysis_complete.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n📊 Plot saved to: cleanup_phase_analysis_complete.png")


def print_analysis(metrics):
    """Print detailed analysis of the results."""
    steps = np.array(metrics['steps'])
    test_accs = np.array(metrics['test_accs'])
    cos_sims = np.array(metrics['cos_sims'])
    weight_norms = np.array(metrics['weight_norms_squared'])
    gini_coeffs = np.array(metrics['gini_coeffs'])
    
    print(f"\n{'='*70}")
    print("CLEANUP PHASE ANALYSIS - COMPLETE RESULTS")
    print(f"{'='*70}")
    
    # Find key transitions
    grokking_start = None
    grokking_end = None
    cos_drop_step = None
    
    if np.any(test_accs > 0.1):
        grokking_start = steps[np.where(test_accs > 0.1)[0][0]]
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
    print(f"   Grokking starts: step {grokking_start if grokking_start else 'N/A'}")
    print(f"   Grokking complete: step {grokking_end if grokking_end else 'N/A'}")
    print(f"   cos(grad,H@grad) drop: step {cos_drop_step if cos_drop_step else 'N/A'}")
    
    if cos_drop_step:
        print(f"\n🔍 CLEANUP PHASE ANALYSIS (around step {cos_drop_step}):")
        
        # Get indices around the drop
        before_idx = max(0, cos_drop_idx - 5)
        after_idx = min(len(steps) - 1, cos_drop_idx + 5)
        
        print(f"\n   Cosine Similarity:")
        print(f"   Before: {cos_sims[before_idx]:.3f}")
        print(f"   After:  {cos_sims[after_idx]:.3f}")
        print(f"   Change: {cos_sims[after_idx] - cos_sims[before_idx]:.3f}")
        
        print(f"\n   Weight Norm Squared:")
        print(f"   Before: {weight_norms[before_idx]:.1f}")
        print(f"   After:  {weight_norms[after_idx]:.1f}")
        print(f"   Change: {(weight_norms[after_idx]/weight_norms[before_idx] - 1)*100:+.1f}%")
        
        print(f"\n   Gini Coefficient:")
        print(f"   Before: {gini_coeffs[before_idx]:.3f}")
        print(f"   After:  {gini_coeffs[after_idx]:.3f}")
        print(f"   Change: {gini_coeffs[after_idx] - gini_coeffs[before_idx]:+.3f}")
        
        # Check Nanda's signatures
        print(f"\n✅ NANDA'S CLEANUP SIGNATURES:")
        weight_decreased = weight_norms[after_idx] < weight_norms[before_idx]
        gini_increased = gini_coeffs[after_idx] > gini_coeffs[before_idx]
        
        print(f"   Weight norm decrease: {'YES ✓' if weight_decreased else 'NO ✗'}")
        print(f"   Gini increase (sparsification): {'YES ✓' if gini_increased else 'NO ✗'}")
        print(f"   cos(grad,H@grad) drop: YES ✓")
        
        if weight_decreased and gini_increased:
            print(f"\n   🎯 CONFIRMED: This corresponds to Nanda's cleanup phase!")
        else:
            print(f"\n   ⚠️  Mixed evidence for Nanda's cleanup phase")
    
    print(f"\n💾 Data saved to: cleanup_metrics_final.pkl")
    print(f"   You can load and analyze further with:")
    print(f"   >>> import pickle")
    print(f"   >>> with open('cleanup_metrics_final.pkl', 'rb') as f:")
    print(f"   >>>     data = pickle.load(f)")


if __name__ == "__main__":
    main()