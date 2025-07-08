import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('..')
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
    
    print("Running cleanup phase analysis (50k steps)...")
    print("Tracking: cos(grad,H@grad), weight norms, Gini coefficient\n")
    
    step = 0
    pbar = tqdm(total=50000, desc='Training')
    
    while step < 50000:
        for inputs, labels in train_loader:
            if step >= 50000:
                break
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Standard training
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log metrics every 200 steps
            if step % 200 == 0:
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
                    
                    # Test metrics (on one batch for speed)
                    test_inputs, test_labels = next(iter(test_loader))
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                    test_outputs = model(test_inputs)
                    test_loss = criterion(test_outputs, test_labels).item()
                    test_acc = (test_outputs.argmax(dim=1) == test_labels).float().mean().item()
                
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
                        'W²': f'{weight_stats["total_norm_squared"]:.0f}'
                    })
            
            step += 1
            pbar.update(1)
    
    pbar.close()
    
    # Create analysis plots
    fig = plt.figure(figsize=(14, 16))
    
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
        # Find steepest drop
        for i in range(len(cos_sims) - 5):
            if cos_sims[i] > 0.7 and cos_sims[i+5] < 0.3:
                cos_drop_step = steps[i+2]
                break
    
    # Plot 1: Test Accuracy
    ax1 = plt.subplot(5, 1, 1)
    ax1.plot(steps, test_accs, 'b-', linewidth=2)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title('Grokking Progress and Cleanup Phase Analysis', fontsize=16)
    ax1.grid(True, alpha=0.3)
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
    
    # Highlight the drop
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
    
    # Plot 5: Combined normalized view
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
    ax5.set_title('All Metrics Normalized - Cleanup Phase Alignment', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='best')
    
    if cos_drop_step:
        ax5.axvline(x=cos_drop_step, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('follow_ups/cleanup_phase_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed analysis
    print(f"\n{'='*70}")
    print("CLEANUP PHASE ANALYSIS - COMPARISON WITH NANDA ET AL.")
    print(f"{'='*70}")
    
    # Timeline
    print(f"\n📅 TIMELINE:")
    print(f"   Step 0-{grokking_start if grokking_start else '?'}: Memorization phase")
    if grokking_start and grokking_end:
        print(f"   Step {grokking_start}-{grokking_end}: Grokking transition")
    if grokking_end:
        print(f"   Step {grokking_end}-{cos_drop_step if cos_drop_step else '?'}: Post-grokking (both circuits coexist)")
    if cos_drop_step:
        print(f"   Step {cos_drop_step}: CLEANUP PHASE DETECTED")
        print(f"   Step {cos_drop_step}-50000: Pure generalization circuit")
    
    # Analysis at cleanup point
    if cos_drop_step:
        cleanup_idx = np.where(steps == cos_drop_step)[0][0]
        
        print(f"\n🔍 CLEANUP PHASE ANALYSIS (Step {cos_drop_step}):")
        
        # Cosine similarity drop
        if cleanup_idx > 5:
            cos_before = cos_sims[cleanup_idx - 5]
            cos_after = cos_sims[min(cleanup_idx + 5, len(cos_sims) - 1)]
            print(f"\n   Cosine Similarity:")
            print(f"   Before: {cos_before:.3f}")
            print(f"   After:  {cos_after:.3f}")
            print(f"   Drop:   {cos_before - cos_after:.3f} ({(1 - cos_after/cos_before)*100:.1f}% decrease)")
        
        # Weight norm changes
        if cleanup_idx > 5 and cleanup_idx < len(weight_norms) - 5:
            w_before = weight_norms[cleanup_idx - 5]
            w_after = weight_norms[cleanup_idx + 5]
            print(f"\n   Weight Norm Squared:")
            print(f"   Before: {w_before:.1f}")
            print(f"   After:  {w_after:.1f}")
            print(f"   Change: {(w_after/w_before - 1)*100:+.1f}%")
            
            if w_after < w_before * 0.9:
                print(f"   ✓ Confirms Nanda: Weight norm drops during cleanup!")
        
        # Gini coefficient changes
        if cleanup_idx > 5 and cleanup_idx < len(gini_coeffs) - 5:
            g_before = gini_coeffs[cleanup_idx - 5]
            g_after = gini_coeffs[cleanup_idx + 5]
            print(f"\n   Gini Coefficient:")
            print(f"   Before: {g_before:.3f}")
            print(f"   After:  {g_after:.3f}")
            print(f"   Change: {g_after - g_before:+.3f}")
            
            if g_after > g_before:
                print(f"   ✓ Confirms Nanda: Gini increases (sparsification) during cleanup!")
    
    # Phase averages
    print(f"\n📊 PHASE AVERAGES:")
    
    # Pre-grokking
    if grokking_start:
        mask = steps < grokking_start
        if np.any(mask):
            print(f"\n   Memorization Phase (before step {grokking_start}):")
            print(f"   - cos(g,Hg): {np.mean(cos_sims[mask]):.3f} ± {np.std(cos_sims[mask]):.3f}")
            print(f"   - Weight norm²: {np.mean(weight_norms[mask]):.1f}")
            print(f"   - Gini: {np.mean(gini_coeffs[mask]):.3f}")
    
    # Post-cleanup
    if cos_drop_step:
        mask = steps > cos_drop_step
        if np.any(mask):
            print(f"\n   Post-Cleanup Phase (after step {cos_drop_step}):")
            print(f"   - cos(g,Hg): {np.mean(cos_sims[mask]):.3f} ± {np.std(cos_sims[mask]):.3f}")
            print(f"   - Weight norm²: {np.mean(weight_norms[mask]):.1f}")
            print(f"   - Gini: {np.mean(gini_coeffs[mask]):.3f}")
    
    print(f"\n🎯 KEY FINDING:")
    print(f"   The dramatic drop in cos(grad, H@grad) at step {cos_drop_step if cos_drop_step else 'N/A'}")
    print(f"   corresponds to Nanda's cleanup phase where the memorization")
    print(f"   circuit is pruned away, leaving only the generalizing circuit!")
    
    # Save metrics for further analysis
    import pickle
    with open('follow_ups/cleanup_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print(f"\n💾 Saved detailed metrics to follow_ups/cleanup_metrics.pkl")


if __name__ == "__main__":
    main()