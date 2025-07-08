import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('/workspace/grokking')
from simple_model import SimpleModularAdditionMLP
from data import generate_modular_addition_data, create_data_loader
import pickle


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
    
    return total_norm_squared, gini


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
    
    # Focus on steps 15k-30k where cleanup likely happens
    start_step = 15000
    end_step = 30000
    log_interval = 100  # More frequent logging in this range
    
    print(f"Quick cleanup phase check: steps {start_step}-{end_step}")
    print("Training to step 15k first...")
    
    # Quick training to step 15k
    step = 0
    pbar = tqdm(total=start_step, desc='Warmup')
    
    while step < start_step:
        for inputs, labels in train_loader:
            if step >= start_step:
                break
            
            inputs, labels = inputs.to(device), labels.to(device)
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            pbar.update(1)
    
    pbar.close()
    
    # Now detailed logging from 15k-30k
    metrics = {
        'steps': [],
        'cos_sims': [],
        'test_accs': [],
        'train_accs': [],
        'weight_norms': [],
        'gini_coeffs': []
    }
    
    print(f"\nDetailed logging from step {start_step} to {end_step}...")
    pbar = tqdm(total=end_step-start_step, desc='Detailed phase')
    
    while step < end_step:
        for inputs, labels in train_loader:
            if step >= end_step:
                break
            
            inputs, labels = inputs.to(device), labels.to(device)
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
                
                # Weight stats
                weight_norm, gini = compute_weight_stats(model)
                
                # Accuracies
                model.eval()
                with torch.no_grad():
                    train_acc = (outputs.argmax(dim=1) == labels).float().mean().item()
                    
                    # Test on one batch
                    test_inputs, test_labels = next(iter(test_loader))
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                    test_outputs = model(test_inputs)
                    test_acc = (test_outputs.argmax(dim=1) == test_labels).float().mean().item()
                
                metrics['steps'].append(step)
                metrics['cos_sims'].append(cos_sim if cos_sim is not None else 0)
                metrics['test_accs'].append(test_acc)
                metrics['train_accs'].append(train_acc)
                metrics['weight_norms'].append(weight_norm)
                metrics['gini_coeffs'].append(gini)
                
                if step % 1000 == 0:
                    pbar.set_postfix({
                        'test_acc': f'{test_acc:.3f}',
                        'cos': f'{cos_sim:.3f}' if cos_sim else 'N/A'
                    })
            
            step += 1
            pbar.update(1)
    
    pbar.close()
    
    # Save data
    with open('follow_ups/cleanup_metrics_focused.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    # Create plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    
    steps = np.array(metrics['steps'])
    cos_sims = np.array(metrics['cos_sims'])
    test_accs = np.array(metrics['test_accs'])
    weight_norms = np.array(metrics['weight_norms'])
    gini_coeffs = np.array(metrics['gini_coeffs'])
    
    # Find the cosine drop
    cos_drop_idx = None
    if len(cos_sims) > 5:
        for i in range(len(cos_sims) - 3):
            if cos_sims[i] > 0.7 and cos_sims[i+3] < 0.3:
                cos_drop_idx = i + 1
                break
    
    # Plot 1: Test Accuracy
    axes[0].plot(steps, test_accs, 'b-', linewidth=2)
    axes[0].set_ylabel('Test Accuracy', fontsize=12)
    axes[0].set_title('Cleanup Phase Analysis (Focused on Steps 15k-30k)', fontsize=16)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)
    
    # Plot 2: Cosine Similarity
    axes[1].plot(steps, cos_sims, 'r-', linewidth=2)
    axes[1].scatter(steps, cos_sims, c='red', s=20, alpha=0.5)
    axes[1].set_ylabel('cos(grad, H@grad)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 1.1)
    
    # Plot 3: Weight Norm
    axes[2].plot(steps, weight_norms, 'g-', linewidth=2)
    axes[2].set_ylabel('Σ||W||²', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    # Plot 4: Gini Coefficient  
    axes[3].plot(steps, gini_coeffs, 'm-', linewidth=2)
    axes[3].set_ylabel('Gini Coefficient', fontsize=12)
    axes[3].set_xlabel('Training Steps', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    
    # Mark cleanup phase if found
    if cos_drop_idx is not None:
        cleanup_step = steps[cos_drop_idx]
        for ax in axes:
            ax.axvline(x=cleanup_step, color='purple', linestyle='--', linewidth=2, alpha=0.7)
        axes[1].text(cleanup_step, 0.5, 'CLEANUP', rotation=90, va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('follow_ups/cleanup_phase_focused.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analysis
    print(f"\n{'='*60}")
    print("CLEANUP PHASE ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    if cos_drop_idx is not None:
        cleanup_step = steps[cos_drop_idx]
        print(f"\n✅ CLEANUP PHASE DETECTED at step {cleanup_step}")
        
        # Before/after analysis
        before_idx = max(0, cos_drop_idx - 3)
        after_idx = min(len(steps) - 1, cos_drop_idx + 3)
        
        print(f"\nMetrics before → after cleanup:")
        print(f"  cos(grad,H@grad): {cos_sims[before_idx]:.3f} → {cos_sims[after_idx]:.3f}")
        print(f"  Weight norm²: {weight_norms[before_idx]:.1f} → {weight_norms[after_idx]:.1f}")
        print(f"  Gini coeff: {gini_coeffs[before_idx]:.3f} → {gini_coeffs[after_idx]:.3f}")
        print(f"  Test accuracy: {test_accs[before_idx]:.3f} → {test_accs[after_idx]:.3f}")
        
        # Check if matches Nanda's predictions
        weight_decreased = weight_norms[after_idx] < weight_norms[before_idx]
        gini_increased = gini_coeffs[after_idx] > gini_coeffs[before_idx]
        
        print(f"\n🔬 Nanda's cleanup signatures:")
        print(f"  Weight norm decrease: {'✓' if weight_decreased else '✗'}")
        print(f"  Gini increase (sparsification): {'✓' if gini_increased else '✗'}")
    else:
        print("\n⚠️  No clear cleanup phase detected in this range")
    
    print(f"\nPlot saved to: follow_ups/cleanup_phase_focused.png")


if __name__ == "__main__":
    main()