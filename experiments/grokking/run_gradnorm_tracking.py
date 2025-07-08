import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
from simple_model import SimpleModularAdditionMLP
from data import generate_modular_addition_data, create_data_loader


def compute_metrics(model, inputs, labels, criterion):
    """Compute both gradient norm and cos(grad, H@grad)."""
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    
    # Gradient norm
    grad_norm = torch.sqrt(sum((g * g).sum() for g in grads)).item()
    
    # Cosine similarity
    grad_norm_sq = sum((g * g).sum() for g in grads)
    try:
        Hgrad = torch.autograd.grad(grad_norm_sq, params, retain_graph=True)
        grad_vec = torch.cat([g.flatten() for g in grads])
        Hgrad_vec = torch.cat([h.flatten() for h in Hgrad])
        cos_sim = torch.nn.functional.cosine_similarity(
            grad_vec.unsqueeze(0), Hgrad_vec.unsqueeze(0)
        ).item()
    except:
        cos_sim = None
    
    return grad_norm, cos_sim


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(42)
    
    # Check for existing checkpoint
    checkpoint_file = 'gradnorm_checkpoint.pkl'
    if os.path.exists(checkpoint_file):
        print("Loading from checkpoint...")
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        start_step = checkpoint['step']
        metrics = checkpoint['metrics']
        
        # Recreate model and optimizer state
        p = 97
        model = SimpleModularAdditionMLP(p=p, hidden_size=256).to(device)
        model.load_state_dict(checkpoint['model_state'])
        
        train_data, test_data = generate_modular_addition_data(p=p, train_frac=0.3, seed=42)
        train_loader = create_data_loader(train_data, batch_size=512, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        print(f"Resuming from step {start_step}")
    else:
        # Fresh start
        p = 97
        model = SimpleModularAdditionMLP(p=p, hidden_size=256).to(device)
        
        train_data, test_data = generate_modular_addition_data(p=p, train_frac=0.3, seed=42)
        train_loader = create_data_loader(train_data, batch_size=512, shuffle=True)
        test_loader = create_data_loader(test_data, batch_size=512, shuffle=False)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
        
        start_step = 0
        metrics = {
            'steps': [],
            'grad_norms': [],
            'cos_sims': [],
            'test_accs': [],
            'train_losses': []
        }
        print("Starting fresh training...")
    
    total_steps = 25000  # Focus on first 25k steps
    log_interval = 100   # Log every 100 steps for detailed tracking
    save_interval = 1000 # Save checkpoint every 1k steps
    
    print(f"Training with gradient norm tracking to step {total_steps}")
    print(f"Logging every {log_interval} steps, saving every {save_interval} steps")
    
    step = start_step
    pbar = tqdm(total=total_steps, initial=start_step, desc='Training')
    
    # Skip ahead in data loader if resuming
    if start_step > 0:
        for _ in range(start_step):
            try:
                next(iter(train_loader))
            except:
                pass
    
    while step < total_steps:
        for inputs, labels in train_loader:
            if step >= total_steps:
                break
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Standard training step
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log metrics
            if step % log_interval == 0:
                # Compute gradient norm and cosine similarity
                grad_norm, cos_sim = compute_metrics(model, inputs, labels, criterion)
                
                # Test accuracy (quick check)
                model.eval()
                with torch.no_grad():
                    test_inputs, test_labels = next(iter(test_loader))
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                    test_outputs = model(test_inputs)
                    test_acc = (test_outputs.argmax(dim=1) == test_labels).float().mean().item()
                
                # Store metrics
                metrics['steps'].append(step)
                metrics['grad_norms'].append(grad_norm)
                metrics['cos_sims'].append(cos_sim if cos_sim is not None else 0)
                metrics['test_accs'].append(test_acc)
                metrics['train_losses'].append(loss.item())
                
                # Update progress bar
                pbar.set_postfix({
                    'test_acc': f'{test_acc:.3f}',
                    'grad_norm': f'{grad_norm:.2e}',
                    'cos_sim': f'{cos_sim:.3f}' if cos_sim else 'N/A'
                })
            
            # Save checkpoint
            if step % save_interval == 0 and step > 0:
                checkpoint = {
                    'step': step,
                    'metrics': metrics,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict()
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint, f)
            
            step += 1
            pbar.update(1)
    
    pbar.close()
    
    # Save final results
    with open('gradnorm_final_results.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    # Create analysis
    analyze_results(metrics)


def analyze_results(metrics):
    """Analyze the gradient norm vs cosine similarity results."""
    steps = np.array(metrics['steps'])
    grad_norms = np.array(metrics['grad_norms'])
    cos_sims = np.array(metrics['cos_sims'])
    test_accs = np.array(metrics['test_accs'])
    
    print(f"\n{'='*70}")
    print("GRADIENT NORM vs COSINE SIMILARITY ANALYSIS")
    print(f"{'='*70}")
    
    # Find grokking point
    grok_step = None
    if np.any(test_accs > 0.9):
        grok_step = steps[np.where(test_accs > 0.9)[0][0]]
        print(f"🎯 Grokking complete at step: {grok_step}")
    
    # Find cosine drop
    cos_drop_step = None
    cos_drop_idx = None
    for i in range(len(cos_sims) - 5):
        if cos_sims[i] > 0.7 and cos_sims[i+5] < 0.3:
            cos_drop_step = steps[i+2]
            cos_drop_idx = i+2
            break
    
    if cos_drop_step:
        print(f"📉 Cosine drop detected at step: {cos_drop_step}")
        
        # Analyze gradient norm around the drop
        before_idx = max(0, cos_drop_idx - 5)
        after_idx = min(len(steps) - 1, cos_drop_idx + 5)
        
        grad_before = grad_norms[before_idx]
        grad_after = grad_norms[after_idx]
        cos_before = cos_sims[before_idx]
        cos_after = cos_sims[after_idx]
        
        print(f"\n🔍 CRITICAL ANALYSIS:")
        print(f"   Before (step {steps[before_idx]}):")
        print(f"     Gradient norm: {grad_before:.2e}")
        print(f"     cos(grad,H@grad): {cos_before:.3f}")
        print(f"   After (step {steps[after_idx]}):")
        print(f"     Gradient norm: {grad_after:.2e}")
        print(f"     cos(grad,H@grad): {cos_after:.3f}")
        
        grad_ratio = grad_after / grad_before
        print(f"\n   📊 Changes:")
        print(f"     Gradient norm ratio: {grad_ratio:.3f}x")
        print(f"     Cosine drop: {cos_before:.3f} → {cos_after:.3f}")
        
        print(f"\n💡 VERDICT:")
        if grad_ratio < 0.1:
            print("❌ GRADIENT VANISHING explains the cosine drop!")
            print("   The dramatic gradient norm decrease makes cos(grad,H@grad) unreliable.")
            print("   This is NOT evidence of cleanup phase - just numerical instability.")
        elif grad_ratio > 0.5:
            print("✅ MEANINGFUL COSINE DROP detected!")
            print("   Gradient norm remains substantial, so the cosine drop reflects")
            print("   real changes in the optimization landscape (likely cleanup phase).")
        else:
            print("⚠️  MIXED EVIDENCE - moderate gradient decrease")
            print("   Some gradient vanishing, but cosine drop might still be meaningful.")
    else:
        print("⚠️  No clear cosine drop detected in this range")
    
    # Create visualization
    create_visualization(metrics, grok_step, cos_drop_step)


def create_visualization(metrics, grok_step=None, cos_drop_step=None):
    """Create comprehensive visualization."""
    steps = np.array(metrics['steps'])
    grad_norms = np.array(metrics['grad_norms'])
    cos_sims = np.array(metrics['cos_sims'])
    test_accs = np.array(metrics['test_accs'])
    
    # Smooth gradient norms for clearer visualization
    from scipy.ndimage import uniform_filter1d
    grad_norms_smooth = uniform_filter1d(grad_norms, size=5)
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    
    # Plot 1: Test Accuracy
    axes[0].plot(steps, test_accs, 'b-', linewidth=2)
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Gradient Norm Hypothesis Test: Complete Analysis', fontsize=16)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)
    
    # Plot 2: Cosine Similarity
    axes[1].plot(steps, cos_sims, 'r-', linewidth=2, alpha=0.7)
    axes[1].scatter(steps[::5], cos_sims[::5], c='red', s=10, alpha=0.3)
    axes[1].set_ylabel('cos(grad, H@grad)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.1, 1.1)
    
    # Plot 3: Gradient Norm (log scale)
    axes[2].plot(steps, grad_norms, 'orange', linewidth=1, alpha=0.5, label='Raw')
    axes[2].plot(steps, grad_norms_smooth, 'darkorange', linewidth=2, label='Smoothed')
    axes[2].set_ylabel('Gradient Norm (log)')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    # Plot 4: Combined view with dual y-axis
    ax4 = axes[3]
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(steps, cos_sims, 'r-', linewidth=2, label='cos(grad, H@grad)')
    line2 = ax4_twin.plot(steps, grad_norms_smooth, 'darkorange', linewidth=2, label='Grad Norm (smoothed)')
    
    ax4.set_ylabel('cos(grad, H@grad)', color='red')
    ax4_twin.set_ylabel('Gradient Norm', color='darkorange')
    ax4_twin.set_yscale('log')
    ax4.set_xlabel('Training Steps')
    ax4.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right')
    
    # Mark key events
    for ax in axes[:3]:
        if grok_step:
            ax.axvline(x=grok_step, color='green', linestyle='--', linewidth=2, alpha=0.7)
        if cos_drop_step:
            ax.axvline(x=cos_drop_step, color='purple', linestyle='--', linewidth=2, alpha=0.7)
    
    # Mark on dual axis plot
    if grok_step:
        ax4.axvline(x=grok_step, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax4_twin.axvline(x=grok_step, color='green', linestyle='--', linewidth=2, alpha=0.7)
    if cos_drop_step:
        ax4.axvline(x=cos_drop_step, color='purple', linestyle='--', linewidth=2, alpha=0.7)
        ax4_twin.axvline(x=cos_drop_step, color='purple', linestyle='--', linewidth=2, alpha=0.7)
        ax4.text(cos_drop_step, 0.5, 'CRITICAL\nTEST', rotation=0, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('gradient_norm_hypothesis_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Visualization saved: gradient_norm_hypothesis_test.png")
    print(f"📊 Data saved: gradnorm_final_results.pkl")


if __name__ == "__main__":
    main()