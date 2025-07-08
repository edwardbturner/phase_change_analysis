import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simple_model import SimpleModularAdditionMLP
from data import generate_modular_addition_data, create_data_loader
import matplotlib.patches as mpatches


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
    """Compute various weight statistics."""
    all_weights = []
    weight_norms_by_layer = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only look at weights, not biases
            weights = param.detach().cpu().numpy().flatten()
            all_weights.extend(weights)
            weight_norms_by_layer[name] = np.linalg.norm(weights)
    
    all_weights = np.array(all_weights)
    
    # Total weight norm squared (for weight decay)
    total_norm_squared = np.sum(all_weights ** 2)
    
    # Gini coefficient (measure of sparsity/inequality)
    abs_weights = np.abs(all_weights)
    sorted_weights = np.sort(abs_weights)
    n = len(sorted_weights)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n
    
    # Sparsity (fraction of near-zero weights)
    sparsity = np.mean(np.abs(all_weights) < 1e-3)
    
    return {
        'total_norm_squared': total_norm_squared,
        'gini': gini,
        'sparsity': sparsity,
        'weight_norms': weight_norms_by_layer
    }


def compute_restricted_loss(model, train_loader, test_loader, criterion, device, 
                           exclude_fraction=0.1):
    """Compute loss on data excluding some fraction of most confident predictions."""
    model.eval()
    
    # Get predictions on train set
    train_losses = []
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get per-sample losses
            losses = criterion(outputs, labels.unsqueeze(0).expand(outputs.shape[0], -1).T).mean(dim=1)
            train_losses.extend(losses.cpu().numpy())
    
    # Exclude most confident (lowest loss) predictions
    train_losses = np.array(train_losses)
    threshold = np.percentile(train_losses, exclude_fraction * 100)
    restricted_loss = np.mean(train_losses[train_losses > threshold])
    
    return restricted_loss


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
        'train_losses': [],
        'restricted_losses': []
    }
    
    print("Running comprehensive grokking analysis (50k steps)...")
    print("Tracking: cos(grad,H@grad), weight norms, Gini coefficient, sparsity")
    print(f"Train data size: {len(train_data[0])}, Test data size: {len(test_data[0])}\n")
    
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
                    
                    # Test metrics
                    test_correct = 0
                    test_total = 0
                    test_loss = 0
                    for test_inputs, test_labels in test_loader:
                        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                        test_outputs = model(test_inputs)
                        test_loss += criterion(test_outputs, test_labels).item()
                        test_correct += (test_outputs.argmax(dim=1) == test_labels).sum().item()
                        test_total += test_labels.size(0)
                    test_acc = test_correct / test_total
                    test_loss = test_loss / len(test_loader)
                
                # Restricted loss (expensive, compute less frequently)
                if step % 1000 == 0:
                    restricted_loss = compute_restricted_loss(
                        model, train_loader, test_loader, criterion, device
                    )
                else:
                    restricted_loss = metrics['restricted_losses'][-1] if metrics['restricted_losses'] else 0
                
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
                metrics['restricted_losses'].append(restricted_loss)
                
                if step % 2000 == 0:
                    pbar.set_postfix({
                        'test_acc': f'{test_acc:.3f}',
                        'cos': f'{cos_sim:.3f}' if cos_sim else 'N/A',
                        'gini': f'{weight_stats["gini"]:.3f}'
                    })
            
            step += 1
            pbar.update(1)
    
    pbar.close()
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 20))
    
    # Identify phases based on test accuracy
    test_accs = np.array(metrics['test_accs'])
    steps = np.array(metrics['steps'])
    
    # Phase boundaries
    memorization_end = None
    grokking_start = None
    grokking_end = None
    
    if np.any(test_accs > 0.1):
        grokking_start = steps[np.where(test_accs > 0.1)[0][0]]
    if np.any(test_accs > 0.9):
        grokking_end = steps[np.where(test_accs > 0.9)[0][0]]
    
    # Helper function to add phase shading
    def add_phase_shading(ax):
        if grokking_start:
            ax.axvspan(0, grokking_start, alpha=0.1, color='red', label='Memorization')
        if grokking_start and grokking_end:
            ax.axvspan(grokking_start, grokking_end, alpha=0.1, color='orange', label='Grokking')
        if grokking_end:
            ax.axvspan(grokking_end, 50000, alpha=0.1, color='green', label='Post-grokking')
    
    # 1. Test Accuracy
    ax1 = plt.subplot(6, 1, 1)
    ax1.plot(steps, test_accs, 'b-', linewidth=2)
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy - Grokking Progress', fontsize=14)
    ax1.grid(True, alpha=0.3)
    add_phase_shading(ax1)
    ax1.set_ylim(-0.05, 1.05)
    
    # 2. Cosine Similarity
    ax2 = plt.subplot(6, 1, 2, sharex=ax1)
    ax2.plot(steps, metrics['cos_sims'], 'r-', linewidth=2)
    ax2.set_ylabel('cos(grad, H@grad)')
    ax2.set_title('Gradient-Hessian Alignment', fontsize=14)
    ax2.grid(True, alpha=0.3)
    add_phase_shading(ax2)
    ax2.set_ylim(-0.1, 1.1)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Find and mark the drop
    cos_sims = np.array(metrics['cos_sims'])
    if len(cos_sims) > 10:
        cos_diff = np.diff(cos_sims)
        biggest_drop_idx = np.argmin(cos_diff)
        if cos_diff[biggest_drop_idx] < -0.2:  # Significant drop
            drop_step = steps[biggest_drop_idx]
            ax2.axvline(x=drop_step, color='purple', linestyle='--', linewidth=2)
            ax2.text(drop_step, 0.5, 'Cleanup?', rotation=90, va='bottom', color='purple')
    
    # 3. Weight Norm Squared (Nanda's metric)
    ax3 = plt.subplot(6, 1, 3, sharex=ax1)
    ax3.plot(steps, metrics['weight_norms_squared'], 'g-', linewidth=2)
    ax3.set_ylabel('Σ||W||²')
    ax3.set_title('Total Weight Norm Squared (Weight Decay Target)', fontsize=14)
    ax3.grid(True, alpha=0.3)
    add_phase_shading(ax3)
    ax3.set_yscale('log')
    
    # 4. Gini Coefficient (Nanda's sparsity metric)
    ax4 = plt.subplot(6, 1, 4, sharex=ax1)
    ax4.plot(steps, metrics['gini_coeffs'], 'm-', linewidth=2)
    ax4.set_ylabel('Gini Coefficient')
    ax4.set_title('Weight Sparsity (Higher = More Sparse)', fontsize=14)
    ax4.grid(True, alpha=0.3)
    add_phase_shading(ax4)
    
    # 5. Losses
    ax5 = plt.subplot(6, 1, 5, sharex=ax1)
    ax5.plot(steps, metrics['train_losses'], 'g-', linewidth=2, alpha=0.7, label='Train')
    ax5.plot(steps, metrics['test_losses'], 'b-', linewidth=2, alpha=0.7, label='Test')
    ax5.set_ylabel('Loss')
    ax5.set_title('Training and Test Losses', fontsize=14)
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    add_phase_shading(ax5)
    
    # 6. All metrics normalized for comparison
    ax6 = plt.subplot(6, 1, 6, sharex=ax1)
    
    # Normalize metrics to [0, 1] for comparison
    def normalize(arr):
        arr = np.array(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    
    ax6.plot(steps, normalize(test_accs), 'b-', linewidth=2, label='Test Acc')
    ax6.plot(steps, normalize(metrics['cos_sims']), 'r-', linewidth=2, label='cos(g,Hg)')
    ax6.plot(steps, 1 - normalize(metrics['weight_norms_squared']), 'g-', linewidth=2, label='1 - Weight Norm')
    ax6.plot(steps, normalize(metrics['gini_coeffs']), 'm-', linewidth=2, label='Gini')
    
    ax6.set_xlabel('Training Steps')
    ax6.set_ylabel('Normalized Value')
    ax6.set_title('All Metrics Normalized for Comparison', fontsize=14)
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    add_phase_shading(ax6)
    
    plt.tight_layout()
    plt.savefig('cleanup_phase_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print analysis
    print(f"\n{'='*60}")
    print("CLEANUP PHASE ANALYSIS")
    print(f"{'='*60}")
    
    # Find the cosine similarity drop
    if len(cos_sims) > 10:
        cos_diff = np.diff(cos_sims)
        biggest_drop_idx = np.argmin(cos_diff)
        drop_magnitude = -cos_diff[biggest_drop_idx]
        
        if drop_magnitude > 0.2:
            drop_step = steps[biggest_drop_idx]
            print(f"\n🔍 Detected cleanup phase at step {drop_step}:")
            print(f"   cos(grad,H@grad) dropped by {drop_magnitude:.3f}")
            
            # Check other metrics at this point
            print(f"\n   At cleanup (step {drop_step}):")
            print(f"   - Test accuracy: {test_accs[biggest_drop_idx]:.3f}")
            print(f"   - Weight norm²: {metrics['weight_norms_squared'][biggest_drop_idx]:.1f}")
            print(f"   - Gini coefficient: {metrics['gini_coeffs'][biggest_drop_idx]:.3f}")
            
            # Compare before/after
            if biggest_drop_idx > 5:
                before_idx = biggest_drop_idx - 5
                after_idx = min(biggest_drop_idx + 5, len(steps) - 1)
                
                print(f"\n   Changes during cleanup:")
                print(f"   - Weight norm²: {metrics['weight_norms_squared'][before_idx]:.1f} → "
                      f"{metrics['weight_norms_squared'][after_idx]:.1f} "
                      f"({(metrics['weight_norms_squared'][after_idx]/metrics['weight_norms_squared'][before_idx] - 1)*100:+.1f}%)")
                print(f"   - Gini: {metrics['gini_coeffs'][before_idx]:.3f} → "
                      f"{metrics['gini_coeffs'][after_idx]:.3f} "
                      f"({(metrics['gini_coeffs'][after_idx] - metrics['gini_coeffs'][before_idx]):+.3f})")
                print(f"   - Sparsity: {metrics['sparsities'][before_idx]:.3f} → "
                      f"{metrics['sparsities'][after_idx]:.3f}")
    
    # Phase analysis
    if grokking_start and grokking_end:
        print(f"\n📊 Phase timeline:")
        print(f"   - Memorization: steps 0 - {grokking_start}")
        print(f"   - Grokking: steps {grokking_start} - {grokking_end}")
        print(f"   - Post-grokking: steps {grokking_end} - 50000")
        
        # Average metrics per phase
        print(f"\n📈 Average metrics by phase:")
        
        # Memorization phase
        mem_mask = steps < grokking_start
        if np.any(mem_mask):
            print(f"\n   Memorization phase:")
            print(f"   - cos(g,Hg): {np.mean(cos_sims[mem_mask]):.3f}")
            print(f"   - Weight norm²: {np.mean(np.array(metrics['weight_norms_squared'])[mem_mask]):.1f}")
            print(f"   - Gini: {np.mean(np.array(metrics['gini_coeffs'])[mem_mask]):.3f}")
        
        # Post-grokking phase
        post_mask = steps > grokking_end
        if np.any(post_mask):
            print(f"\n   Post-grokking phase:")
            print(f"   - cos(g,Hg): {np.mean(cos_sims[post_mask]):.3f}")
            print(f"   - Weight norm²: {np.mean(np.array(metrics['weight_norms_squared'])[post_mask]):.1f}")
            print(f"   - Gini: {np.mean(np.array(metrics['gini_coeffs'])[post_mask]):.3f}")
    
    print(f"\n✅ Analysis complete! See 'cleanup_phase_analysis.png' for visualization")


if __name__ == "__main__":
    main()