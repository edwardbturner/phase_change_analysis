import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simple_model import SimpleModularAdditionMLP
from data import generate_modular_addition_data, create_data_loader


def compute_cos_grad_hgrad(model, inputs, labels, criterion):
    """Compute cos(grad, H@grad) using autograd."""
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Get gradient
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    
    # Compute g^T @ g
    grad_norm_sq = sum((g * g).sum() for g in grads)
    
    # Get H @ grad by taking gradient of g^T @ g
    try:
        Hgrad = torch.autograd.grad(grad_norm_sq, params, retain_graph=True)
        
        # Compute cosine similarity
        grad_vec = torch.cat([g.flatten() for g in grads])
        Hgrad_vec = torch.cat([h.flatten() for h in Hgrad])
        
        cos_sim = torch.nn.functional.cosine_similarity(
            grad_vec.unsqueeze(0), 
            Hgrad_vec.unsqueeze(0)
        ).item()
        
        return cos_sim
    except:
        return None


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
    cos_sims = []
    test_accs = []
    train_accs = []
    steps_logged = []
    
    print("Running full grokking experiment (50k steps)...")
    print("Logging cos(grad, H@grad) every 200 steps")
    
    step = 0
    for epoch in tqdm(range(100), desc='Training'):
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
            
            # Log every 200 steps
            if step % 200 == 0:
                # Compute eigenvector alignment
                with torch.no_grad():
                    batch_inputs, batch_labels = next(iter(train_loader))
                    batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                
                cos_sim = compute_cos_grad_hgrad(model, batch_inputs, batch_labels, criterion)
                
                if cos_sim is not None:
                    # Compute accuracies
                    model.eval()
                    with torch.no_grad():
                        # Train accuracy
                        train_pred = outputs.argmax(dim=1)
                        train_acc = (train_pred == labels).float().mean().item()
                        
                        # Test accuracy (full test set)
                        test_correct = 0
                        test_total = 0
                        for test_inputs, test_labels in test_loader:
                            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                            test_outputs = model(test_inputs)
                            test_correct += (test_outputs.argmax(dim=1) == test_labels).sum().item()
                            test_total += test_labels.size(0)
                        test_acc = test_correct / test_total
                    
                    cos_sims.append(cos_sim)
                    test_accs.append(test_acc)
                    train_accs.append(train_acc)
                    steps_logged.append(step)
                    
                    if step % 2000 == 0:
                        print(f"Step {step}: cos={cos_sim:+.3f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")
            
            step += 1
            
        if step >= 50000:
            break
    
    # Create the plot
    fig = plt.figure(figsize=(14, 10))
    
    # Main plot: cosine similarity over time
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(steps_logged, cos_sims, 'r-', linewidth=2, alpha=0.8, label='cos(grad, H@grad)')
    ax1.scatter(steps_logged, cos_sims, c='red', s=10, alpha=0.3)
    ax1.set_ylabel('Cosine Similarity', fontsize=12)
    ax1.set_title('Gradient-Hessian Eigenvector Alignment During Grokking (50k steps)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=1, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axhline(y=-1, color='purple', linestyle='--', alpha=0.3, linewidth=1)
    
    # Highlight negative regions if any
    cos_array = np.array(cos_sims)
    negative_mask = cos_array < 0
    if np.any(negative_mask):
        negative_steps = np.array(steps_logged)[negative_mask]
        negative_cos = cos_array[negative_mask]
        ax1.scatter(negative_steps, negative_cos, c='blue', s=50, alpha=0.8, 
                   marker='o', label=f'Negative ({len(negative_cos)} points)')
    
    # Second plot: accuracies
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(steps_logged, train_accs, 'g-', linewidth=2, alpha=0.8, label='Train')
    ax2.plot(steps_logged, test_accs, 'b-', linewidth=2, alpha=0.8, label='Test')
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Progress', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)
    
    # Third plot: overlay
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    
    # Plot test accuracy
    ax3.plot(steps_logged, test_accs, 'b-', linewidth=2, alpha=0.8)
    ax3.set_ylabel('Test Accuracy', color='b', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.set_ylim(-0.05, 1.05)
    
    # Plot cosine similarity on twin axis
    ax3_twin = ax3.twinx()
    ax3_twin.plot(steps_logged, cos_sims, 'r-', linewidth=2, alpha=0.8)
    ax3_twin.set_ylabel('cos(grad, H@grad)', color='r', fontsize=12)
    ax3_twin.tick_params(axis='y', labelcolor='r')
    ax3_twin.set_ylim(-1.1, 1.1)
    ax3_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    
    ax3.set_xlabel('Training Steps', fontsize=12)
    ax3.set_title('Eigenvector Alignment vs Grokking Progress', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Mark grokking phases
    test_array = np.array(test_accs)
    if len(test_array) > 0:
        # Mark when test accuracy starts increasing
        if np.any(test_array > 0.1):
            grok_start_idx = np.where(test_array > 0.1)[0][0]
            for ax in [ax1, ax2, ax3]:
                ax.axvline(x=steps_logged[grok_start_idx], color='orange', 
                          linestyle='--', alpha=0.7, linewidth=2)
            ax1.text(steps_logged[grok_start_idx], 0.5, 'Grokking starts', 
                    rotation=90, va='bottom', fontsize=10)
        
        # Mark when test accuracy reaches high value
        if np.any(test_array > 0.9):
            grok_end_idx = np.where(test_array > 0.9)[0][0]
            for ax in [ax1, ax2, ax3]:
                ax.axvline(x=steps_logged[grok_end_idx], color='green', 
                          linestyle='--', alpha=0.7, linewidth=2)
            ax1.text(steps_logged[grok_end_idx], 0.5, 'Generalization', 
                    rotation=90, va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('grokking_cosine_similarity_50k.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print(f"\n{'='*60}")
    print("FULL GROKKING ANALYSIS (50k steps)")
    print(f"{'='*60}")
    
    print(f"\nTotal measurements: {len(cos_sims)}")
    print(f"Cosine similarity range: [{min(cos_sims):.4f}, {max(cos_sims):.4f}]")
    
    # Check for negative values
    negative_values = [c for c in cos_sims if c < 0]
    if negative_values:
        print(f"\n⚠️  Found {len(negative_values)} NEGATIVE cosine similarities!")
        print(f"   Most negative: {min(negative_values):.4f}")
        print(f"   Negative fraction: {len(negative_values)/len(cos_sims):.1%}")
        neg_indices = [i for i, c in enumerate(cos_sims) if c < 0]
        print(f"   First negative at step: {steps_logged[neg_indices[0]]}")
    else:
        print(f"\n✓ No negative values found - gradient never anti-aligned with H@grad")
    
    # Analyze by grokking phase
    if len(test_array) > 0 and np.any(test_array > 0.9):
        # Pre-grokking (test acc < 0.1)
        pre_grok_mask = np.array(test_accs) < 0.1
        if np.any(pre_grok_mask):
            pre_grok_cos = np.array(cos_sims)[pre_grok_mask]
            print(f"\nPre-grokking phase (test acc < 0.1):")
            print(f"  Steps: 0 - {steps_logged[np.where(pre_grok_mask)[0][-1]]}")
            print(f"  cos(g,Hg) = {pre_grok_cos.mean():.3f} ± {pre_grok_cos.std():.3f}")
            print(f"  Range: [{pre_grok_cos.min():.3f}, {pre_grok_cos.max():.3f}]")
        
        # During grokking (0.1 <= test acc <= 0.9)
        during_mask = (np.array(test_accs) >= 0.1) & (np.array(test_accs) <= 0.9)
        if np.any(during_mask):
            during_cos = np.array(cos_sims)[during_mask]
            during_steps = np.array(steps_logged)[during_mask]
            print(f"\nDuring grokking (0.1 ≤ test acc ≤ 0.9):")
            print(f"  Steps: {during_steps[0]} - {during_steps[-1]}")
            print(f"  cos(g,Hg) = {during_cos.mean():.3f} ± {during_cos.std():.3f}")
            print(f"  Range: [{during_cos.min():.3f}, {during_cos.max():.3f}]")
        
        # Post-grokking (test acc > 0.9)
        post_mask = np.array(test_accs) > 0.9
        if np.any(post_mask):
            post_cos = np.array(cos_sims)[post_mask]
            post_steps = np.array(steps_logged)[post_mask]
            print(f"\nPost-grokking phase (test acc > 0.9):")
            print(f"  Steps: {post_steps[0]} - {post_steps[-1]}")
            print(f"  cos(g,Hg) = {post_cos.mean():.3f} ± {post_cos.std():.3f}")
            print(f"  Range: [{post_cos.min():.3f}, {post_cos.max():.3f}]")
    
    print(f"\nFinal values:")
    print(f"  cos(grad, H@grad) = {cos_sims[-1]:.3f}")
    print(f"  Test accuracy = {test_accs[-1]:.3f}")


if __name__ == "__main__":
    main()