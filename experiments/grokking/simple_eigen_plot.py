import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simple_model import SimpleModularAdditionMLP
from data import generate_modular_addition_data, create_data_loader


def compute_eigen_alignment(model, inputs, labels, criterion):
    """Compute cos(grad, H@grad) using autograd."""
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    
    # Get parameters and gradients
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    
    # Compute dot product grad^T @ grad
    grad_dot_grad = sum((g * g).sum() for g in grads)
    
    # Take gradient to get H @ grad
    try:
        Hgrad = torch.autograd.grad(grad_dot_grad, params, retain_graph=True)
        
        # Compute cosine similarity
        grad_flat = torch.cat([g.flatten() for g in grads])
        Hgrad_flat = torch.cat([h.flatten() for h in Hgrad])
        
        cos_sim = torch.nn.functional.cosine_similarity(
            grad_flat.unsqueeze(0), 
            Hgrad_flat.unsqueeze(0)
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
    
    print("Tracking cos(grad, H@grad) during grokking...")
    
    step = 0
    for epoch in tqdm(range(50), desc='Epochs'):
        for inputs, labels in train_loader:
            if step > 30000:
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
                
                cos_sim = compute_eigen_alignment(model, batch_inputs, batch_labels, criterion)
                
                if cos_sim is not None:
                    # Compute accuracies
                    model.eval()
                    with torch.no_grad():
                        # Train accuracy
                        train_pred = outputs.argmax(dim=1)
                        train_acc = (train_pred == labels).float().mean().item()
                        
                        # Test accuracy
                        test_correct = 0
                        test_total = 0
                        for test_batch in test_loader:
                            test_inputs, test_labels = test_batch
                            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                            test_outputs = model(test_inputs)
                            test_correct += (test_outputs.argmax(dim=1) == test_labels).sum().item()
                            test_total += test_labels.size(0)
                        test_acc = test_correct / test_total
                    
                    cos_sims.append(cos_sim)
                    test_accs.append(test_acc)
                    train_accs.append(train_acc)
                    steps_logged.append(step)
                    
                    if step % 1000 == 0:
                        print(f"Step {step}: cos={cos_sim:+.3f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")
            
            step += 1
            
        if step > 30000:
            break
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(12, 10))
    
    # Main plot: cosine similarity over time
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(steps_logged, cos_sims, 'r-', linewidth=2, alpha=0.8, label='cos(grad, H@grad)')
    ax1.scatter(steps_logged, cos_sims, c='red', s=20, alpha=0.5)
    ax1.set_ylabel('Cosine Similarity', fontsize=12)
    ax1.set_title('Gradient-Hessian Eigenvector Alignment During Training', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axhline(y=1, color='green', linestyle='--', alpha=0.3, linewidth=1, label='Perfect alignment')
    ax1.axhline(y=-1, color='purple', linestyle='--', alpha=0.3, linewidth=1, label='Perfect anti-alignment')
    ax1.legend()
    
    # Highlight negative regions if any
    cos_array = np.array(cos_sims)
    negative_mask = cos_array < 0
    if np.any(negative_mask):
        negative_steps = np.array(steps_logged)[negative_mask]
        negative_cos = cos_array[negative_mask]
        ax1.scatter(negative_steps, negative_cos, c='blue', s=50, alpha=0.8, 
                   edgecolors='blue', linewidth=2, label='Negative values')
    
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
    
    # Mark grokking transition
    test_array = np.array(test_accs)
    if np.any(test_array > 0.1) and np.any(test_array < 0.9):
        grok_start = steps_logged[np.where(test_array > 0.1)[0][0]]
        ax3.axvline(x=grok_start, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        ax3.text(grok_start, 0.5, 'Grokking starts', rotation=90, va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('cosine_similarity_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print(f"\n{'='*50}")
    print("ANALYSIS RESULTS")
    print(f"{'='*50}")
    
    print(f"\nTotal measurements: {len(cos_sims)}")
    print(f"Cosine similarity range: [{min(cos_sims):.4f}, {max(cos_sims):.4f}]")
    
    # Check for negative values
    negative_values = [c for c in cos_sims if c < 0]
    if negative_values:
        print(f"\n⚠️  Found {len(negative_values)} NEGATIVE cosine similarities!")
        print(f"   Most negative: {min(negative_values):.4f}")
        print(f"   Negative fraction: {len(negative_values)/len(cos_sims):.1%}")
    else:
        print(f"\n✓ No negative values found - gradient never anti-aligned with H@grad")
    
    # Analyze phases
    print(f"\nPhase Analysis:")
    
    # Early training (first 20%)
    early_idx = int(len(cos_sims) * 0.2)
    print(f"Early training (first {early_idx} measurements):")
    print(f"  cos(g,Hg) = {np.mean(cos_sims[:early_idx]):.3f} ± {np.std(cos_sims[:early_idx]):.3f}")
    
    # Late training (last 20%)
    late_idx = int(len(cos_sims) * 0.8)
    print(f"Late training (last {len(cos_sims)-late_idx} measurements):")
    print(f"  cos(g,Hg) = {np.mean(cos_sims[late_idx:]):.3f} ± {np.std(cos_sims[late_idx:]):.3f}")
    
    # Grokking phase
    if np.any(test_array > 0.9):
        pre_grok = [cos_sims[i] for i, acc in enumerate(test_accs) if acc < 0.1]
        during_grok = [cos_sims[i] for i, acc in enumerate(test_accs) if 0.1 <= acc <= 0.9]
        post_grok = [cos_sims[i] for i, acc in enumerate(test_accs) if acc > 0.9]
        
        if pre_grok:
            print(f"\nPre-grokking (test acc < 0.1):")
            print(f"  cos(g,Hg) = {np.mean(pre_grok):.3f} ± {np.std(pre_grok):.3f}")
        if during_grok:
            print(f"During grokking (0.1 ≤ test acc ≤ 0.9):")
            print(f"  cos(g,Hg) = {np.mean(during_grok):.3f} ± {np.std(during_grok):.3f}")
        if post_grok:
            print(f"Post-grokking (test acc > 0.9):")
            print(f"  cos(g,Hg) = {np.mean(post_grok):.3f} ± {np.std(post_grok):.3f}")


if __name__ == "__main__":
    main()