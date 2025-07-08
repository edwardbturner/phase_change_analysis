import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simple_model import SimpleModularAdditionMLP
from data import generate_modular_addition_data, create_data_loader


def compute_cosine_similarity(grad, Hgrad):
    """Fast cosine similarity computation."""
    dot = (grad * Hgrad).sum()
    norm_g = torch.norm(grad)
    norm_Hg = torch.norm(Hgrad)
    
    if norm_g > 0 and norm_Hg > 0:
        return (dot / (norm_g * norm_Hg)).item()
    return 0.0


def main():
    """Analyze cos(grad, H@grad) throughout grokking."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    
    # Simple setup for speed
    p = 97
    model = SimpleModularAdditionMLP(p=p, hidden_size=128).to(device)
    
    # Data
    train_data, test_data = generate_modular_addition_data(p=p, train_frac=0.3, seed=42)
    train_loader = create_data_loader(train_data, batch_size=512, shuffle=True)
    test_loader = create_data_loader(test_data, batch_size=1024, shuffle=False)
    
    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    
    # Storage
    cos_sims = []
    test_accs = []
    steps = []
    
    print("Running grokking experiment with eigenvector analysis...")
    
    step = 0
    with tqdm(total=30000) as pbar:
        for epoch in range(100):
            for inputs, labels in train_loader:
                if step > 30000:
                    break
                    
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Standard training step
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Every 500 steps, compute eigenvector alignment
                if step % 500 == 0:
                    # Get current parameters as a single vector
                    params_vec = torch.cat([p.data.view(-1) for p in model.parameters()])
                    grad_vec = torch.cat([p.grad.view(-1) for p in model.parameters()])
                    
                    # Finite difference approximation of H@grad
                    eps = 1e-4
                    
                    # Perturb parameters
                    model_perturbed = SimpleModularAdditionMLP(p=p, hidden_size=128).to(device)
                    model_perturbed.load_state_dict(model.state_dict())
                    
                    # Add eps * grad to parameters
                    idx = 0
                    for p in model_perturbed.parameters():
                        numel = p.numel()
                        p.data += eps * grad_vec[idx:idx+numel].view(p.shape)
                        idx += numel
                    
                    # Compute gradient at perturbed point
                    outputs_pert = model_perturbed(inputs)
                    loss_pert = criterion(outputs_pert, labels)
                    
                    optimizer.zero_grad()
                    loss_pert.backward()
                    
                    grad_vec_pert = torch.cat([p.grad.view(-1) for p in model_perturbed.parameters()])
                    
                    # Approximate H@grad
                    Hgrad_approx = (grad_vec_pert - grad_vec) / eps
                    
                    # Compute cosine similarity
                    cos_sim = compute_cosine_similarity(grad_vec, Hgrad_approx)
                    
                    # Test accuracy
                    model.eval()
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for test_inputs, test_labels in test_loader:
                            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                            test_outputs = model(test_inputs)
                            correct += (test_outputs.argmax(dim=1) == test_labels).sum().item()
                            total += test_labels.size(0)
                        test_acc = correct / total
                    model.train()
                    
                    cos_sims.append(cos_sim)
                    test_accs.append(test_acc)
                    steps.append(step)
                    
                    pbar.set_postfix({'test_acc': f'{test_acc:.3f}', 'cos_sim': f'{cos_sim:+.3f}'})
                
                # Actual optimizer step
                optimizer.step()
                step += 1
                pbar.update(1)
                
            if step > 30000:
                break
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Test accuracy
    ax1.plot(steps, test_accs, 'b-', linewidth=2)
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Grokking Progress')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Cosine similarity
    ax2.plot(steps, cos_sims, 'r-', linewidth=2, alpha=0.8)
    ax2.scatter(steps, cos_sims, c='red', s=30, alpha=0.6)
    ax2.set_ylabel('cos(grad, H@grad)')
    ax2.set_title('Gradient-Hessian Eigenvector Alignment (Raw Values)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.3, label='Eigenvector')
    ax2.axhline(y=-1, color='purple', linestyle='--', alpha=0.3, label='Negative eigenvector')
    
    # Both on same plot with different scales
    ax3.plot(steps, test_accs, 'b-', linewidth=2, label='Test Acc')
    ax3.set_ylabel('Test Accuracy', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.set_ylim(-0.05, 1.05)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(steps, cos_sims, 'r-', linewidth=2, alpha=0.8, label='cos(g,Hg)')
    ax3_twin.set_ylabel('cos(grad, H@grad)', color='r')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    ax3_twin.set_ylim(-1.1, 1.1)
    ax3_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    
    ax3.set_xlabel('Training Steps')
    ax3.set_title('Eigenvector Alignment vs Grokking')
    ax3.grid(True, alpha=0.3)
    
    # Mark grokking phases
    test_accs_arr = np.array(test_accs)
    if np.any(test_accs_arr > 0.1):
        grok_start_idx = np.where(test_accs_arr > 0.1)[0][0]
        ax3.axvline(x=steps[grok_start_idx], color='orange', linestyle='--', alpha=0.5)
        ax3.text(steps[grok_start_idx], 0.5, 'Grokking', rotation=90, va='bottom')
    
    plt.tight_layout()
    plt.savefig('eigenvector_evolution.png', dpi=150)
    plt.show()
    
    # Analysis
    print(f"\n=== Results ===")
    print(f"Cosine similarity range: [{min(cos_sims):.3f}, {max(cos_sims):.3f}]")
    print(f"Final cos(grad, H@grad): {cos_sims[-1]:.3f}")
    
    # Check for negative values
    negative_cos = [c for c in cos_sims if c < 0]
    if negative_cos:
        print(f"\nFound {len(negative_cos)} negative cosine similarities!")
        print(f"Most negative: {min(negative_cos):.3f}")
    else:
        print(f"\nNo negative cosine similarities found - gradient never anti-aligned with H@grad")
    
    # Analyze by phase
    if np.any(test_accs_arr > 0.9):
        pre_grok = [cos_sims[i] for i, acc in enumerate(test_accs) if acc < 0.1]
        post_grok = [cos_sims[i] for i, acc in enumerate(test_accs) if acc > 0.9]
        
        if pre_grok:
            print(f"\nPre-grokking: cos(g,Hg) = {np.mean(pre_grok):.3f} ± {np.std(pre_grok):.3f}")
        if post_grok:
            print(f"Post-grokking: cos(g,Hg) = {np.mean(post_grok):.3f} ± {np.std(post_grok):.3f}")


if __name__ == "__main__":
    main()