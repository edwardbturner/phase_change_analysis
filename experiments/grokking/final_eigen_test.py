import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from simple_model import SimpleModularAdditionMLP
from data import generate_modular_addition_data, create_data_loader


def compute_hvp(loss, params, vector):
    """Compute Hessian-vector product H @ v correctly."""
    # Get gradients
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    
    # Compute dot product of gradient with vector
    dot_product = sum((g * v).sum() for g, v in zip(grads, vector))
    
    # Take gradient of dot product to get H @ v
    hvp = torch.autograd.grad(dot_product, params, retain_graph=True)
    return hvp


def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two lists of tensors."""
    # Flatten and concatenate
    v1_flat = torch.cat([t.flatten() for t in v1])
    v2_flat = torch.cat([t.flatten() for t in v2])
    
    # Compute cosine similarity
    dot = (v1_flat * v2_flat).sum()
    norm1 = torch.norm(v1_flat)
    norm2 = torch.norm(v2_flat)
    
    if norm1 > 0 and norm2 > 0:
        return (dot / (norm1 * norm2)).item()
    return 0.0


def main():
    """Test gradient eigenvector alignment during grokking."""
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
    results = {
        'steps': [],
        'cos_sims': [],
        'train_accs': [],
        'test_accs': [],
        'eigenvalues': []
    }
    
    print("Starting training with eigenvector analysis...")
    print("Computing cos(grad, H@grad) to see if gradient is eigenvector of Hessian\n")
    
    step = 0
    for epoch in range(50):  # Run for more epochs
        for inputs, labels in train_loader:
            if step > 30000:  # Stop after 30000 steps
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Analyze eigenvector alignment every 200 steps (more manageable for longer run)
            if step % 200 == 0:
                # Recompute for clean computation graph
                model.eval()
                with torch.no_grad():
                    outputs_clean = model(inputs)
                    loss_clean = criterion(outputs_clean, labels)
                
                model.train()
                outputs_analysis = model(inputs)
                loss_analysis = criterion(outputs_analysis, labels)
                
                try:
                    # Get gradient
                    params = list(model.parameters())
                    grads = torch.autograd.grad(loss_analysis, params, create_graph=True, retain_graph=True)
                    
                    # Compute H @ grad
                    Hg = compute_hvp(loss_analysis, params, grads)
                    
                    # Compute cosine similarity
                    cos_sim = compute_cosine_similarity(grads, Hg)
                    
                    # Compute eigenvalue estimate (Rayleigh quotient)
                    grad_flat = torch.cat([g.flatten() for g in grads])
                    Hg_flat = torch.cat([h.flatten() for h in Hg])
                    eigenvalue = ((grad_flat * Hg_flat).sum() / (grad_flat * grad_flat).sum()).item()
                    
                    # Compute accuracies
                    model.eval()
                    with torch.no_grad():
                        # Train accuracy
                        train_correct = (outputs.argmax(dim=1) == labels).float().mean().item()
                        
                        # Test accuracy (sample)
                        test_inputs, test_labels = next(iter(test_loader))
                        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                        test_outputs = model(test_inputs)
                        test_correct = (test_outputs.argmax(dim=1) == test_labels).float().mean().item()
                    model.train()
                    
                    # Log results
                    results['steps'].append(step)
                    results['cos_sims'].append(cos_sim)
                    results['train_accs'].append(train_correct)
                    results['test_accs'].append(test_correct)
                    results['eigenvalues'].append(eigenvalue)
                    
                    print(f"Step {step:4d}: cos(g,Hg)={cos_sim:+.3f}, λ≈{eigenvalue:+.2e}, "
                          f"train_acc={train_correct:.3f}, test_acc={test_correct:.3f}")
                    
                except Exception as e:
                    print(f"Step {step}: Analysis failed - {str(e)[:50]}")
            
            # Optimizer step
            optimizer.step()
            step += 1
            
        if step > 30000:
            break
    
    # Plot results
    if results['steps']:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracies
        ax = axes[0, 0]
        ax.plot(results['steps'], results['train_accs'], label='Train', alpha=0.8)
        ax.plot(results['steps'], results['test_accs'], label='Test', alpha=0.8)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cosine similarity
        ax = axes[0, 1]
        ax.plot(results['steps'], results['cos_sims'], 'r-', linewidth=2, alpha=0.8)
        ax.scatter(results['steps'], results['cos_sims'], c='red', s=20, alpha=0.6)
        ax.set_xlabel('Steps')
        ax.set_ylabel('cos(grad, H@grad)')
        ax.set_title('Gradient Eigenvector Alignment (Raw Cosine Similarity)')
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.3, label='Perfect alignment')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, label='Orthogonal')
        ax.axhline(y=-1, color='black', linestyle='--', alpha=0.3, label='Perfect anti-alignment')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        
        # Eigenvalues
        ax = axes[1, 0]
        ax.plot(results['steps'], results['eigenvalues'], 'g-', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Eigenvalue estimate')
        ax.set_title('Rayleigh Quotient (g^T H g / ||g||²)')
        ax.set_yscale('symlog')
        ax.grid(True, alpha=0.3)
        
        # Combined view
        ax = axes[1, 1]
        ax2 = ax.twinx()
        
        # Plot test accuracy
        l1 = ax.plot(results['steps'], results['test_accs'], 'b-', label='Test Acc', alpha=0.8)
        ax.set_ylabel('Test Accuracy', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot cosine similarity
        l2 = ax2.plot(results['steps'], results['cos_sims'], 'r-', label='cos(g,Hg)', alpha=0.8)
        ax2.set_ylabel('cos(grad, H@grad)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(-1.1, 1.1)
        
        ax.set_xlabel('Steps')
        ax.set_title('Eigenvector Alignment vs Test Accuracy')
        ax.grid(True, alpha=0.3)
        
        # Add phase annotations based on test accuracy
        test_accs = np.array(results['test_accs'])
        if np.any(test_accs > 0.1):
            grokking_start = results['steps'][np.where(test_accs > 0.1)[0][0]]
            ax.axvline(x=grokking_start, color='orange', linestyle='--', alpha=0.5)
            ax.text(grokking_start, 0.5, 'Grokking starts', rotation=90, verticalalignment='bottom')
        
        plt.tight_layout()
        plt.savefig('eigenvector_alignment_analysis.png', dpi=150)
        plt.show()
        
        # Summary statistics
        print("\n=== Summary ===")
        cos_sims = np.array(results['cos_sims'])
        print(f"Average cos(grad, H@grad): {np.mean(cos_sims):.3f}")
        print(f"Final cos(grad, H@grad): {cos_sims[-1]:.3f}")
        
        # Check correlation with grokking
        if np.any(test_accs > 0.9):
            pre_grok = cos_sims[test_accs < 0.1]
            post_grok = cos_sims[test_accs > 0.9]
            if len(pre_grok) > 0 and len(post_grok) > 0:
                print(f"\nPre-grokking avg cos(g,Hg): {np.mean(pre_grok):.3f}")
                print(f"Post-grokking avg cos(g,Hg): {np.mean(post_grok):.3f}")


if __name__ == "__main__":
    main()