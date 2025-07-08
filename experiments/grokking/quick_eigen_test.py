import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from simple_model import SimpleModularAdditionMLP
from data import generate_modular_addition_data, create_data_loader


def compute_cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    v1_flat = torch.cat([t.flatten() for t in v1])
    v2_flat = torch.cat([t.flatten() for t in v2])
    
    cos_sim = torch.nn.functional.cosine_similarity(v1_flat.unsqueeze(0), v2_flat.unsqueeze(0))
    return cos_sim.item()


def analyze_eigenvector_alignment():
    """Quick test of gradient-Hessian eigenvector alignment during training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simple setup
    p = 97
    model = SimpleModularAdditionMLP(p=p, hidden_size=128).to(device)
    
    # Data
    train_data, _ = generate_modular_addition_data(p=p, train_frac=0.3, seed=42)
    train_loader = create_data_loader(train_data, batch_size=512, shuffle=True)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    
    cos_sims = []
    accuracies = []
    
    print("Training and computing cos(grad, H@grad)...")
    
    # Train for a few steps
    for step, (inputs, labels) in enumerate(train_loader):
        if step >= 100:  # Just 100 steps for quick test
            break
            
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Get gradient
        optimizer.zero_grad()
        loss.backward()
        
        # Every 10 steps, compute eigenvector alignment
        if step % 10 == 0:
            # Recompute for clean graph
            outputs2 = model(inputs)
            loss2 = criterion(outputs2, labels)
            
            # Get gradients
            grads = torch.autograd.grad(loss2, model.parameters(), create_graph=True, retain_graph=True)
            
            try:
                # Compute Hessian-vector product H @ grad
                grad_list = list(grads)
                # Compute g^T @ g (this is wrong - we need grad of (grad dot v))
                grad_norm_sq = sum((g * g).sum() for g in grad_list)
                
                # Correct way: compute grad of (sum of grad[i] * v[i])
                grad_dot_v = sum((g1 * g2).sum() for g1, g2 in zip(grad_list, grad_list))
                Hg = torch.autograd.grad(grad_dot_v, model.parameters(), retain_graph=True)
                
                # Compute cosine similarity
                cos_sim = compute_cosine_similarity(grad_list, list(Hg))
                cos_sims.append(cos_sim)
                
                # Compute accuracy
                with torch.no_grad():
                    acc = (outputs.argmax(dim=1) == labels).float().mean().item()
                    accuracies.append(acc)
                
                print(f"Step {step}: cos(grad, H@grad) = {cos_sim:.3f}, acc = {acc:.3f}")
            except Exception as e:
                print(f"Step {step}: Failed - {str(e)[:50]}")
        
        # Optimizer step
        optimizer.step()
    
    # Plot results
    if cos_sims:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(cos_sims, 'o-')
        plt.xlabel('Measurement (every 10 steps)')
        plt.ylabel('cos(grad, H@grad)')
        plt.title('Gradient-Hessian Eigenvector Alignment')
        plt.ylim(-1.1, 1.1)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(accuracies, 'o-')
        plt.xlabel('Measurement (every 10 steps)')
        plt.ylabel('Training Accuracy')
        plt.title('Training Progress')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('quick_eigen_test.png', dpi=150)
        plt.show()
        
        print(f"\nAverage cos(grad, H@grad) = {np.mean(cos_sims):.3f}")
        print(f"Final cos(grad, H@grad) = {cos_sims[-1]:.3f}")
    else:
        print("No successful measurements")


if __name__ == "__main__":
    analyze_eigenvector_alignment()