import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from simple_model import SimpleModularAdditionMLP
from data import generate_modular_addition_data, create_data_loader


def minimal_gradient_test():
    """Minimal test to check gradient norms around step 20k."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    
    p = 97
    model = SimpleModularAdditionMLP(p=p, hidden_size=256).to(device)
    
    train_data, test_data = generate_modular_addition_data(p=p, train_frac=0.3, seed=42)
    train_loader = create_data_loader(train_data, batch_size=512, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    
    # Sample points around the expected drop
    sample_points = [19000, 19500, 20000, 20500, 21000, 21500, 22000]
    results = {}
    
    print("Training to sample gradient norms around step 20k...")
    
    step = 0
    pbar = tqdm(total=max(sample_points), desc='Training')
    
    while step <= max(sample_points):
        for inputs, labels in train_loader:
            if step > max(sample_points):
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Training step
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Sample at key points
            if step in sample_points:
                # Compute gradient norm
                model.train()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                params = list(model.parameters())
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
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
                
                results[step] = {
                    'grad_norm': grad_norm,
                    'cos_sim': cos_sim,
                    'loss': loss.item()
                }
                
                pbar.set_postfix({
                    'step': step,
                    'grad_norm': f'{grad_norm:.2e}',
                    'cos_sim': f'{cos_sim:.3f}' if cos_sim else 'N/A'
                })
            
            step += 1
            pbar.update(1)
    
    pbar.close()
    return results


def analyze_minimal_results(results):
    """Analyze the minimal results."""
    print(f"\n{'='*60}")
    print("GRADIENT NORM TEST RESULTS")
    print(f"{'='*60}")
    
    steps = sorted(results.keys())
    
    print(f"\n📊 RESULTS:")
    print(f"{'Step':<8} {'Grad Norm':<12} {'cos(g,Hg)':<12} {'Loss':<12}")
    print("-" * 50)
    
    for step in steps:
        data = results[step]
        cos_str = f"{data['cos_sim']:.3f}" if data['cos_sim'] is not None else "N/A"
        print(f"{step:<8} {data['grad_norm']:<12.2e} {cos_str:<12} {data['loss']:<12.2e}")
    
    # Create simple plot
    grad_norms = [results[s]['grad_norm'] for s in steps]
    cos_sims = [results[s]['cos_sim'] for s in steps if results[s]['cos_sim'] is not None]
    cos_steps = [s for s in steps if results[s]['cos_sim'] is not None]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Gradient norm
    ax1.plot(steps, grad_norms, 'orange', marker='o', linewidth=2, markersize=8)
    ax1.set_ylabel('Gradient Norm')
    ax1.set_title('Gradient Norm Around Step 20k')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Cosine similarity
    if cos_sims:
        ax2.plot(cos_steps, cos_sims, 'red', marker='o', linewidth=2, markersize=8)
        ax2.set_ylabel('cos(grad, H@grad)')
        ax2.set_xlabel('Training Steps')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('minimal_gradient_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📁 Saved: minimal_gradient_test.png")
    
    # Analysis
    if len(results) >= 2:
        first_step = min(steps)
        last_step = max(steps)
        
        grad_first = results[first_step]['grad_norm']
        grad_last = results[last_step]['grad_norm']
        
        cos_first = results[first_step]['cos_sim']
        cos_last = results[last_step]['cos_sim']
        
        print(f"\n🔍 ANALYSIS:")
        print(f"From step {first_step} to {last_step}:")
        print(f"  Gradient norm: {grad_first:.2e} → {grad_last:.2e} ({grad_last/grad_first:.2f}x)")
        
        if cos_first and cos_last:
            print(f"  cos(grad,H@grad): {cos_first:.3f} → {cos_last:.3f}")
            
            print(f"\n💡 CONCLUSION:")
            if grad_last/grad_first < 0.1:
                print("❌ GRADIENT VANISHING explains the cosine drop!")
            elif grad_last/grad_first > 0.5:
                print("✅ GRADIENT NORM STABLE - cosine drop is meaningful!")
            else:
                print("⚠️  MIXED EVIDENCE")


if __name__ == "__main__":
    results = minimal_gradient_test()
    analyze_minimal_results(results)