import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from simple_model import SimpleModularAdditionMLP
from data import generate_modular_addition_data, create_data_loader


def check_gradient_norms_around_drop():
    """Quick check of gradient norms around the cosine drop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    
    # Setup (same as original)
    p = 97
    model = SimpleModularAdditionMLP(p=p, hidden_size=256).to(device)
    
    # Data
    train_data, test_data = generate_modular_addition_data(p=p, train_frac=0.3, seed=42)
    train_loader = create_data_loader(train_data, batch_size=512, shuffle=True)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    
    target_steps = [18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000]
    results = {}
    
    print("Training and checking gradient norms at key steps...")
    
    step = 0
    for inputs, labels in train_loader:
        if step > max(target_steps):
            break
            
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Standard training
        model.train()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check if this is one of our target steps
        if step in target_steps:
            print(f"Checking step {step}...")
            
            # Compute gradient norm and cosine similarity
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
            
            # Test accuracy
            model.eval()
            with torch.no_grad():
                test_correct = 0
                test_total = 0
                for test_inputs, test_labels in test_loader:
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                    test_outputs = model(test_inputs)
                    test_correct += (test_outputs.argmax(dim=1) == test_labels).sum().item()
                    test_total += test_labels.size(0)
                test_acc = test_correct / test_total
            
            results[step] = {
                'grad_norm': grad_norm,
                'cos_sim': cos_sim,
                'test_acc': test_acc,
                'loss': loss.item()
            }
            
            print(f"  grad_norm={grad_norm:.2e}, cos_sim={cos_sim:.3f if cos_sim else 'N/A'}, test_acc={test_acc:.3f}")
        
        step += 1
    
    return results


def analyze_results(results):
    """Analyze the gradient norm results."""
    print(f"\n{'='*80}")
    print("GRADIENT NORM ANALYSIS AROUND COSINE DROP")
    print(f"{'='*80}")
    
    steps = sorted(results.keys())
    
    print(f"\n📊 DETAILED RESULTS:")
    print(f"{'Step':<8} {'Test Acc':<10} {'Loss':<12} {'Grad Norm':<12} {'cos(g,Hg)':<12}")
    print("-" * 70)
    
    for step in steps:
        data = results[step]
        cos_str = f"{data['cos_sim']:.3f}" if data['cos_sim'] is not None else "N/A"
        print(f"{step:<8} {data['test_acc']:<10.3f} {data['loss']:<12.2e} {data['grad_norm']:<12.2e} {cos_str:<12}")
    
    # Find the transition
    cos_values = [results[s]['cos_sim'] for s in steps if results[s]['cos_sim'] is not None]
    grad_values = [results[s]['grad_norm'] for s in steps]
    
    if len(cos_values) >= 2:
        # Find where cosine drops significantly
        drop_indices = []
        for i in range(len(cos_values) - 1):
            if cos_values[i] > 0.7 and cos_values[i+1] < 0.3:
                drop_indices.append(i)
        
        if drop_indices:
            drop_idx = drop_indices[0]
            before_step = steps[drop_idx]
            after_step = steps[drop_idx + 1]
            
            grad_before = results[before_step]['grad_norm']
            grad_after = results[after_step]['grad_norm']
            cos_before = results[before_step]['cos_sim']
            cos_after = results[after_step]['cos_sim']
            
            print(f"\n🔍 TRANSITION ANALYSIS:")
            print(f"Cosine drop detected between steps {before_step} and {after_step}")
            print(f"  Gradient norm: {grad_before:.2e} → {grad_after:.2e} ({grad_after/grad_before:.2f}x)")
            print(f"  cos(grad,H@grad): {cos_before:.3f} → {cos_after:.3f}")
            
            print(f"\n💡 INTERPRETATION:")
            if grad_after/grad_before < 0.1:
                print("❌ GRADIENT VANISHING explains the cosine drop!")
                print("   The dramatic decrease in gradient norm makes the cosine metric unreliable.")
                print("   This is NOT evidence of cleanup phase - just numerical instability.")
            elif grad_after/grad_before > 0.5:
                print("✅ GRADIENT NORM STABLE - cosine drop is meaningful!")
                print("   The gradient norm remains substantial, so the cosine drop likely")
                print("   reflects real changes in the optimization landscape (cleanup phase).")
            else:
                print("⚠️  MIXED EVIDENCE - partial gradient vanishing")
                print("   Some gradient decrease, but cosine drop might still reflect real changes.")
        else:
            print("⚠️  No clear cosine drop detected in this range")
    
    # Create visualization
    create_visualization(results)


def create_visualization(results):
    """Create a visualization of the results."""
    steps = sorted(results.keys())
    grad_norms = [results[s]['grad_norm'] for s in steps]
    cos_sims = [results[s]['cos_sim'] for s in steps if results[s]['cos_sim'] is not None]
    cos_steps = [s for s in steps if results[s]['cos_sim'] is not None]
    test_accs = [results[s]['test_acc'] for s in steps]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot 1: Test accuracy
    axes[0].plot(steps, test_accs, 'b-o', linewidth=2, markersize=6)
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Gradient Norm Hypothesis Test: Key Steps Around Cosine Drop')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)
    
    # Plot 2: Cosine similarity
    if cos_sims:
        axes[1].plot(cos_steps, cos_sims, 'r-o', linewidth=2, markersize=6)
        axes[1].set_ylabel('cos(grad, H@grad)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-0.1, 1.1)
    
    # Plot 3: Gradient norm
    axes[2].plot(steps, grad_norms, 'orange', linewidth=2, marker='o', markersize=6)
    axes[2].set_ylabel('Gradient Norm')
    axes[2].set_xlabel('Training Steps')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_norm_hypothesis_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📁 Saved: gradient_norm_hypothesis_test.png")


if __name__ == "__main__":
    print("Running focused gradient norm check around cosine drop...")
    results = check_gradient_norms_around_drop()
    analyze_results(results)