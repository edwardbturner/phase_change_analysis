import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
from simple_model import SimpleModularAdditionMLP
from data import generate_modular_addition_data, create_data_loader


def compute_gradient_norm_at_steps(steps_to_check=[18000, 20000, 22000, 24000]):
    """Check gradient norms at key steps around the cosine drop."""
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
    
    results = {}
    
    print("Training to key steps and measuring gradient norms...")
    
    step = 0
    for inputs, labels in train_loader:
        if step >= max(steps_to_check):
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
        if step in steps_to_check:
            # Compute gradient norm
            model.train()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            params = list(model.parameters())
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            
            grad_norm = torch.sqrt(sum((g * g).sum() for g in grads)).item()
            
            # Also compute cosine similarity for comparison
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
            
            print(f"Step {step}: grad_norm={grad_norm:.2e}, cos_sim={cos_sim:.3f if cos_sim else 'N/A'}")
        
        step += 1
    
    return results


def analyze_gradient_hypothesis():
    """Analyze whether gradient norm explains the cosine drop."""
    print("="*70)
    print("GRADIENT NORM HYPOTHESIS TEST")
    print("="*70)
    
    # Key steps: before, during, and after the cosine drop
    steps_to_check = [18000, 20000, 22000, 24000]
    
    results = compute_gradient_norm_at_steps(steps_to_check)
    
    print(f"\n📊 RESULTS:")
    print(f"{'Step':<8} {'Grad Norm':<12} {'cos(g,Hg)':<12} {'Loss':<12}")
    print("-" * 50)
    
    for step in sorted(results.keys()):
        data = results[step]
        cos_str = f"{data['cos_sim']:.3f}" if data['cos_sim'] is not None else "N/A"
        print(f"{step:<8} {data['grad_norm']:<12.2e} {cos_str:<12} {data['loss']:<12.2e}")
    
    # Analysis
    if len(results) >= 2:
        steps = sorted(results.keys())
        before_step = steps[0]
        after_step = steps[-1]
        
        grad_before = results[before_step]['grad_norm']
        grad_after = results[after_step]['grad_norm']
        
        cos_before = results[before_step]['cos_sim']
        cos_after = results[after_step]['cos_sim']
        
        print(f"\n🔍 ANALYSIS:")
        print(f"From step {before_step} to {after_step}:")
        print(f"  Gradient norm: {grad_before:.2e} → {grad_after:.2e} ({grad_after/grad_before:.2f}x)")
        
        if cos_before is not None and cos_after is not None:
            print(f"  cos(grad,H@grad): {cos_before:.3f} → {cos_after:.3f}")
            
            print(f"\n💡 CONCLUSION:")
            if grad_after/grad_before < 0.1:
                print("❌ GRADIENT VANISHING explains the cosine drop!")
                print("   The dramatic decrease in gradient norm makes cos(grad,H@grad)")
                print("   numerically unstable and unreliable.")
            elif grad_after/grad_before > 0.5:
                print("✅ GRADIENT NORM STABLE - cosine drop is meaningful!")
                print("   The gradient norm remains substantial, so the cosine drop")
                print("   reflects real changes in the optimization landscape.")
            else:
                print("⚠️  MIXED EVIDENCE - partial gradient vanishing")
                print("   Some gradient norm decrease, but cosine drop might still")
                print("   reflect real optimization changes.")
        else:
            print("⚠️  Could not compute cosine similarity")
    
    return results


def create_focused_plot():
    """Create a focused plot around the transition."""
    print("\n📊 Creating focused visualization...")
    
    # Load the existing 50k data if available
    try:
        with open('cleanup_metrics_final.pkl', 'rb') as f:
            data = pickle.load(f)
        
        steps = np.array(data['steps'])
        cos_sims = np.array(data['cos_sims'])
        test_accs = np.array(data['test_accs'])
        
        # Focus on transition period
        mask = (steps >= 15000) & (steps <= 30000)
        focused_steps = steps[mask]
        focused_cos = cos_sims[mask]
        focused_test = test_accs[mask]
        
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Test accuracy
        plt.subplot(2, 1, 1)
        plt.plot(focused_steps, focused_test, 'b-', linewidth=2)
        plt.ylabel('Test Accuracy')
        plt.title('Focused View: Steps 15k-30k (Around Cosine Drop)')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=20000, color='red', linestyle='--', alpha=0.7, label='Suspected cleanup')
        plt.legend()
        
        # Plot 2: Cosine similarity
        plt.subplot(2, 1, 2)
        plt.plot(focused_steps, focused_cos, 'r-', linewidth=2)
        plt.scatter(focused_steps, focused_cos, c='red', s=20, alpha=0.6)
        plt.ylabel('cos(grad, H@grad)')
        plt.xlabel('Training Steps')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=20000, color='red', linestyle='--', alpha=0.7)
        
        # Mark the points where we'll check gradient norms
        check_points = [18000, 20000, 22000, 24000]
        for point in check_points:
            if point in focused_steps:
                idx = np.where(focused_steps == point)[0][0]
                plt.scatter(point, focused_cos[idx], c='orange', s=100, zorder=5)
        
        plt.text(20000, 0.5, 'Gradient norm\ncheck points', ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('gradient_hypothesis_focus.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📁 Saved: gradient_hypothesis_focus.png")
        
    except:
        print("⚠️  Could not load existing data - run the full analysis first")


if __name__ == "__main__":
    # Run the analysis
    results = analyze_gradient_hypothesis()
    
    # Create focused plot
    create_focused_plot()
    
    print(f"\n🎯 SUMMARY:")
    print(f"This analysis checks if the cos(grad, H@grad) drop is due to")
    print(f"vanishing gradients rather than actual changes in the Hessian structure.")
    print(f"If gradient norms drop significantly, the cosine metric becomes unreliable.")