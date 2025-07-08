import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the results
with open('gradnorm_final_results.pkl', 'rb') as f:
    data = pickle.load(f)

steps = np.array(data['steps'])
grad_norms = np.array(data['grad_norms'])
cos_sims = np.array(data['cos_sims'])
test_accs = np.array(data['test_accs'])

print("="*70)
print("GRADIENT NORM HYPOTHESIS TEST - RESULTS")
print("="*70)

# Find grokking point
grok_step = None
grok_idx = None
for i, acc in enumerate(test_accs):
    if acc > 0.9:
        grok_step = steps[i]
        grok_idx = i
        break

print(f"🎯 Grokking complete at step: {grok_step}")
print(f"📊 Training completed to step: {steps[-1]}")

# Key analysis points
start_idx = 0
grok_idx = grok_idx if grok_idx else len(steps)//3
end_idx = len(steps) - 1

print(f"\n📈 GRADIENT NORM EVOLUTION:")
print(f"   Start (step {steps[start_idx]}): {grad_norms[start_idx]:.2e}")
if grok_idx < len(steps):
    print(f"   Grokking (step {steps[grok_idx]}): {grad_norms[grok_idx]:.2e}")
print(f"   End (step {steps[end_idx]}): {grad_norms[end_idx]:.2e}")

grad_ratio_total = grad_norms[end_idx] / grad_norms[start_idx]
if grok_idx < len(steps):
    grad_ratio_post_grok = grad_norms[end_idx] / grad_norms[grok_idx]
else:
    grad_ratio_post_grok = None

print(f"\n📉 COSINE SIMILARITY EVOLUTION:")
print(f"   Start (step {steps[start_idx]}): {cos_sims[start_idx]:.3f}")
if grok_idx < len(steps):
    print(f"   Grokking (step {steps[grok_idx]}): {cos_sims[grok_idx]:.3f}")
print(f"   End (step {steps[end_idx]}): {cos_sims[end_idx]:.3f}")

print(f"\n🔍 CRITICAL ANALYSIS:")
print(f"   Total gradient norm change: {grad_ratio_total:.2e}x")
if grad_ratio_post_grok:
    print(f"   Post-grokking gradient change: {grad_ratio_post_grok:.2e}x")

print(f"\n💡 VERDICT:")
if grad_ratio_total < 1e-3:
    print("❌ MASSIVE GRADIENT VANISHING detected!")
    print("   Gradient norm drops by >1000x during training.")
    print("   The cos(grad,H@grad) metric becomes completely unreliable.")
    print("   This is NOT evidence of cleanup phase - just numerical breakdown.")
    verdict = "VANISHING_GRADIENTS"
elif grad_ratio_total < 0.1:
    print("⚠️  SIGNIFICANT GRADIENT DECAY detected!")
    print("   Gradient norm drops by >10x during training.")
    print("   The cos(grad,H@grad) metric is likely unreliable.")
    print("   Probably NOT meaningful evidence of cleanup phase.")
    verdict = "LIKELY_VANISHING"
else:
    print("✅ GRADIENT NORMS REMAIN SUBSTANTIAL!")
    print("   The cos(grad,H@grad) drop likely reflects real optimization changes.")
    print("   This could be evidence of cleanup phase.")
    verdict = "MEANINGFUL"

# Create visualization
fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# Test Accuracy
axes[0].plot(steps, test_accs, 'b-', linewidth=2)
axes[0].set_ylabel('Test Accuracy')
axes[0].set_title('Gradient Norm Hypothesis Test: Complete Results', fontsize=16)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-0.05, 1.05)
if grok_step:
    axes[0].axvline(x=grok_step, color='green', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].text(grok_step, 0.5, 'Grokking', rotation=90, va='bottom')

# Cosine Similarity
axes[1].plot(steps, cos_sims, 'r-', linewidth=2)
axes[1].set_ylabel('cos(grad, H@grad)')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.1, 1.1)
if grok_step:
    axes[1].axvline(x=grok_step, color='green', linestyle='--', linewidth=2, alpha=0.7)

# Gradient Norm (log scale)
axes[2].plot(steps, grad_norms, 'orange', linewidth=2)
axes[2].set_ylabel('Gradient Norm (log)')
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3)
if grok_step:
    axes[2].axvline(x=grok_step, color='green', linestyle='--', linewidth=2, alpha=0.7)

# Combined view with dual y-axis
ax3 = axes[3]
ax3_twin = ax3.twinx()

line1 = ax3.plot(steps, cos_sims, 'r-', linewidth=2, label='cos(grad, H@grad)')
line2 = ax3_twin.plot(steps, grad_norms, 'orange', linewidth=2, label='Gradient Norm')

ax3.set_ylabel('cos(grad, H@grad)', color='red')
ax3_twin.set_ylabel('Gradient Norm (log)', color='orange')
ax3_twin.set_yscale('log')
ax3.set_xlabel('Training Steps')
ax3.grid(True, alpha=0.3)

# Mark grokking
if grok_step:
    ax3.axvline(x=grok_step, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax3_twin.axvline(x=grok_step, color='green', linestyle='--', linewidth=2, alpha=0.7)

# Combined legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='upper right')

# Add verdict text
if verdict == "VANISHING_GRADIENTS":
    color = 'red'
    text = 'VERDICT:\nGradient Vanishing\nExplains Cosine Drop'
elif verdict == "LIKELY_VANISHING":
    color = 'orange'
    text = 'VERDICT:\nLikely Gradient\nVanishing'
else:
    color = 'green'
    text = 'VERDICT:\nMeaningful\nCosine Drop'

ax3.text(0.7, 0.8, text, transform=ax3.transAxes, fontsize=12, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.7),
         verticalalignment='top')

plt.tight_layout()
plt.savefig('gradient_norm_hypothesis_final.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n📊 Visualization saved: gradient_norm_hypothesis_final.png")

# Detailed analysis by phases
print(f"\n📋 PHASE-BY-PHASE ANALYSIS:")

if grok_step:
    # Pre-grokking phase
    pre_grok_mask = steps < grok_step
    if np.any(pre_grok_mask):
        pre_grad_start = grad_norms[pre_grok_mask][0]
        pre_grad_end = grad_norms[pre_grok_mask][-1]
        pre_cos_start = cos_sims[pre_grok_mask][0]
        pre_cos_end = cos_sims[pre_grok_mask][-1]
        
        print(f"\n   Pre-Grokking (0 to {grok_step}):")
        print(f"     Gradient norm: {pre_grad_start:.2e} → {pre_grad_end:.2e} ({pre_grad_end/pre_grad_start:.2f}x)")
        print(f"     Cosine sim: {pre_cos_start:.3f} → {pre_cos_end:.3f}")
    
    # Post-grokking phase
    post_grok_mask = steps >= grok_step
    if np.any(post_grok_mask):
        post_grad_start = grad_norms[post_grok_mask][0]
        post_grad_end = grad_norms[post_grok_mask][-1]
        post_cos_start = cos_sims[post_grok_mask][0]
        post_cos_end = cos_sims[post_grok_mask][-1]
        
        print(f"\n   Post-Grokking ({grok_step} to {steps[-1]}):")
        print(f"     Gradient norm: {post_grad_start:.2e} → {post_grad_end:.2e} ({post_grad_end/post_grad_start:.2f}x)")
        print(f"     Cosine sim: {post_cos_start:.3f} → {post_cos_end:.3f}")

print(f"\n🎯 FINAL CONCLUSION:")
if verdict == "VANISHING_GRADIENTS":
    print("   Your hypothesis is CORRECT!")
    print("   The cos(grad, H@grad) drop is explained by vanishing gradients.")
    print("   This is NOT evidence of a cleanup phase.")
elif verdict == "LIKELY_VANISHING":
    print("   Your hypothesis is LIKELY CORRECT!")
    print("   Significant gradient decay makes the cosine metric unreliable.")
else:
    print("   The cosine drop appears to be meaningful.")
    print("   Could represent real optimization landscape changes.")