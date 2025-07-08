#!/usr/bin/env python3
"""
Demonstrate the phase transition in induction head formation.
This script runs a focused experiment to clearly show the abrupt transition.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from data import generate_induction_sequences, create_induction_data_loader, analyze_pattern_completion_accuracy
from model import InductionTransformer


def run_phase_transition_experiment():
    """Run experiment optimized to show clear phase transition."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Parameters chosen to encourage clear phase transition
    vocab_size = 30
    seq_length = 128
    n_sequences = 5000
    
    # Generate data with clear patterns
    print("\nGenerating data with repeated patterns...")
    sequences, targets, pattern_info = generate_induction_sequences(
        vocab_size=vocab_size,
        seq_length=seq_length,
        n_sequences=n_sequences,
        min_pattern_length=3,
        max_pattern_length=8,
        n_repeats_per_seq=4,
        seed=42
    )
    
    # Split data
    n_train = int(0.9 * len(sequences))
    train_sequences = sequences[:n_train]
    train_targets = targets[:n_train]
    train_pattern_info = pattern_info[:n_train]
    
    test_sequences = sequences[n_train:]
    test_targets = targets[n_train:]
    test_pattern_info = pattern_info[n_train:]
    
    # Data loaders
    train_loader = create_induction_data_loader(
        train_sequences, train_targets, 
        batch_size=64, shuffle=True
    )
    
    # Model - slightly larger to show clearer transition
    model = InductionTransformer(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=8,
        n_layers=2,
        d_ff=512,
        max_seq_len=seq_length,
        dropout=0.1,
        use_layernorm=True
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=3e-3,  # Slightly higher LR for faster convergence
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )
    criterion = nn.CrossEntropyLoss()
    
    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=3000, eta_min=3e-4
    )
    
    # Tracking
    steps = []
    train_losses = []
    pattern_accuracies = []
    random_accuracies = []
    overall_accuracies = []
    
    print("\nTraining and monitoring for phase transition...")
    print("Watch for sudden jump in pattern accuracy!\n")
    
    step = 0
    max_steps = 3000
    log_interval = 25
    
    # Training loop
    while step < max_steps:
        for batch_idx, (batch_seq, batch_tgt) in enumerate(train_loader):
            if step >= max_steps:
                break
                
            batch_seq = batch_seq.to(device)
            batch_tgt = batch_tgt.to(device)
            
            # Forward pass
            model.train()
            outputs = model(batch_seq)
            loss = criterion(outputs.view(-1, outputs.size(-1)), batch_tgt.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Detailed logging
            if step % log_interval == 0:
                model.eval()
                
                # Overall accuracy
                with torch.no_grad():
                    preds = outputs.argmax(dim=-1)
                    acc = (preds == batch_tgt).float().mean().item()
                
                # Pattern-specific accuracy on test set
                pattern_acc, random_acc, per_pos = analyze_pattern_completion_accuracy(
                    model, test_sequences[:200], test_pattern_info[:200], device
                )
                
                # Store metrics
                steps.append(step)
                train_losses.append(loss.item())
                pattern_accuracies.append(pattern_acc)
                random_accuracies.append(random_acc)
                overall_accuracies.append(acc)
                
                # Print update
                print(f"Step {step:4d} | Loss: {loss.item():.4f} | "
                      f"Pattern: {pattern_acc:.3f} | Random: {random_acc:.3f} | "
                      f"Ratio: {pattern_acc/max(random_acc, 0.001):.2f}x | "
                      f"LR: {optimizer.param_groups[0]['lr']:.5f}")
                
                # Check for phase transition
                if len(pattern_accuracies) > 5:
                    recent_improvement = pattern_accuracies[-1] - pattern_accuracies[-5]
                    if recent_improvement > 0.2:  # 20% jump in 5 logs
                        print("\n*** PHASE TRANSITION DETECTED! ***\n")
            
            step += 1
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_pattern_acc, final_random_acc, final_per_pos = analyze_pattern_completion_accuracy(
        model, test_sequences, test_pattern_info, device
    )
    
    print(f"Final pattern completion: {final_pattern_acc:.3f}")
    print(f"Final random accuracy: {final_random_acc:.3f}")
    print(f"Improvement ratio: {final_pattern_acc/max(final_random_acc, 0.001):.2f}x")
    
    # Visualize results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Loss curve
    ax1.plot(steps, train_losses, 'b-', alpha=0.8)
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Progress')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Accuracy curves
    ax2.plot(steps, pattern_accuracies, 'g-', linewidth=2, label='Pattern Completion')
    ax2.plot(steps, random_accuracies, 'orange', linewidth=2, label='Random Tokens')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Induction Head Formation - Phase Transition')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    # Highlight phase transition region
    if len(pattern_accuracies) > 10:
        # Find steepest increase
        diffs = np.diff(pattern_accuracies)
        if len(diffs) > 0:
            max_diff_idx = np.argmax(diffs)
            transition_step = steps[max_diff_idx]
            ax2.axvline(transition_step, color='red', linestyle='--', 
                       linewidth=2, alpha=0.7, label='Phase Transition')
            ax2.legend()
    
    # Improvement ratio
    ratios = [p/max(r, 0.001) for p, r in zip(pattern_accuracies, random_accuracies)]
    ax3.plot(steps, ratios, 'purple', linewidth=2)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Pattern/Random Ratio')
    ax3.set_title('Relative Performance (Pattern vs Random)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(1.0, color='black', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f'phase_transition_demo_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")
    
    plt.show()
    
    return {
        'steps': steps,
        'pattern_accuracies': pattern_accuracies,
        'random_accuracies': random_accuracies,
        'final_pattern_acc': final_pattern_acc,
        'final_random_acc': final_random_acc
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Induction Head Phase Transition Demonstration")
    print("=" * 70)
    print("\nThis experiment shows the abrupt phase transition when")
    print("transformer models suddenly develop pattern-matching abilities.\n")
    
    results = run_phase_transition_experiment()