import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import os

from data import generate_induction_sequences, create_induction_data_loader, analyze_pattern_completion_accuracy
from model import InductionTransformer
from eigen_analysis import HessianEigenAnalyzer


def quick_test():
    """Quick test to demonstrate induction head formation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Small setup for quick testing
    vocab_size = 20
    seq_length = 64
    n_sequences = 1000
    
    # Generate data
    print("\nGenerating test data...")
    sequences, targets, pattern_info = generate_induction_sequences(
        vocab_size=vocab_size,
        seq_length=seq_length,
        n_sequences=n_sequences,
        min_pattern_length=2,
        max_pattern_length=5,
        n_repeats_per_seq=3,
        seed=42
    )
    
    # Create simple data loader
    train_loader = create_induction_data_loader(
        sequences[:800], targets[:800], 
        batch_size=32, shuffle=True
    )
    
    # Small model
    model = InductionTransformer(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=seq_length,
        dropout=0.1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Quick training loop
    print("\nTraining for 500 steps...")
    accuracies = []
    pattern_accs = []
    
    step = 0
    for epoch in range(10):
        for batch_idx, (batch_seq, batch_tgt) in enumerate(train_loader):
            if step >= 500:
                break
                
            batch_seq = batch_seq.to(device)
            batch_tgt = batch_tgt.to(device)
            
            # Forward pass
            outputs = model(batch_seq)
            loss = criterion(outputs.view(-1, outputs.size(-1)), batch_tgt.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log every 50 steps
            if step % 50 == 0:
                # Quick accuracy check
                with torch.no_grad():
                    preds = outputs.argmax(dim=-1)
                    acc = (preds == batch_tgt).float().mean().item()
                    
                # Pattern-specific accuracy (on small sample)
                pattern_acc, random_acc, _ = analyze_pattern_completion_accuracy(
                    model, sequences[:100], pattern_info[:100], device
                )
                
                accuracies.append(acc)
                pattern_accs.append(pattern_acc)
                
                print(f"Step {step:3d} | Loss: {loss.item():.4f} | "
                      f"Acc: {acc:.3f} | Pattern: {pattern_acc:.3f} | Random: {random_acc:.3f}")
            
            step += 1
            
        if step >= 500:
            break
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    pattern_acc, random_acc, per_pos = analyze_pattern_completion_accuracy(
        model, sequences[:200], pattern_info[:200], device
    )
    
    print(f"Pattern completion accuracy: {pattern_acc:.3f}")
    print(f"Random token accuracy: {random_acc:.3f}")
    print(f"Improvement ratio: {pattern_acc/max(random_acc, 0.001):.2f}x")
    
    if per_pos:
        print("\nPer-position accuracy in patterns:")
        for pos, acc in sorted(per_pos.items())[:5]:
            print(f"  Position {pos}: {acc:.3f}")
    
    # Check if induction behavior emerged
    if pattern_acc > random_acc * 1.5:
        print("\n✓ Induction head behavior detected!")
    else:
        print("\n✗ No clear induction head behavior yet")
    
    return {
        'accuracies': accuracies,
        'pattern_accs': pattern_accs,
        'final_pattern_acc': pattern_acc,
        'final_random_acc': random_acc
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Quick Induction Head Formation Test")
    print("=" * 60)
    
    results = quick_test()
    
    print("\n" + "=" * 60)
    print("Test complete!")
    
    # Simple visualization
    if len(results['pattern_accs']) > 1:
        print("\nPattern accuracy progression:")
        for i, acc in enumerate(results['pattern_accs']):
            bar = '█' * int(acc * 20)
            print(f"Step {i*50:3d}: [{bar:<20}] {acc:.3f}")