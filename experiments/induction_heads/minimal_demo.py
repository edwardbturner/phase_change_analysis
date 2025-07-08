#!/usr/bin/env python3
"""Minimal demonstration of induction head formation concepts."""

import torch
import torch.nn as nn
import numpy as np

# Simplified pattern data
def create_simple_patterns(n_samples=100, vocab_size=10, seq_len=20):
    """Create simple sequences with repeated patterns like: A B C ... A B C"""
    data = []
    targets = []
    
    for _ in range(n_samples):
        seq = np.random.randint(1, vocab_size, seq_len)
        
        # Insert a repeated pattern
        pattern = np.random.randint(1, vocab_size, 3)
        pos1 = 2
        pos2 = 10
        
        seq[pos1:pos1+3] = pattern
        seq[pos2:pos2+3] = pattern
        
        # Target is next token
        target = np.roll(seq, -1)
        target[-1] = 0
        
        data.append(seq)
        targets.append(target)
    
    return torch.tensor(data), torch.tensor(targets)


def test_pattern_accuracy(model, sequences, pattern_positions):
    """Test if model learned to complete patterns."""
    model.eval()
    with torch.no_grad():
        outputs = model(sequences)
        preds = outputs.argmax(dim=-1)
        
        # Check accuracy at positions after patterns
        pattern_correct = 0
        random_correct = 0
        pattern_total = 0
        random_total = 0
        
        for i in range(len(sequences)):
            for j in range(len(sequences[i])-1):
                if j in pattern_positions:
                    if preds[i, j] == sequences[i, j+1]:
                        pattern_correct += 1
                    pattern_total += 1
                else:
                    if preds[i, j] == sequences[i, j+1]:
                        random_correct += 1
                    random_total += 1
        
        pattern_acc = pattern_correct / pattern_total if pattern_total > 0 else 0
        random_acc = random_correct / random_total if random_total > 0 else 0
        
        return pattern_acc, random_acc


# Simple transformer
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(50, d_model)
        self.transformer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=256, 
            batch_first=True, dropout=0.1
        )
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        x = self.embed(x) + self.pos_embed(pos)
        x = self.transformer(x)
        return self.output(x)


def main():
    print("Minimal Induction Head Demo")
    print("=" * 40)
    
    # Generate data
    sequences, targets = create_simple_patterns(500, vocab_size=20, seq_len=30)
    pattern_positions = [12, 13, 14]  # Positions after second pattern
    
    # Model
    model = MiniTransformer(vocab_size=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("\nTraining mini transformer...")
    print("Step | Loss   | Pattern | Random | Ratio")
    print("-" * 40)
    
    # Quick training
    for step in range(200):
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs.view(-1, 20), targets.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluate every 20 steps
        if step % 20 == 0:
            pattern_acc, random_acc = test_pattern_accuracy(model, sequences, pattern_positions)
            ratio = pattern_acc / max(random_acc, 0.01)
            print(f"{step:4d} | {loss:.4f} | {pattern_acc:.3f}   | {random_acc:.3f}  | {ratio:.2f}x")
    
    # Final evaluation
    pattern_acc, random_acc = test_pattern_accuracy(model, sequences, pattern_positions)
    print("\n" + "=" * 40)
    print(f"Final Pattern Accuracy: {pattern_acc:.3f}")
    print(f"Final Random Accuracy: {random_acc:.3f}")
    print(f"Improvement Ratio: {pattern_acc/max(random_acc, 0.01):.2f}x")
    
    if pattern_acc > random_acc * 1.5:
        print("\n✓ Induction behavior emerged!")
    else:
        print("\n✗ No clear induction behavior")


if __name__ == "__main__":
    main()