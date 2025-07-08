#!/usr/bin/env python3
"""Quick test to verify the setup works correctly."""

import torch
import numpy as np
from data import generate_induction_sequences, analyze_pattern_completion_accuracy
from model import InductionTransformer
from eigen_analysis import HessianEigenAnalyzer


def test_data_generation():
    """Test data generation."""
    print("Testing data generation...")
    sequences, targets, pattern_info = generate_induction_sequences(
        vocab_size=20,
        seq_length=64,
        n_sequences=100,
        min_pattern_length=2,
        max_pattern_length=5,
        n_repeats_per_seq=3
    )
    
    print(f"✓ Generated {len(sequences)} sequences")
    print(f"  Sequence shape: {sequences.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Example pattern: {pattern_info[0]}")
    
    # Verify patterns exist
    patterns_found = sum(len(p) > 0 for p in pattern_info)
    print(f"✓ Patterns found in {patterns_found}/{len(pattern_info)} sequences")
    

def test_model():
    """Test model initialization and forward pass."""
    print("\nTesting model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    model = InductionTransformer(
        vocab_size=20,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=64
    ).to(device)
    
    print(f"✓ Model initialized")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch = torch.randint(0, 20, (4, 64)).to(device)
    output = model(batch)
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {output.shape}")
    

def test_eigen_analysis():
    """Test eigenvalue computation."""
    print("\nTesting eigenvalue analysis...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Small model for quick test
    model = InductionTransformer(
        vocab_size=10,
        d_model=32,
        n_heads=2,
        n_layers=1
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    analyzer = HessianEigenAnalyzer(
        model=model,
        loss_fn=criterion,
        device=device,
        track_top_k=3
    )
    
    # Test data
    inputs = torch.randint(0, 10, (4, 32)).to(device)
    targets = torch.randint(0, 10, (4, 32)).to(device)
    
    # Compute gradient info
    grad_info = analyzer.compute_gradient_info((inputs, targets))
    print(f"✓ Gradient computation successful")
    print(f"  Gradient norm: {grad_info['grad_norm']:.4f}")
    
    # Compute eigenspectrum (might be slow)
    print("  Computing eigenvalues (this may take a moment)...")
    eigen_info = analyzer.compute_hessian_eigenspectrum((inputs, targets))
    print(f"✓ Eigenvalue computation successful")
    print(f"  Top eigenvalues: {eigen_info['eigenvalues'].tolist()}")
    

def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Induction Head Analysis Setup")
    print("=" * 50)
    
    try:
        test_data_generation()
        test_model()
        test_eigen_analysis()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed! Setup is working correctly.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()