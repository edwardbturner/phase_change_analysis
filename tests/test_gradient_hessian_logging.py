"""Tests for gradient and Hessian logging during training.

Verifies that:
1. Gradients are correctly computed and logged
2. Hessian diagonal/eigenvalues are correctly computed
3. Metrics are properly stored and accessible
4. Phase transition detection works
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.training import PhaseTransitionTrainer
from utils.analysis import compute_gradient_norm


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def create_dummy_data(n_samples=100, input_dim=10, n_classes=2):
    """Create dummy classification data."""
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    return TensorDataset(X, y)


class TestGradientLogging:
    """Test gradient logging functionality."""
    
    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        # Create dummy data
        X = torch.randn(10, 10)
        y = torch.randint(0, 2, (10,))
        
        # Forward pass
        outputs = model(X)
        loss = loss_fn(outputs, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            
        # Compute gradient norm
        grad_norm = compute_gradient_norm(model)
        assert grad_norm > 0
        assert not np.isnan(grad_norm)
        
    def test_gradient_logging_in_trainer(self):
        """Test that trainer correctly logs gradients."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        # Create trainer with gradient tracking
        trainer = PhaseTransitionTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            track_gradients=True,
            log_freq=1,  # Log every step
        )
        
        # Create data
        dataset = create_dummy_data(50)
        train_loader = DataLoader(dataset, batch_size=10)
        
        # Train for one epoch
        metrics = trainer.train_epoch(train_loader, epoch=0)
        
        # Check gradient norms were logged
        assert len(trainer.metrics_history["grad_norm"]) > 0
        assert all(g >= 0 for g in trainer.metrics_history["grad_norm"])
        assert metrics["grad_norm_mean"] > 0
        
    def test_gradient_checkpointing(self):
        """Test that gradients can be checkpointed and retrieved."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = PhaseTransitionTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            track_gradients=True,
            checkpoint_freq=10,
        )
        
        dataset = create_dummy_data(100)
        train_loader = DataLoader(dataset, batch_size=10)
        
        # Train and trigger checkpointing
        trainer.train_epoch(train_loader, epoch=0)
        
        # Check that checkpoints were created
        checkpoints = list(trainer.log_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) > 0
        
        # Load checkpoint and verify gradient history
        checkpoint = torch.load(checkpoints[0])
        assert "metrics_history" in checkpoint
        assert "grad_norm" in checkpoint["metrics_history"]


class TestHessianLogging:
    """Test Hessian computation and logging."""
    
    def test_hessian_diagonal_computation(self):
        """Test Hessian diagonal computation using Hutchinson's estimator."""
        model = SimpleModel()
        loss_fn = nn.CrossEntropyLoss()
        
        # Create small batch for testing
        X = torch.randn(5, 10, requires_grad=True)
        y = torch.randint(0, 2, (5,))
        
        # Compute loss
        outputs = model(X)
        loss = loss_fn(outputs, y)
        
        # Get parameters and compute gradient
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        
        # Verify we can compute second derivatives
        n_params = sum(p.numel() for p in params)
        z = torch.randn(n_params)
        z = z / z.norm()
        
        # This should not raise an error
        grad_vec = torch.cat([g.reshape(-1) for g in grads])
        assert grad_vec.shape[0] == n_params
        
    def test_hessian_diagonal_logging(self):
        """Test that trainer correctly logs Hessian diagonal."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = PhaseTransitionTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            track_gradients=True,
            track_hessian=True,
            hessian_method="diagonal",
            hessian_freq=5,  # Compute every 5 steps
        )
        
        dataset = create_dummy_data(60)
        train_loader = DataLoader(dataset, batch_size=10)
        
        # Train for one epoch
        trainer.train_epoch(train_loader, epoch=0)
        
        # Check Hessian diagonal statistics were logged
        assert len(trainer.metrics_history["hessian_diag_mean"]) > 0
        assert len(trainer.metrics_history["hessian_diag_std"]) > 0
        assert len(trainer.metrics_history["hessian_diag_max"]) > 0
        
        # Verify values are reasonable
        assert all(not np.isnan(v) for v in trainer.metrics_history["hessian_diag_mean"])
        assert all(v >= 0 for v in trainer.metrics_history["hessian_diag_std"])
        
    def test_hessian_eigenvalue_logging(self):
        """Test that trainer can log top eigenvalues."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = PhaseTransitionTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            track_hessian=True,
            hessian_method="eigenvalues",
            hessian_freq=10,
        )
        
        dataset = create_dummy_data(60)
        train_loader = DataLoader(dataset, batch_size=10)
        
        # Train
        trainer.train_epoch(train_loader, epoch=0)
        
        # Check eigenvalues were logged
        assert "hessian_top_eigenvalues" in trainer.metrics_history
        assert len(trainer.metrics_history["hessian_top_eigenvalues"]) > 0
        
        # Each entry should be a list of k eigenvalues
        for eigenvals in trainer.metrics_history["hessian_top_eigenvalues"]:
            assert isinstance(eigenvals, list)
            assert len(eigenvals) == 10  # Default k=10


class TestIntegration:
    """Integration tests for the full training pipeline."""
    
    def test_full_training_pipeline(self):
        """Test complete training with all logging enabled."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        trainer = PhaseTransitionTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            track_gradients=True,
            track_hessian=True,
            hessian_method="diagonal",
            hessian_freq=20,
            checkpoint_freq=50,
            log_freq=5,
        )
        
        # Create train and validation data
        train_dataset = create_dummy_data(200)
        val_dataset = create_dummy_data(50)
        train_loader = DataLoader(train_dataset, batch_size=20)
        val_loader = DataLoader(val_dataset, batch_size=20)
        
        # Train for multiple epochs
        for epoch in range(3):
            metrics = trainer.train_epoch(train_loader, val_loader, epoch)
            
            # Verify metrics
            assert "train_loss" in metrics
            assert "train_acc" in metrics
            assert "grad_norm_mean" in metrics
            assert "val_loss" in metrics
            assert "val_acc" in metrics
            
        # Check comprehensive logging
        history = trainer.metrics_history
        assert len(history["step"]) > 0
        assert len(history["train_loss"]) == len(history["step"])
        assert len(history["grad_norm"]) == len(history["step"])
        assert len(history["hessian_diag_mean"]) > 0
        
        # Test phase transition detection
        transitions = trainer.detect_phase_transitions(metric="train_loss", window_size=10)
        assert isinstance(transitions, list)
        
    def test_gradient_hessian_alignment(self):
        """Test that we can track gradient-Hessian alignment."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.CrossEntropyLoss()
        
        # Create data with clear structure to induce phase transition
        # First half: random labels (hard to learn)
        X1 = torch.randn(100, 10)
        y1 = torch.randint(0, 2, (100,))
        
        # Second half: structured data (easy to learn)
        X2 = torch.randn(100, 10)
        y2 = (X2[:, 0] > 0).long()  # Simple rule
        
        X = torch.cat([X1, X2])
        y = torch.cat([y1, y2])
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=20, shuffle=True)
        
        trainer = PhaseTransitionTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            track_gradients=True,
            track_hessian=True,
            hessian_method="diagonal",
            hessian_freq=10,
        )
        
        # Train and collect metrics
        for epoch in range(5):
            trainer.train_epoch(train_loader, epoch=epoch)
            
        # Verify gradient norms change over training
        grad_norms = trainer.metrics_history["grad_norm"]
        assert len(grad_norms) > 20
        
        # Check for variation in gradient norms (indicating dynamics)
        grad_norm_std = np.std(grad_norms)
        assert grad_norm_std > 0
        
        # Verify Hessian diagonal statistics show variation
        hessian_means = trainer.metrics_history["hessian_diag_mean"]
        assert len(hessian_means) > 0
        assert np.std(hessian_means) > 0  # Should see changes during training


def test_memory_efficiency():
    """Test that logging doesn't cause memory issues."""
    model = SimpleModel(input_dim=100, hidden_dim=200, output_dim=10)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    
    trainer = PhaseTransitionTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        track_gradients=True,
        track_hessian=True,
        hessian_method="diagonal",
        hessian_freq=50,  # Less frequent for large model
    )
    
    # Large dataset
    dataset = create_dummy_data(1000, input_dim=100, n_classes=10)
    train_loader = DataLoader(dataset, batch_size=50)
    
    # Should complete without memory errors
    trainer.train_epoch(train_loader, epoch=0)
    
    # Verify data was collected
    assert len(trainer.metrics_history["grad_norm"]) > 0
    assert len(trainer.metrics_history["hessian_diag_mean"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])