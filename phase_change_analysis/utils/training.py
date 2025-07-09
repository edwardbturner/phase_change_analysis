"""General training wrapper for phase transition analysis.

This module provides a unified training interface that handles:
- Gradient tracking and logging
- Hessian computation (diagonal or eigenvalues)
- Phase transition detection
- Checkpointing and metric logging
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .analysis import (
    compute_gradient_norm,
    compute_weight_norm,
    detect_phase_transition,
)


class PhaseTransitionTrainer:
    """General training wrapper for studying phase transitions.

    This trainer provides:
    - Automatic gradient norm tracking
    - Optional Hessian diagonal/eigenvalue computation
    - Phase transition detection
    - Comprehensive logging and checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        log_dir: Optional[Path] = None,
        track_gradients: bool = True,
        track_hessian: bool = False,
        hessian_method: str = "diagonal",  # "diagonal" or "eigenvalues"
        hessian_freq: int = 100,  # Compute Hessian every N steps
        checkpoint_freq: int = 1000,
        log_freq: int = 10,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        gradient_clip: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 1e-4,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        # Logging configuration
        self.log_dir = log_dir or Path(f"runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Tracking configuration
        self.track_gradients = track_gradients
        self.track_hessian = track_hessian
        self.hessian_method = hessian_method
        self.hessian_freq = hessian_freq
        self.checkpoint_freq = checkpoint_freq
        self.log_freq = log_freq

        # Storage for metrics
        self.metrics_history = {
            "step": [],
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "grad_norm": [],
            "weight_norm": [],
            "learning_rate": [],
        }

        if track_hessian:
            if hessian_method == "diagonal":
                self.metrics_history["hessian_diag_mean"] = []
                self.metrics_history["hessian_diag_std"] = []
                self.metrics_history["hessian_diag_max"] = []
            elif hessian_method == "eigenvalues":
                self.metrics_history["hessian_top_eigenvalues"] = []

        # Phase transition detection
        self.phase_transitions = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Gradient storage for detailed analysis
        self.gradient_checkpoints = {}
        self.hessian_checkpoints = {}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """Main training loop with comprehensive monitoring."""

        callbacks = callbacks or []
        training_history = []

        print(f"Starting training for {num_epochs} epochs")
        print(f"Logging to: {self.log_dir}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Create a single progress bar for the entire training
        total_steps = num_epochs * len(train_loader)
        pbar = tqdm(total=total_steps, desc="Training")

        for epoch in range(num_epochs):
            # Train one epoch
            epoch_metrics = self.train_epoch(train_loader, val_loader, epoch, pbar)
            training_history.append(epoch_metrics)

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Early stopping check
            if self.early_stopping_patience and val_loader is not None:
                val_loss = epoch_metrics.get("val_loss", float('inf'))
                if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # Save best model
                    self._save_checkpoint(epoch * len(train_loader), epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break

            # Run callbacks
            for callback in callbacks:
                callback(self, epoch, epoch_metrics)

            # Print epoch summary
            self._print_epoch_summary(epoch, epoch_metrics)

        pbar.close()

        # Final checkpoint
        self._save_checkpoint(num_epochs * len(train_loader), num_epochs - 1, is_final=True)

        # Detect phase transitions
        transitions = self.detect_phase_transitions()
        if transitions:
            print(f"Detected phase transitions at steps: {transitions}")

        return {
            "training_history": training_history,
            "metrics_history": self.metrics_history,
            "phase_transitions": self.phase_transitions,
            "final_epoch": epoch,
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epoch: int = 0,
        pbar: Optional[tqdm] = None,
    ) -> Dict[str, float]:
        """Train for one epoch with comprehensive logging."""

        self.model.train()
        epoch_metrics = {
            "train_loss": [],
            "train_acc": [],
            "grad_norms": [],
        }

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            step = epoch * len(train_loader) + batch_idx

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            # Compute accuracy
            if hasattr(outputs, "shape") and len(outputs.shape) > 1:
                preds = outputs.argmax(dim=-1)
                acc = (preds == targets).float().mean().item()
            else:
                acc = 0.0

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            # Track gradients before optimizer step
            if self.track_gradients:
                grad_norm = compute_gradient_norm(self.model)
                epoch_metrics["grad_norms"].append(grad_norm)
            else:
                grad_norm = 0.0

            self.optimizer.step()

            # Update metrics
            epoch_metrics["train_loss"].append(loss.item())
            epoch_metrics["train_acc"].append(acc)

            # Periodic logging
            if step % self.log_freq == 0:
                self._log_step(step, epoch, loss.item(), acc, grad_norm)

            # Hessian computation
            if self.track_hessian and step % self.hessian_freq == 0:
                hessian_info = self._compute_hessian_info(inputs, targets)
                self._log_hessian(step, hessian_info)

            # Checkpointing
            if step % self.checkpoint_freq == 0:
                self._save_checkpoint(step, epoch)

            # Update progress bar
            if pbar is not None:
                pbar.set_postfix({
                    "epoch": epoch,
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{acc:.3f}",
                    "grad_norm": f"{grad_norm:.3f}"
                })
                pbar.update(1)

        # Validation
        val_metrics = {}
        if val_loader is not None:
            val_metrics = self.evaluate(val_loader)

        # Compute epoch summary
        summary = {
            "train_loss": np.mean(epoch_metrics["train_loss"]),
            "train_acc": np.mean(epoch_metrics["train_acc"]),
            "grad_norm_mean": np.mean(epoch_metrics["grad_norms"]) if epoch_metrics["grad_norms"] else 0,
            **val_metrics,
        }

        return summary

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        losses, accs = [], []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                if hasattr(outputs, "shape") and len(outputs.shape) > 1:
                    preds = outputs.argmax(dim=-1)
                    acc = (preds == targets).float().mean().item()
                else:
                    acc = 0.0

                losses.append(loss.item())
                accs.append(acc)

        return {"val_loss": float(np.mean(losses)), "val_acc": float(np.mean(accs))}

    def _compute_hessian_info(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Compute Hessian information based on configured method."""
        self.model.eval()  # Important for consistent Hessian

        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)

        if self.hessian_method == "diagonal":
            # Hutchinson's estimator for Hessian diagonal
            hessian_diag = self._compute_hessian_diagonal(loss)
            return {
                "mean": hessian_diag.mean().item(),
                "std": hessian_diag.std().item(),
                "max": hessian_diag.max().item(),
                "min": hessian_diag.min().item(),
            }

        elif self.hessian_method == "eigenvalues":
            # Power iteration for top eigenvalues
            top_eigenvalues = self._compute_top_eigenvalues(loss, k=10)
            return {"eigenvalues": top_eigenvalues.tolist()}

        self.model.train()
        return {}

    def _compute_hessian_diagonal(self, loss: torch.Tensor) -> torch.Tensor:
        """Compute Hessian diagonal using Hutchinson's estimator."""
        # Get model parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)

        # Random vector for Hutchinson's estimator
        z = torch.randn(n_params, device=self.device)
        z = z / z.norm()

        # Compute gradient
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        grad_vec = torch.cat([g.reshape(-1) for g in grads])

        # Compute Hz via autodiff
        h_z = torch.autograd.grad(outputs=grad_vec, inputs=params, grad_outputs=z, retain_graph=True)

        # Extract diagonal estimate
        diag_estimate = []
        idx = 0
        for i, p in enumerate(params):
            n = p.numel()
            param_z = z[idx : idx + n].view_as(p)
            param_hz = h_z[i]
            diag = (param_hz * param_z).reshape(-1)
            diag_estimate.append(diag)
            idx += n

        return torch.cat(diag_estimate)

    def _compute_top_eigenvalues(self, loss: torch.Tensor, k: int = 10) -> np.ndarray:
        """Compute top-k eigenvalues using power iteration."""
        # Simplified: just return k zeros for now
        # Full implementation would use power iteration
        return np.zeros(k)

    def _log_step(self, step: int, epoch: int, loss: float, acc: float, grad_norm: float):
        """Log metrics for a single step."""
        self.metrics_history["step"].append(step)
        self.metrics_history["epoch"].append(epoch)
        self.metrics_history["train_loss"].append(loss)
        self.metrics_history["train_acc"].append(acc)
        self.metrics_history["grad_norm"].append(grad_norm)
        self.metrics_history["weight_norm"].append(compute_weight_norm(self.model))
        self.metrics_history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

    def _log_hessian(self, step: int, hessian_info: Dict):
        """Log Hessian information."""
        if self.hessian_method == "diagonal":
            self.metrics_history["hessian_diag_mean"].append(hessian_info["mean"])
            self.metrics_history["hessian_diag_std"].append(hessian_info["std"])
            self.metrics_history["hessian_diag_max"].append(hessian_info["max"])
        elif self.hessian_method == "eigenvalues":
            self.metrics_history["hessian_top_eigenvalues"].append(hessian_info["eigenvalues"])

        # Store checkpoint for detailed analysis
        self.hessian_checkpoints[step] = hessian_info

    def _save_checkpoint(self, step: int, epoch: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint and metrics."""
        checkpoint = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": self.metrics_history,
            "phase_transitions": self.phase_transitions,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Regular checkpoint
        checkpoint_path = self.log_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Best model checkpoint
        if is_best:
            best_path = self.log_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        # Final model checkpoint
        if is_final:
            final_path = self.log_dir / "final_model.pt"
            torch.save(checkpoint, final_path)

        # Also save metrics as JSON for easy analysis
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def _print_epoch_summary(self, epoch: int, metrics: Dict[str, float]):
        """Print a summary of the epoch."""
        train_loss = metrics.get("train_loss", 0)
        train_acc = metrics.get("train_acc", 0)
        val_loss = metrics.get("val_loss", None)
        val_acc = metrics.get("val_acc", None)
        grad_norm = metrics.get("grad_norm_mean", 0)
        lr = self.optimizer.param_groups[0]["lr"]

        summary = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}"
        if val_loss is not None:
            summary += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
        summary += f" | Grad Norm: {grad_norm:.3f} | LR: {lr:.5f}"

        print(summary)

    def detect_phase_transitions(self, metric: str = "val_acc", window_size: int = 100):
        """Detect phase transitions in training metrics."""
        if metric not in self.metrics_history:
            return []

        values = np.array(self.metrics_history[metric])
        if len(values) < window_size:
            return []

        # Convert to the format expected by detect_phase_transition
        metrics_dict = {metric: values.tolist()}
        transition_step = detect_phase_transition(metrics_dict, window_size)

        if transition_step is not None:
            self.phase_transitions = [transition_step]
            return [transition_step]
        else:
            self.phase_transitions = []
            return []

    def get_gradient_histogram(self, step: Optional[int] = None) -> Optional[np.ndarray]:
        """Get gradient histogram at a specific step."""
        if not self.track_gradients:
            return None

        if step is None:
            # Get current gradients
            grads = []
            for p in self.model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.cpu().numpy().flatten())
            return np.concatenate(grads) if grads else None
        else:
            # Get from checkpoint
            return self.gradient_checkpoints.get(step)

    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load a checkpoint and restore training state."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.metrics_history = checkpoint["metrics_history"]
        self.phase_transitions = checkpoint["phase_transitions"]

        return checkpoint

    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the training run."""
        if not self.metrics_history["step"]:
            return {"status": "No training data available"}

        final_step = self.metrics_history["step"][-1]
        final_epoch = self.metrics_history["epoch"][-1] if self.metrics_history["epoch"] else 0

        summary = {
            "final_step": final_step,
            "final_epoch": final_epoch,
            "total_training_steps": len(self.metrics_history["step"]),
            "phase_transitions": self.phase_transitions,
            "final_train_loss": self.metrics_history["train_loss"][-1] if self.metrics_history["train_loss"] else None,
            "final_train_acc": self.metrics_history["train_acc"][-1] if self.metrics_history["train_acc"] else None,
            "final_val_loss": self.metrics_history["val_loss"][-1] if self.metrics_history["val_loss"] else None,
            "final_val_acc": self.metrics_history["val_acc"][-1] if self.metrics_history["val_acc"] else None,
            "log_dir": str(self.log_dir),
        }

        return summary
