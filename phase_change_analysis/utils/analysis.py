from typing import Dict, List, Optional

import numpy as np
import torch


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute the L2 norm of gradients across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return np.sqrt(total_norm)


def compute_weight_norm(model: torch.nn.Module) -> float:
    """Compute the L2 norm of weights across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        total_norm += p.data.norm(2).item() ** 2
    return np.sqrt(total_norm)


def compute_eigenvalues(matrix: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """
    Compute eigenvalues of a matrix.

    Args:
        matrix: Input matrix
        k: Number of top eigenvalues to return (None for all)

    Returns:
        Eigenvalues in descending order
    """
    eigenvalues = torch.linalg.eigvalsh(matrix)
    eigenvalues = torch.sort(eigenvalues, descending=True)[0]

    if k is not None:
        eigenvalues = eigenvalues[:k]

    return eigenvalues


def analyze_weight_matrix(W: torch.Tensor) -> Dict[str, float]:
    """Analyze properties of a weight matrix."""
    with torch.no_grad():
        # Compute singular values
        U, S, V = torch.svd(W)

        # Compute condition number
        condition_number = S[0] / S[-1] if S[-1] > 1e-10 else float("inf")

        # Compute effective rank (number of singular values > threshold)
        threshold = 1e-3 * S[0]
        effective_rank = (S > threshold).sum().item()

        # Compute Frobenius norm
        frobenius_norm = torch.norm(W, "fro").item()

        # Compute spectral norm (largest singular value)
        spectral_norm = S[0].item()

    return {
        "condition_number": float(condition_number),
        "effective_rank": float(effective_rank),
        "frobenius_norm": frobenius_norm,
        "spectral_norm": spectral_norm,
        "mean_singular_value": S.mean().item(),
        "std_singular_value": S.std().item(),
    }


def detect_phase_transition(
    metrics: Dict[str, List[float]], window_size: int = 100, threshold: float = 2.0
) -> Optional[int]:
    """
    Detect phase transition point based on rate of change in metrics.

    Args:
        metrics: Dictionary of metric histories
        window_size: Size of sliding window for computing derivatives
        threshold: Threshold for detecting significant changes

    Returns:
        Step index of detected phase transition, or None
    """
    # Use test accuracy as primary indicator
    if "test_acc" not in metrics:
        return None

    test_acc = np.array(metrics["test_acc"])

    if len(test_acc) < 2 * window_size:
        return None

    # Compute rolling gradient
    gradients = []
    for i in range(window_size, len(test_acc) - window_size):
        left_mean = np.mean(test_acc[i - window_size : i])
        right_mean = np.mean(test_acc[i : i + window_size])
        gradient = (right_mean - left_mean) / window_size
        gradients.append(gradient)

    gradients = np.array(gradients)

    # Find peak gradient
    if len(gradients) > 0:
        peak_idx = np.argmax(gradients)
        if gradients[peak_idx] > threshold / 1000:  # Normalize threshold
            return int(peak_idx + window_size)

    return None


def compute_alignment(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    v1_norm = v1 / (v1.norm() + 1e-8)
    v2_norm = v2 / (v2.norm() + 1e-8)
    return torch.dot(v1_norm, v2_norm).item()
