import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class HessianEigenAnalyzer:
    """
    Analyzes Hessian eigenvalues/eigenvectors during training to track
    phase transitions and induction head formation.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: torch.device,
        track_top_k: int = 20,
        use_power_iteration: bool = True,
        track_attention_heads: bool = True,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.track_top_k = track_top_k
        self.use_power_iteration = use_power_iteration
        self.track_attention_heads = track_attention_heads

        # Storage for tracking
        self.eigenvalue_history = []
        self.eigenvector_projections = []
        self.gradient_info = []
        self.attention_head_info = []

    def compute_gradient_info(self, data_batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        """Compute gradient statistics."""
        inputs, targets = data_batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.model.train()
        outputs = self.model(inputs)

        # Reshape for loss computation
        outputs_flat = outputs.view(-1, outputs.size(-1))
        targets_flat = targets.view(-1)

        loss = self.loss_fn(outputs_flat, targets_flat)

        # Compute gradients
        grads = torch.autograd.grad(loss, list(self.model.parameters()), create_graph=True)

        # Flatten gradients
        grad_vec = torch.cat([g.reshape(-1) for g in grads])

        # Compute gradient norm and direction
        grad_norm = torch.norm(grad_vec).item()

        return {"loss": loss.item(), "grad_norm": grad_norm, "grad_vec": grad_vec.detach()}

    def compute_hessian_eigenspectrum(
        self, data_batch: Tuple[torch.Tensor, torch.Tensor], method: str = "power_iteration"
    ) -> Dict:
        """
        Compute top eigenvalues and eigenvectors of the Hessian.

        Args:
            data_batch: Batch of data for Hessian computation
            method: 'power_iteration' for large models, 'full' for small models

        Returns:
            Dict with eigenvalues, eigenvectors, and computation time
        """
        start_time = time.time()

        if method == "power_iteration":
            result = self._power_iteration_spectrum(data_batch)
        else:
            result = self._full_hessian_spectrum(data_batch)

        result["computation_time"] = time.time() - start_time
        return result

    def _power_iteration_spectrum(self, data_batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        """Compute top-k eigenvalues using power iteration."""
        inputs, targets = data_batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # Compute loss
        self.model.train()
        outputs = self.model(inputs)
        outputs_flat = outputs.view(-1, outputs.size(-1))
        targets_flat = targets.view(-1)
        loss = self.loss_fn(outputs_flat, targets_flat)

        # Get model parameters
        params = list(self.model.parameters())
        n_params = sum(p.numel() for p in params)

        eigenvalues = []
        eigenvectors = []

        # Deflation method for multiple eigenvalues
        deflation_vectors = []

        for i in range(min(self.track_top_k, 10)):  # Limit to 10 for efficiency
            # Initialize random vector
            v = torch.randn(n_params, device=self.device)
            v = v / torch.norm(v)

            # Orthogonalize against previous eigenvectors
            for prev_v in deflation_vectors:
                v = v - torch.dot(v, prev_v) * prev_v
                v = v / torch.norm(v)

            # Power iteration
            for _ in range(20):  # 20 iterations usually sufficient
                # Compute Hv
                Hv = self._compute_hvp(loss, params, v)

                # Orthogonalize
                for prev_v in deflation_vectors:
                    Hv = Hv - torch.dot(Hv, prev_v) * prev_v

                # Normalize
                eigenvalue = torch.norm(Hv).item()
                if eigenvalue > 1e-10:
                    v = Hv / eigenvalue
                else:
                    break

            eigenvalues.append(eigenvalue)
            eigenvectors.append(v.detach())
            deflation_vectors.append(v.detach())

        return {
            "eigenvalues": torch.tensor(eigenvalues),
            "eigenvectors": torch.stack(eigenvectors) if eigenvectors else None,
            "method": "power_iteration",
        }

    def _full_hessian_spectrum(self, data_batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        """Compute full Hessian spectrum (for small models only)."""
        inputs, targets = data_batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # Compute loss
        self.model.train()
        outputs = self.model(inputs)
        outputs_flat = outputs.view(-1, outputs.size(-1))
        targets_flat = targets.view(-1)
        loss = self.loss_fn(outputs_flat, targets_flat)

        # Get model parameters
        params = list(self.model.parameters())
        n_params = sum(p.numel() for p in params)

        # Build full Hessian matrix
        H = torch.zeros(n_params, n_params, device=self.device)

        # Build Hessian column by column using finite differences
        for i in range(n_params):
            # Create unit vector
            unit_vec = torch.zeros(n_params, device=self.device)
            unit_vec[i] = 1.0

            # Compute Hessian-vector product
            Hv = self._compute_hvp(loss, params, unit_vec)
            H[:, i] = Hv

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(H)

        # Sort in descending order
        sorted_indices = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Take top-k
        top_k = min(self.track_top_k, len(eigenvalues))

        return {
            "eigenvalues": eigenvalues[:top_k],
            "eigenvectors": eigenvectors[:, :top_k].T,  # Transpose to match power iteration format
            "method": "full",
        }

    def _compute_hvp(self, loss: torch.Tensor, params: List[torch.nn.Parameter], vector: torch.Tensor) -> torch.Tensor:
        """Compute Hessian-vector product."""
        # First compute gradients
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

        # Reshape vector to match parameter shapes
        start = 0
        vec_params = []
        for p in params:
            size = p.numel()
            vec_params.append(vector[start : start + size].view_as(p))
            start += size

        # Compute grad @ vector as a tensor
        grad_dot_vec = torch.stack([(g * v).sum() for g, v in zip(grads, vec_params)]).sum()

        # Compute Hessian @ vector
        Hv_params = torch.autograd.grad(grad_dot_vec, params, retain_graph=True)

        # Flatten back to vector
        Hv = torch.cat([h.reshape(-1) for h in Hv_params])

        return Hv

    def compute_gradient_hessian_alignment(self, data_batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict:
        """
        Compute alignment between gradient and Hessian eigenvectors.
        This is crucial for understanding optimization dynamics.
        """
        # Get gradient info
        grad_info = self.compute_gradient_info(data_batch)
        grad_vec = grad_info["grad_vec"]
        grad_vec_normalized = grad_vec / torch.norm(grad_vec)

        # Get Hessian spectrum
        eigen_info = self.compute_hessian_eigenspectrum(data_batch)

        alignments = []
        if eigen_info["eigenvectors"] is not None:
            for i, eigvec in enumerate(eigen_info["eigenvectors"]):
                alignment = torch.dot(grad_vec_normalized, eigvec).item()
                alignments.append(
                    {
                        "eigenvalue": eigen_info["eigenvalues"][i].item(),
                        "alignment": alignment,
                        "aligned_component": alignment**2,  # Squared projection
                    }
                )

        # Compute gradient decomposition in eigenspace
        if eigen_info["eigenvectors"] is not None and len(eigen_info["eigenvectors"]) > 0:
            # Project gradient onto top eigenspace
            projections = torch.matmul(eigen_info["eigenvectors"], grad_vec_normalized)
            reconstruction_error = torch.norm(
                grad_vec_normalized - torch.matmul(projections, eigen_info["eigenvectors"])
            ).item()
        else:
            projections = None
            reconstruction_error = 1.0

        return {
            "grad_norm": grad_info["grad_norm"],
            "eigenvalues": eigen_info["eigenvalues"],
            "alignments": alignments,
            "projections": projections,
            "reconstruction_error": reconstruction_error,
            "top_eigenvalue": eigen_info["eigenvalues"][0].item() if len(eigen_info["eigenvalues"]) > 0 else 0,
            "spectral_norm": eigen_info["eigenvalues"][0].item() if len(eigen_info["eigenvalues"]) > 0 else 0,
        }

    def analyze_attention_heads(self, model: nn.Module) -> Dict:
        """
        Analyze attention head parameters to detect induction head formation.

        Looks for:
        - QK circuit formation (for pattern matching)
        - OV circuit formation (for copying)
        """
        attention_analysis = {}

        # Try to find attention layers in the model
        attention_layers = []

        # Common patterns for attention layers
        for name, module in model.named_modules():
            if hasattr(module, "W_Q") and hasattr(module, "W_K") and hasattr(module, "W_V") and hasattr(module, "W_O"):
                attention_layers.append((name, module))
            elif (
                hasattr(module, "q_proj")
                and hasattr(module, "k_proj")
                and hasattr(module, "v_proj")
                and hasattr(module, "out_proj")
            ):
                attention_layers.append((name, module))
            elif (
                hasattr(module, "to_q")
                and hasattr(module, "to_k")
                and hasattr(module, "to_v")
                and hasattr(module, "to_out")
            ):
                attention_layers.append((name, module))

        if not attention_layers:
            # If no attention layers found, return empty analysis
            return {"error": "No attention layers found in model"}

        for layer_idx, (layer_name, attn) in enumerate(attention_layers):
            try:
                # Try different common naming patterns for attention weights
                if hasattr(attn, "W_Q"):
                    W_Q = attn.W_Q.weight.detach()
                    W_K = attn.W_K.weight.detach()
                    W_V = attn.W_V.weight.detach()
                    W_O = attn.W_O.weight.detach()
                elif hasattr(attn, "q_proj"):
                    W_Q = attn.q_proj.weight.detach()
                    W_K = attn.k_proj.weight.detach()
                    W_V = attn.v_proj.weight.detach()
                    W_O = attn.out_proj.weight.detach()
                elif hasattr(attn, "to_q"):
                    W_Q = attn.to_q.weight.detach()
                    W_K = attn.to_k.weight.detach()
                    W_V = attn.to_v.weight.detach()
                    W_O = attn.to_out[0].weight.detach()  # Assuming to_out is a Sequential
                else:
                    continue

                # Analyze QK circuit (should learn to match positions)
                QK = torch.matmul(W_Q.T, W_K)  # d_model x d_model
                qk_eigenvalues = torch.linalg.eigvalsh(QK)

                # Analyze OV circuit (should learn to copy)
                OV = torch.matmul(W_O, W_V.T)  # d_model x d_model
                ov_eigenvalues = torch.linalg.eigvalsh(OV)

                # Check for "copying" behavior in OV circuit
                # Strong diagonal elements suggest copying
                ov_diag_strength = torch.mean(torch.abs(torch.diag(OV))).item()
                ov_off_diag_strength = torch.mean(torch.abs(OV - torch.diag(torch.diag(OV)))).item()

                attention_analysis[f"layer_{layer_idx}_{layer_name}"] = {
                    "qk_top_eigenvalue": qk_eigenvalues[-1].item(),
                    "qk_eigenvalue_gap": (
                        (qk_eigenvalues[-1] - qk_eigenvalues[-2]).item() if len(qk_eigenvalues) > 1 else 0
                    ),
                    "ov_top_eigenvalue": ov_eigenvalues[-1].item(),
                    "ov_eigenvalue_gap": (
                        (ov_eigenvalues[-1] - ov_eigenvalues[-2]).item() if len(ov_eigenvalues) > 1 else 0
                    ),
                    "ov_diag_strength": ov_diag_strength,
                    "ov_copy_ratio": ov_diag_strength / (ov_off_diag_strength + 1e-8),
                }
            except Exception as e:
                attention_analysis[f"layer_{layer_idx}_{layer_name}"] = {"error": f"Failed to analyze layer: {str(e)}"}

        return attention_analysis

    def track_training_step(self, data_batch: Tuple[torch.Tensor, torch.Tensor], step: int, model_accuracy: float):
        """Track eigenspectrum and related metrics during training."""
        # Compute gradient-Hessian alignment
        alignment_info = self.compute_gradient_hessian_alignment(data_batch)

        # Analyze attention heads if requested
        if self.track_attention_heads:
            attn_info = self.analyze_attention_heads(self.model)
        else:
            attn_info = {}

        # Store results
        self.eigenvalue_history.append(
            {
                "step": step,
                "accuracy": model_accuracy,
                "top_eigenvalues": (
                    alignment_info["eigenvalues"][:5].tolist()
                    if len(alignment_info["eigenvalues"]) >= 5
                    else alignment_info["eigenvalues"].tolist()
                ),
                "grad_norm": alignment_info["grad_norm"],
                "top_alignment": alignment_info["alignments"][0]["alignment"] if alignment_info["alignments"] else 0,
                "spectral_norm": alignment_info["spectral_norm"],
                "attention_info": attn_info,
            }
        )

    def detect_phase_transition(self, window_size: int = 100) -> Optional[int]:
        """
        Detect phase transitions based on eigenvalue dynamics.

        Returns step where phase transition likely occurred.
        """
        if len(self.eigenvalue_history) < window_size:
            return None

        # Look for rapid changes in top eigenvalue
        top_eigenvalues = [h["top_eigenvalues"][0] for h in self.eigenvalue_history if h["top_eigenvalues"]]

        if len(top_eigenvalues) < window_size:
            return None

        # Compute rate of change
        rates = []
        for i in range(window_size, len(top_eigenvalues)):
            window_before = top_eigenvalues[i - window_size : i - window_size // 2]
            window_after = top_eigenvalues[i - window_size // 2 : i]

            mean_before = np.mean(window_before)
            mean_after = np.mean(window_after)

            if mean_before > 0:
                rate = abs(mean_after - mean_before) / mean_before
                rates.append((i, rate))

        if rates:
            # Find maximum rate of change
            max_step, max_rate = max(rates, key=lambda x: x[1])

            # Threshold for phase transition detection
            if max_rate > 0.5:  # 50% change
                return self.eigenvalue_history[max_step]["step"]

        return None
