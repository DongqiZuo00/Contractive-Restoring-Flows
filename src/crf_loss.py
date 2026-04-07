"""
Contractive Restoring Flow (CRF) Loss.

The CRF loss enforces strict transversal contraction by penalising deviations
from the ideal restoring force response −αP⊥δ. Under first-order Taylor expansion,
it decomposes into:

    L_CRF = ‖P⊥rₗ‖²  +  σ² ‖P⊥(J_f + αI)‖²_F
            (Drift)       (Spectral Energy)

When minimised, this achieves ‖P⊥Jₗ‖₂ = 1 − α < 1.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def compute_transversal_projection(
    delta_star: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute tangent direction and transversal projection.

    Args:
        delta_star: Teacher residual flux Δ*_l = h*_{l+1} - h*_l, shape (B, d).

    Returns:
        u_star: Unit tangent direction, shape (B, d).
        mu: Flux magnitude ‖Δ*_l‖₂, shape (B,).
        P_perp_delta: Function not returned; projection is applied inline via
                      the rank-1 formula P⊥x = x - (x·u*)u*.
    """
    mu = delta_star.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, 1)
    u_star = delta_star / mu  # (B, d)
    return u_star, mu.squeeze(-1)


def project_transversal(x: torch.Tensor, u_star: torch.Tensor) -> torch.Tensor:
    """Apply transversal projection P⊥x = x - (x·u*)u*.

    Args:
        x: Vectors to project, shape (B, d).
        u_star: Unit tangent direction, shape (B, d).

    Returns:
        P⊥x, shape (B, d).
    """
    # (x · u*) u*
    coeff = (x * u_star).sum(dim=-1, keepdim=True)  # (B, 1)
    return x - coeff * u_star


class CRFLoss(nn.Module):
    """Contractive Restoring Flow loss for a single layer.

    Implements Eq. (5) from the paper:
        L^(l)_CRF = E_δ[‖P⊥(f_θ(h* + δ) - Δ*) + α P⊥δ‖²]

    In practice, the expectation is estimated with a single noise sample per
    forward pass (unbiased, variance reduced by batch averaging).
    """

    def __init__(self, alpha: float = 0.10, sigma: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma

    def forward(
        self,
        student_residual_fn,
        h_star: torch.Tensor,
        delta_star: torch.Tensor,
        u_star: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the CRF loss at one layer.

        Args:
            student_residual_fn: Callable that maps h → f^(l)_θ(h), i.e. the
                student's residual block (attention + FFN, without the skip connection).
            h_star: Teacher hidden state h*_l, shape (B, d). Detached (stop-gradient).
            delta_star: Teacher residual flux Δ*_l, shape (B, d).
            u_star: Unit tangent direction u*_l, shape (B, d).

        Returns:
            Scalar CRF loss for this layer.
        """
        B, d = h_star.shape

        # Step 1: Sample isotropic Gaussian probe
        delta = torch.randn_like(h_star) * self.sigma  # (B, d)

        # Step 2: Student response at perturbed input
        h_perturbed = h_star.detach() + delta
        delta_tilde = student_residual_fn(h_perturbed)  # f^(l)_θ(h* + δ)

        # Step 3: Compute transversal error with restoring force target
        # e⊥ = P⊥(f_θ(h*+δ) - Δ*) + α P⊥δ
        fitting_error = delta_tilde - delta_star  # f_θ(h*+δ) - Δ*
        e_perp = project_transversal(fitting_error, u_star)
        delta_perp = project_transversal(delta, u_star)

        # The +αP⊥δ is the key: it shifts the minimisation target from
        # P⊥J_f = 0 (marginal stability) to P⊥J_f = -αP⊥ (strict contraction)
        e_total = e_perp + self.alpha * delta_perp

        # Step 4: Squared norm
        loss = (e_total ** 2).sum(dim=-1).mean()

        return loss


class GatedCRFLoss(nn.Module):
    """Full CRF loss with kinetic gating across all layers.

    Implements Eq. (8):
        L_CRF = Σ_l  w_l · L^(l)_CRF
        w_l = σ_gate((μ_l - τ) / ε_gate)
    """

    def __init__(
        self,
        alpha: float = 0.10,
        sigma: float = 0.01,
        tau: Optional[float] = None,
        eps_gate: Optional[float] = None,
        activation_threshold: float = 0.05,
    ):
        super().__init__()
        self.crf = CRFLoss(alpha=alpha, sigma=sigma)
        self.tau = tau
        self.eps_gate = eps_gate
        self.activation_threshold = activation_threshold

    def kinetic_gate(self, mu: torch.Tensor) -> torch.Tensor:
        """Smooth gating weight based on flux magnitude.

        Args:
            mu: Residual flux magnitude ‖Δ*_l‖₂, shape (B,).

        Returns:
            Gate weight w_l, shape (B,).
        """
        if self.tau is None:
            return torch.ones_like(mu)
        return torch.sigmoid((mu - self.tau) / self.eps_gate)

    def calibrate_gate(self, all_flux_magnitudes: torch.Tensor):
        """Set τ from the 5th percentile of flux magnitudes.

        Args:
            all_flux_magnitudes: Tensor of μ_l values from calibration pass.
        """
        self.tau = torch.quantile(all_flux_magnitudes, 0.05).item()
        self.eps_gate = self.tau / 5.0
        print(f"Kinetic gate calibrated: τ={self.tau:.4f}, ε_gate={self.eps_gate:.4f}")

    def forward(
        self,
        student_residual_fns,
        h_stars: list,
        delta_stars: list,
    ) -> torch.Tensor:
        """Compute gated CRF loss across all layers.

        Args:
            student_residual_fns: List of L callables, one per layer.
            h_stars: List of L teacher hidden states, each (B, d).
            delta_stars: List of L teacher residual fluxes, each (B, d).

        Returns:
            Scalar total CRF loss.
        """
        total_loss = torch.tensor(0.0, device=h_stars[0].device)
        num_active = 0

        for l, (res_fn, h_star, delta_star) in enumerate(
            zip(student_residual_fns, h_stars, delta_stars)
        ):
            u_star, mu = compute_transversal_projection(delta_star)
            w = self.kinetic_gate(mu)  # (B,)

            # Skip passive layers (w ≤ threshold)
            mean_w = w.mean().item()
            if mean_w <= self.activation_threshold:
                continue

            layer_loss = self.crf(res_fn, h_star, delta_star, u_star)
            total_loss = total_loss + mean_w * layer_loss
            num_active += 1

        return total_loss
