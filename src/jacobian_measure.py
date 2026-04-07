"""
Post-hoc measurement of the transversal contraction rate ‖P⊥Jₗ‖₂.

Uses a Hutchinson-style random-projection estimator (Appendix G.1):
  1. Sample k unit vectors {vᵢ} ~ Uniform(S^{d-1})
  2. Compute finite-difference Jacobian–vector products: Jₗvᵢ ≈ (T(h*+ηv) - T(h*))/η
  3. Project: P⊥(Jₗvᵢ)
  4. Estimate: ‖P⊥Jₗ‖₂ ≈ max_i ‖P⊥(Jₗvᵢ)‖₂
"""

import torch
import numpy as np
from typing import List, Dict
from crf_loss import compute_transversal_projection, project_transversal


@torch.no_grad()
def measure_transversal_contraction(
    model,
    h_stars: List[torch.Tensor],
    delta_stars: List[torch.Tensor],
    num_projections: int = 64,
    fd_step: float = 1e-3,
) -> Dict[str, np.ndarray]:
    """Measure ‖P⊥Jₗ‖₂ at every layer via random projection.

    Args:
        model: Student model (in eval mode).
        h_stars: Teacher hidden states per layer, each (B, d).
        delta_stars: Teacher residual fluxes per layer, each (B, d).
        num_projections: Number of random unit vectors (k=64 recommended).
        fd_step: Finite-difference step η (default 1e-3).

    Returns:
        Dict with:
          - "contraction_rates": (L,) array of estimated ‖P⊥Jₗ‖₂
          - "flux_magnitudes": (L,) array of μₗ = ‖Δ*ₗ‖₂
          - "per_projection": (L, k) array of individual ‖P⊥(Jₗvᵢ)‖₂
    """
    model.eval()
    L = len(h_stars)
    d = h_stars[0].shape[-1]
    device = h_stars[0].device

    contraction_rates = []
    flux_magnitudes = []
    per_projection = []

    for l in range(L):
        h_star = h_stars[l]  # (B, d) — we use B=1 for measurement
        delta_star = delta_stars[l]

        # Tangent direction and flux
        u_star, mu = compute_transversal_projection(delta_star)
        flux_magnitudes.append(mu.mean().item())

        if mu.mean().item() < 1e-6:
            # Passive layer — ‖P⊥Jₗ‖₂ ≈ 1 by definition
            contraction_rates.append(1.0)
            per_projection.append(np.ones(num_projections))
            continue

        # Get the layer's residual function
        layer_module = model.model.layers[l]

        norms = []
        for _ in range(num_projections):
            # Random unit vector
            v = torch.randn(1, d, device=device)
            v = v / v.norm(dim=-1, keepdim=True)

            # Finite-difference: Jₗv ≈ (T(h*+ηv) - T(h*)) / η
            # T^(l)(h) = h + f^(l)(h), so Jₗv = v + (f(h*+ηv) - f(h*))/η
            with torch.no_grad():
                h_plus = h_star + fd_step * v

                # Forward through layer (simplified — actual implementation
                # depends on model architecture)
                f_h = _layer_residual(layer_module, h_star)
                f_h_plus = _layer_residual(layer_module, h_plus)

                jv = v + (f_h_plus - f_h) / fd_step  # Jₗv = v + J_f·v

            # Transversal projection
            p_perp_jv = project_transversal(jv, u_star)
            norms.append(p_perp_jv.norm(dim=-1).mean().item())

        norms = np.array(norms)
        per_projection.append(norms)
        contraction_rates.append(norms.max())

    return {
        "contraction_rates": np.array(contraction_rates),
        "flux_magnitudes": np.array(flux_magnitudes),
        "per_projection": np.array(per_projection),
    }


def _layer_residual(layer_module, h: torch.Tensor) -> torch.Tensor:
    """Compute f^(l)(h) for a single transformer layer.

    This is architecture-specific. For Qwen2.5 / LLaMA-style models:
    f^(l)(h) = attn(LN₁(h)) + ffn(LN₂(h + attn(LN₁(h))))
    """
    residual = h
    # Attention sublayer
    hidden = layer_module.input_layernorm(h)
    attn_out, _, _ = layer_module.self_attn(
        hidden_states=hidden.unsqueeze(1),
        attention_mask=None,
        position_ids=None,
    )
    attn_out = attn_out.squeeze(1)
    h_mid = residual + attn_out
    # FFN sublayer
    hidden2 = layer_module.post_attention_layernorm(h_mid)
    ffn_out = layer_module.mlp(hidden2)
    return attn_out + ffn_out
