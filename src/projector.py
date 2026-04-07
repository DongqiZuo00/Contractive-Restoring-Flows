"""
Dimension alignment for cross-dimension CRF distillation.

When d_teacher ≠ d_student, we need:
  1. A learned linear projector W: R^{d_teacher} → R^{d_student}
  2. A layer mapping: which teacher layers correspond to which student layers

The projector maps teacher hidden states and residual fluxes into the student's
space before computing the CRF loss. The CRF theory applies in the projected
space: the contraction guarantee ‖P⊥Jₗ‖₂ = 1−α holds in R^{d_student}.
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple


class DimensionProjector(nn.Module):
    """Learned linear projector from teacher space to student space.

    Maps h*_teacher ∈ R^{d_T} → h*_student ∈ R^{d_S} via a trained linear layer.
    The projector is trained jointly with the student during CRF distillation.
    """

    def __init__(self, d_teacher: int, d_student: int):
        super().__init__()
        self.d_teacher = d_teacher
        self.d_student = d_student

        if d_teacher == d_student:
            self.proj = None  # Identity — no projection needed
        else:
            self.proj = nn.Linear(d_teacher, d_student, bias=False)
            # Initialize near-orthogonal for stability
            nn.init.orthogonal_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project from teacher to student dimension.

        Args:
            x: shape (..., d_teacher)
        Returns:
            shape (..., d_student)
        """
        if self.proj is None:
            return x
        return self.proj(x.float()).to(x.dtype)


def build_layer_mapping(
    L_teacher: int,
    L_student: int,
) -> List[int]:
    """Map student layers to teacher layers via uniform spacing.

    For L_T=36 teacher layers and L_S=24 student layers,
    maps student layer i to teacher layer round(i * (L_T-1) / (L_S-1)).

    This ensures the first and last layers are always aligned,
    and intermediate layers are evenly distributed.

    Args:
        L_teacher: Number of teacher layers.
        L_student: Number of student layers.

    Returns:
        List of length L_student, where entry i is the teacher layer index.
    """
    if L_teacher == L_student:
        return list(range(L_student))

    mapping = []
    for i in range(L_student):
        t_idx = round(i * (L_teacher - 1) / (L_student - 1))
        mapping.append(t_idx)

    return mapping


def project_trajectory(
    h_stars_teacher: List[torch.Tensor],
    delta_stars_teacher: List[torch.Tensor],
    projector: DimensionProjector,
    layer_mapping: List[int],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Project teacher trajectory into student's hidden space.

    Args:
        h_stars_teacher: L_T tensors of shape (T, d_T).
        delta_stars_teacher: L_T tensors of shape (T, d_T).
        projector: DimensionProjector mapping d_T → d_S.
        layer_mapping: List mapping student layer → teacher layer.

    Returns:
        h_stars_projected: L_S tensors of shape (T, d_S).
        delta_stars_projected: L_S tensors of shape (T, d_S).
    """
    h_proj = []
    delta_proj = []

    for s_idx, t_idx in enumerate(layer_mapping):
        h = projector(h_stars_teacher[t_idx])
        d = projector(delta_stars_teacher[t_idx])
        h_proj.append(h)
        delta_proj.append(d)

    return h_proj, delta_proj
