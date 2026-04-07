# Contractive Restoring Flows

**Robust Reasoning Distillation via Orbital Stability**

> We develop a mathematical framework for reasoning distillation grounded in dynamical systems theory. By modeling the Transformer's residual stream as a discrete dynamical system, we derive the Contractive Restoring Flow (CRF) loss from first principles and prove that it achieves strict transversal contraction at rate γ = 1−α, tangential neutrality, and a global basin of attraction with bounded steady-state error. All core results have been formally verified in Lean 4 with zero `sorry`.

## Key Idea

Standard trajectory matching (SFT / hidden-state matching) controls the student's residual function at a single point on the teacher's trajectory but leaves the **Jacobian unconstrained**. This yields marginal stability (∥P⊥Jₗ∥₂ → 1), allowing transversal errors to accumulate across layers.

CRF adds a **restoring force**: the loss drives P⊥J_f → −αP⊥, which overcomes the identity contribution of the residual connection and achieves strict contraction ∥P⊥Jₗ∥₂ = 1 − α < 1.

```
L_CRF = E_δ [ ‖ P⊥(f_θ(h* + δ) − Δ*) + α P⊥δ ‖² ]
       = ‖P⊥rₗ‖²  +  σ² ‖P⊥(J_f + αI)‖²_F
         ─────────     ──────────────────────
         Transversal    Spectral Energy
           Drift        (Jacobian constraint)
```

## Repository Structure

```
crf-distillation/
├── lean/                    # Lean 4 formal verification
│   ├── CRF_Complete.lean    # Complete proof chain (0 sorry)
│   ├── lakefile.lean
│   └── lean-toolchain
├── src/
│   ├── crf_loss.py          # CRF loss implementation
│   ├── projector.py         # Dimension alignment (d_T ≠ d_S)
│   ├── trainer.py           # CRF distillation trainer
│   ├── dataset.py           # Hidden-state trajectory dataset
│   └── jacobian_measure.py  # Post-hoc ‖P⊥Jₗ‖₂ measurement
├── configs/
│   ├── local_verify.yaml     # Single-GPU verification (3B→0.5B)
│   ├── qwen2.5_7b.yaml      # Regime A: 10× compression
│   ├── qwen2.5_3b.yaml      # Regime B: 20× compression
│   └── qwen2.5_0.5b.yaml    # Regime C: 140× compression
├── scripts/
│   ├── run_local.py          # End-to-end single-GPU pipeline
│   ├── collect_trajectories.py
│   ├── train.py
│   ├── eval.py
│   └── measure_contraction.py
└── figures/
    ├── plot_contraction.py   # Figure 3: ‖P⊥Jₗ‖₂ per layer
    └── plot_basin.py         # Figure 4: noise injection curves
```

## Quick Start: Local Verification (Single GPU)

**Requires**: One GPU with ≥ 12GB VRAM (e.g., RTX 5070/4070 Ti/3090).

This uses Qwen2.5-3B as teacher and Qwen2.5-0.5B as student to verify the full pipeline on a single GPU. It does NOT reproduce paper numbers (which use 72B teacher), but confirms that:
- CRF loss computes and backprops correctly
- Training converges (both SFT and CRF losses decrease)
- ‖P⊥Jₗ‖₂ moves toward 1−α on active layers
- The dimension projector and layer mapping work

```bash
pip install -r requirements.txt

# Run the full pipeline (collect → train → measure)
python scripts/run_local.py --stage all

# Or run stages separately:
python scripts/run_local.py --stage collect   # ~10 min, needs ~8GB VRAM
python scripts/run_local.py --stage train     # ~20 min, needs ~5GB VRAM
python scripts/run_local.py --stage measure   # ~5 min,  needs ~2GB VRAM
```

**What to expect**: After training, the measurement stage prints a per-layer table of ‖P⊥Jₗ‖₂. On active layers, values should be measurably below 1.0 (toward the 1−α = 0.90 target), while passive layers remain near 1.0.

### Cross-Dimension Distillation

When teacher and student have different hidden dimensions (d_T ≠ d_S), a learned linear projector maps teacher states into the student's space. Layer counts are aligned via uniform spacing. See `src/projector.py`.

## Full-Scale Reproduction (Multi-GPU)

### Lean 4 Verification (CPU only)

```bash
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh
cd lean && lake build
```

### 2. Collect Teacher Trajectories

```bash
python scripts/collect_trajectories.py \
  --teacher Qwen/Qwen2.5-72B \
  --data data/aime_train.jsonl \
  --output trajectories/qwen72b/
```

### 3. Train

```bash
python scripts/train.py --config configs/qwen2.5_7b.yaml
```

### 4. Measure Contraction Rate

```bash
python scripts/measure_contraction.py \
  --model checkpoints/crf_7b/ \
  --trajectories trajectories/qwen72b/ \
  --num_projections 64
```

## Hyperparameters

| Symbol | Range | Rationale |
|--------|-------|-----------|
| α | [0.05, 0.20] | Must exceed max(2κJ_max, H_max·δ_SFT) |
| σ | [0.005, 0.02] | Taylor validity: σ‖J_f‖₂ ≪ 1 |
| τ | 5th %ile of {μₗ} | Exclude clearly passive layers |
| ε_gate | τ/5 | Sharp but smooth gating transition |
| λ | [0.1, 1.0] | Balance with SFT loss |

## Results

### Extreme Compression (Qwen2.5-72B → Qwen2.5-{7B, 3B, 0.5B})

| Method | 7B Avg.Δ | 3B Avg.Δ | 0.5B Avg.Δ |
|--------|----------|----------|------------|
| CoT KD | +1.8 | +1.8 | +1.5 |
| DASD | +4.9 | +4.0 | +6.1 |
| CRF | **+5.5** | **+4.7** | **+6.6** |

CRF's relative gains grow as model size decreases—orbital stability specifically benefits fragile students where per-layer error accumulation is otherwise unchecked.

## Citation

```bibtex
@article{crf2026,
  title={Contractive Restoring Flows: Robust Reasoning Distillation via Orbital Stability},
  author={Zuo, Dongqi and Wang, Xinyu},
  year={2026}
}
```

## License

MIT
