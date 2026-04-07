"""
End-to-end CRF distillation pipeline for local verification.

Runs on a single RTX 5070 (12GB VRAM):
  Teacher: Qwen2.5-3B (bf16 ≈ 6GB)
  Student: Qwen2.5-0.5B (bf16 ≈ 1GB, +optimizer ≈ 3GB)

Usage:
  python scripts/run_local.py --stage all
  python scripts/run_local.py --stage collect
  python scripts/run_local.py --stage train
  python scripts/run_local.py --stage measure

The pipeline:
  1. collect  — Run 3B teacher on a small dataset, save (h*, Δ*) to disk
  2. train    — Train 0.5B student with CRF loss (+ SFT)
  3. measure  — Measure ‖P⊥Jₗ‖₂ per layer on the trained student
"""

import argparse
import os
import sys
import json
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import math

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TEACHER_MODEL = "Qwen/Qwen2.5-3B"
STUDENT_MODEL = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "outputs/local_verify"
TRAJ_DIR = "outputs/local_verify/trajectories"

# CRF hyperparameters
ALPHA = 0.10        # Contraction coefficient
SIGMA = 0.01        # Probe noise scale
LAMBDA_CRF = 0.5    # CRF loss weight
LR = 2e-5
NUM_EPOCHS = 2
BATCH_SIZE = 1       # Gradient accumulation compensates
GRAD_ACCUM = 4
MAX_SEQ_LEN = 512    # Short for local verification
NUM_TRAIN_SAMPLES = 50
NUM_EVAL_SAMPLES = 10

# Synthetic reasoning prompts for verification
PROMPTS = [
    "Solve step by step: What is the sum of all integers from 1 to 100?",
    "Solve step by step: If 3x + 7 = 22, what is x?",
    "Solve step by step: A triangle has sides 3, 4, and 5. What is its area?",
    "Solve step by step: What is 17 × 23?",
    "Solve step by step: Find the GCD of 48 and 36.",
    "Solve step by step: What is the derivative of x^3 + 2x^2 - 5x + 1?",
    "Solve step by step: If a car travels 60 km/h for 2.5 hours, how far does it go?",
    "Solve step by step: What is 2^10?",
    "Solve step by step: Simplify the fraction 84/126.",
    "Solve step by step: What is the sum of interior angles of a hexagon?",
    "Solve step by step: Find x if 2^x = 64.",
    "Solve step by step: What is 15% of 240?",
    "Solve step by step: A rectangle has area 48 and width 6. What is the length?",
    "Solve step by step: What is the next prime after 29?",
    "Solve step by step: Convert 0.375 to a fraction.",
    "Solve step by step: What is the volume of a cube with side length 7?",
    "Solve step by step: If f(x) = x^2 - 4x + 3, find f(5).",
    "Solve step by step: What is the LCM of 12 and 18?",
    "Solve step by step: A circle has radius 10. What is its circumference?",
    "Solve step by step: Solve the system: x + y = 10, x - y = 4.",
    "Solve step by step: What is 7! (7 factorial)?",
    "Solve step by step: Find the median of {3, 7, 1, 9, 5, 2, 8}.",
    "Solve step by step: What is log base 2 of 128?",
    "Solve step by step: A train travels 300 km in 4 hours. What is its speed in m/s?",
    "Solve step by step: What is the 10th Fibonacci number?",
    "Solve step by step: Simplify: (x^2 - 9) / (x + 3).",
    "Solve step by step: What is the area of a circle with diameter 14?",
    "Solve step by step: If 5 workers can build a wall in 12 days, how many days for 3 workers?",
    "Solve step by step: What is the square root of 1764?",
    "Solve step by step: Find the slope of the line passing through (2,3) and (5,9).",
    "Solve step by step: What is C(10,3)?",
    "Solve step by step: Convert 255 from decimal to binary.",
    "Solve step by step: What is the sum of the first 20 odd numbers?",
    "Solve step by step: A cone has radius 3 and height 4. What is its volume?",
    "Solve step by step: If sin(θ) = 0.6, what is cos(θ)?",
    "Solve step by step: What is 3^5 - 2^5?",
    "Solve step by step: Find two numbers whose sum is 50 and product is 600.",
    "Solve step by step: What is the perimeter of a regular pentagon with side 8?",
    "Solve step by step: Evaluate the integral of 2x dx from 0 to 5.",
    "Solve step by step: How many diagonals does a decagon have?",
    "Solve step by step: What is 999 × 1001?",
    "Solve step by step: Find the value of i^42 where i is imaginary unit.",
    "Solve step by step: What is 1/2 + 1/3 + 1/6?",
    "Solve step by step: A sphere has volume 36π. What is its radius?",
    "Solve step by step: What is the determinant of matrix [[3,1],[2,4]]?",
    "Solve step by step: How many ways to arrange the letters in MISSISSIPPI?",
    "Solve step by step: If x^2 + y^2 = 25 and x = 3, what are the possible values of y?",
    "Solve step by step: What is the sum of 1/1 + 1/2 + 1/4 + 1/8 + ... (infinite series)?",
    "Solve step by step: Find the distance between points (1,2,3) and (4,6,3).",
    "Solve step by step: What is the remainder when 2^100 is divided by 7?",
    "Solve step by step: Solve |2x - 5| = 11.",
    "Solve step by step: What is the probability of rolling a sum of 7 with two dice?",
    "Solve step by step: Find the vertex of the parabola y = x^2 - 6x + 5.",
    "Solve step by step: What is 0.1 + 0.2 in exact arithmetic?",
    "Solve step by step: A right triangle has legs 5 and 12. What is the hypotenuse?",
    "Solve step by step: How many zeros does 100! have at the end?",
    "Solve step by step: What is the arithmetic mean of all prime numbers less than 20?",
    "Solve step by step: Expand (a + b)^4.",
    "Solve step by step: Find the inverse of 7 modulo 11.",
    "Solve step by step: What is the surface area of a cylinder with r=3, h=10?",
]


# ===========================================================================
# Stage 1: Collect Teacher Trajectories
# ===========================================================================

def collect_trajectories():
    """Run teacher forward pass, save hidden states and residual fluxes."""
    print("=" * 60)
    print("Stage 1: Collecting teacher trajectories")
    print(f"  Teacher: {TEACHER_MODEL}")
    print(f"  Prompts: {NUM_TRAIN_SAMPLES + NUM_EVAL_SAMPLES}")
    print("=" * 60)

    os.makedirs(TRAJ_DIR, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading teacher model...")
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Get model dimensions
    config = model.config
    d_teacher = config.hidden_size
    L_teacher = config.num_hidden_layers
    print(f"  d_teacher={d_teacher}, L_teacher={L_teacher}")

    prompts = PROMPTS[: NUM_TRAIN_SAMPLES + NUM_EVAL_SAMPLES]

    for i, prompt in enumerate(tqdm(prompts, desc="Collecting")):
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_SEQ_LEN // 2,
            truncation=True,
        ).to("cuda")

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )

        # outputs.hidden_states: tuple of (L+1) tensors, each (1, T, d)
        hidden_states = outputs.hidden_states  # (L+1) tensors

        h_stars = []
        delta_stars = []
        for l in range(L_teacher):
            h_l = hidden_states[l][0].cpu().to(torch.float16)      # (T, d)
            h_l1 = hidden_states[l + 1][0].cpu().to(torch.float16)  # (T, d)
            delta = (h_l1.float() - h_l.float()).to(torch.float16)
            h_stars.append(h_l)
            delta_stars.append(delta)

        # Also generate teacher response for SFT
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=MAX_SEQ_LEN // 2,
                do_sample=False,
            )

        save_data = {
            "prompt": prompt,
            "input_ids": inputs["input_ids"][0].cpu(),
            "gen_ids": gen_ids[0].cpu(),
            "h_stars": h_stars,
            "delta_stars": delta_stars,
            "d_teacher": d_teacher,
            "L_teacher": L_teacher,
        }

        torch.save(save_data, os.path.join(TRAJ_DIR, f"traj_{i:04d}.pt"))

    # Save metadata
    meta = {
        "teacher_model": TEACHER_MODEL,
        "d_teacher": d_teacher,
        "L_teacher": L_teacher,
        "num_samples": len(prompts),
        "max_seq_len": MAX_SEQ_LEN,
    }
    with open(os.path.join(TRAJ_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(prompts)} trajectories to {TRAJ_DIR}")

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()


# ===========================================================================
# Stage 2: Train Student with CRF
# ===========================================================================

class TrajectoryDataset(Dataset):
    """Load pre-collected trajectories for CRF training."""

    def __init__(self, traj_dir: str, split: str = "train"):
        self.traj_dir = Path(traj_dir)
        all_files = sorted(self.traj_dir.glob("traj_*.pt"))
        if split == "train":
            self.files = all_files[:NUM_TRAIN_SAMPLES]
        else:
            self.files = all_files[NUM_TRAIN_SAMPLES : NUM_TRAIN_SAMPLES + NUM_EVAL_SAMPLES]
        print(f"  [{split}] {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], map_location="cpu", weights_only=False)


def project_transversal(x: torch.Tensor, u_star: torch.Tensor) -> torch.Tensor:
    """P⊥x = x - (x·u*)u*"""
    coeff = (x * u_star).sum(dim=-1, keepdim=True)
    return x - coeff * u_star


def compute_crf_loss_single_layer(
    student_layer: nn.Module,
    h_star: torch.Tensor,       # (T, d_s) — projected teacher state
    delta_star: torch.Tensor,   # (T, d_s) — projected teacher flux
    alpha: float,
    sigma: float,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """CRF loss for one layer: Eq. (5).

    L^(l)_CRF = E_δ[‖P⊥(f_θ(h*+δ) - Δ*) + α P⊥δ‖²]
    """
    T, d = h_star.shape

    # Tangent direction
    mu = delta_star.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    u_star = delta_star / mu

    # Kinetic gate: skip passive layers (mean flux too small)
    mean_mu = mu.mean().item()
    if mean_mu < 0.1:
        return torch.tensor(0.0, device=h_star.device, requires_grad=True)

    # Gate weight
    tau = 0.5  # Will be recalibrated
    w = torch.sigmoid((mu.squeeze(-1) - tau) / (tau / 5 + 1e-8)).mean().item()
    if w < 0.05:
        return torch.tensor(0.0, device=h_star.device, requires_grad=True)

    # Sample probe noise
    delta = torch.randn_like(h_star) * sigma

    # Student response at perturbed input
    h_perturbed = h_star.detach() + delta

    # Run through student layer
    # The student layer expects (batch, seq, d) format
    h_in = h_perturbed.unsqueeze(0)  # (1, T, d)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        layer_out = student_layer(
            h_in,
            position_ids=position_ids,
        )
    h_out = layer_out[0].squeeze(0).float()  # (T, d)

    # Residual: f_θ(h) = T(h) - h = layer_out - h_in
    f_student = h_out - h_perturbed.float()

    # CRF loss with restoring force
    fitting_error = f_student - delta_star.float()
    e_perp = project_transversal(fitting_error, u_star.float())
    delta_perp = project_transversal(delta.float(), u_star.float())
    e_total = e_perp + alpha * delta_perp

    loss = (e_total ** 2).sum(dim=-1).mean() * w
    return loss


def train_student():
    """Train 0.5B student with CRF + SFT."""
    print("=" * 60)
    print("Stage 2: Training student with CRF")
    print(f"  Student: {STUDENT_MODEL}")
    print(f"  α={ALPHA}, σ={SIGMA}, λ={LAMBDA_CRF}")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load metadata
    with open(os.path.join(TRAJ_DIR, "meta.json")) as f:
        meta = json.load(f)
    d_teacher = meta["d_teacher"]
    L_teacher = meta["L_teacher"]

    # Load student
    print("Loading student model...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    d_student = student.config.hidden_size
    L_student = student.config.num_hidden_layers
    print(f"  d_student={d_student}, L_student={L_student}")
    print(f"  d_teacher={d_teacher}, L_teacher={L_teacher}")

    # Enable gradient checkpointing to save VRAM
    student.gradient_checkpointing_enable()

    # Build dimension projector
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from projector import DimensionProjector, build_layer_mapping

    projector = DimensionProjector(d_teacher, d_student).cuda().to(torch.bfloat16)
    layer_mapping = build_layer_mapping(L_teacher, L_student)
    print(f"  Layer mapping (student→teacher): {layer_mapping[:5]}...{layer_mapping[-3:]}")

    # Optimizer: student params + projector params
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()),
        lr=LR,
        weight_decay=0.01,
    )

    # Dataset
    train_dataset = TrajectoryDataset(TRAJ_DIR, split="train")

    save_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    student.train()
    global_step = 0
    log_interval = 5

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        epoch_loss_sft = 0.0
        epoch_loss_crf = 0.0
        epoch_steps = 0

        for sample_idx in range(len(train_dataset)):
            sample = train_dataset[sample_idx]

            # --- SFT loss ---
            gen_ids = sample["gen_ids"].unsqueeze(0).cuda()
            T_gen = min(gen_ids.shape[1], MAX_SEQ_LEN)
            gen_ids = gen_ids[:, :T_gen]

            if T_gen < 4:
                continue

            attn_mask = torch.ones_like(gen_ids)
            labels = gen_ids.clone()
            labels[:, :-1] = gen_ids[:, 1:]
            labels[:, -1] = -100

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = student(
                    input_ids=gen_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                )
            loss_sft = outputs.loss

            # --- CRF loss ---
            h_stars_teacher = sample["h_stars"]       # List of L_T tensors (T_prompt, d_T)
            delta_stars_teacher = sample["delta_stars"]

            # Project to student space
            T_prompt = h_stars_teacher[0].shape[0]
            loss_crf = torch.tensor(0.0, device="cuda")
            num_active_layers = 0

            position_ids = torch.arange(T_prompt, device="cuda").unsqueeze(0)

            for s_layer_idx in range(L_student):
                t_layer_idx = layer_mapping[s_layer_idx]

                h_t = h_stars_teacher[t_layer_idx].cuda()          # (T, d_T)
                d_t = delta_stars_teacher[t_layer_idx].cuda()      # (T, d_T)

                # Project to student dimension
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    h_proj = projector(h_t).float()     # (T, d_S)
                    d_proj = projector(d_t).float()     # (T, d_S)

                # Get student layer
                student_layer = student.model.layers[s_layer_idx]

                layer_crf = compute_crf_loss_single_layer(
                    student_layer=student_layer,
                    h_star=h_proj,
                    delta_star=d_proj,
                    alpha=ALPHA,
                    sigma=SIGMA,
                    position_ids=position_ids,
                )

                if layer_crf.item() > 0:
                    loss_crf = loss_crf + layer_crf
                    num_active_layers += 1

            if num_active_layers > 0:
                loss_crf = loss_crf / num_active_layers

            # Total loss
            loss_total = loss_sft + LAMBDA_CRF * loss_crf
            loss_scaled = loss_total / GRAD_ACCUM
            loss_scaled.backward()

            epoch_loss_sft += loss_sft.item()
            epoch_loss_crf += loss_crf.item()
            epoch_steps += 1

            if (sample_idx + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(projector.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_interval == 0:
                    avg_sft = epoch_loss_sft / epoch_steps
                    avg_crf = epoch_loss_crf / epoch_steps
                    mem = torch.cuda.max_memory_allocated() / 1e9
                    print(
                        f"  step {global_step:4d} | "
                        f"SFT={avg_sft:.4f} | CRF={avg_crf:.4f} | "
                        f"active_layers={num_active_layers} | "
                        f"VRAM={mem:.1f}GB"
                    )

        # Flush remaining gradients
        if epoch_steps % GRAD_ACCUM != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_sft = epoch_loss_sft / max(epoch_steps, 1)
        avg_crf = epoch_loss_crf / max(epoch_steps, 1)
        print(f"  Epoch {epoch+1} done: SFT={avg_sft:.4f}, CRF={avg_crf:.4f}")

    # Save
    print(f"Saving checkpoint to {save_dir}")
    student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    torch.save(projector.state_dict(), os.path.join(save_dir, "projector.pt"))
    with open(os.path.join(save_dir, "layer_mapping.json"), "w") as f:
        json.dump({"mapping": layer_mapping}, f)

    print("Training complete.")

    del student, projector, optimizer
    gc.collect()
    torch.cuda.empty_cache()


# ===========================================================================
# Stage 3: Measure Contraction Rate
# ===========================================================================

@torch.no_grad()
def measure_contraction():
    """Measure ‖P⊥Jₗ‖₂ per layer via Hutchinson estimator."""
    print("=" * 60)
    print("Stage 3: Measuring transversal contraction rate")
    print("=" * 60)

    from transformers import AutoModelForCausalLM

    save_dir = os.path.join(OUTPUT_DIR, "checkpoints")

    with open(os.path.join(TRAJ_DIR, "meta.json")) as f:
        meta = json.load(f)
    d_teacher = meta["d_teacher"]

    with open(os.path.join(save_dir, "layer_mapping.json")) as f:
        layer_mapping = json.load(f)["mapping"]

    # Load student
    print("Loading trained student...")
    student = AutoModelForCausalLM.from_pretrained(
        save_dir,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    student.eval()
    d_student = student.config.hidden_size
    L_student = student.config.num_hidden_layers

    # Load projector
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from projector import DimensionProjector

    projector = DimensionProjector(d_teacher, d_student).cuda()
    projector.load_state_dict(
        torch.load(os.path.join(save_dir, "projector.pt"), map_location="cuda", weights_only=True)
    )
    projector.eval()

    # Load eval trajectories
    eval_dataset = TrajectoryDataset(TRAJ_DIR, split="eval")

    K = 32  # Number of random projections (reduced for speed)
    FD_STEP = 1e-3

    all_rates = [[] for _ in range(L_student)]
    all_flux = [[] for _ in range(L_student)]

    num_eval = min(len(eval_dataset), 5)
    for sample_idx in range(num_eval):
        sample = eval_dataset[sample_idx]
        h_stars_t = sample["h_stars"]
        delta_stars_t = sample["delta_stars"]

        T = h_stars_t[0].shape[0]
        # Use a single token position for measurement (middle)
        t_pos = T // 2

        for s_idx in range(L_student):
            t_idx = layer_mapping[s_idx]
            h_t = h_stars_t[t_idx][t_pos:t_pos+1].cuda()
            d_t = delta_stars_t[t_idx][t_pos:t_pos+1].cuda()

            # Project
            h_proj = projector(h_t).float()
            d_proj = projector(d_t).float()

            mu = d_proj.norm().item()
            all_flux[s_idx].append(mu)

            if mu < 0.01:
                all_rates[s_idx].append(1.0)
                continue

            u_star = d_proj / d_proj.norm(dim=-1, keepdim=True)

            student_layer = student.model.layers[s_idx]
            position_ids = torch.tensor([[t_pos]], device="cuda")

            norms = []
            for _ in range(K):
                v = torch.randn(1, d_student, device="cuda")
                v = v / v.norm(dim=-1, keepdim=True)

                # T(h) = h + f(h), computed via layer forward
                h_in = h_proj.unsqueeze(0).to(torch.bfloat16)
                h_plus_in = (h_proj + FD_STEP * v).unsqueeze(0).to(torch.bfloat16)

                out_base = student_layer(h_in, position_ids=position_ids)[0].squeeze(0).float()
                out_plus = student_layer(h_plus_in, position_ids=position_ids)[0].squeeze(0).float()

                # Jₗv ≈ (T(h+ηv) - T(h)) / η
                jv = (out_plus - out_base) / FD_STEP

                # P⊥(Jₗv)
                p_perp_jv = project_transversal(jv, u_star)
                norms.append(p_perp_jv.norm().item())

            all_rates[s_idx].append(max(norms))

    # Print results
    print("\n" + "=" * 60)
    print("Contraction measurement results:")
    print(f"{'Layer':>6} {'‖P⊥Jₗ‖₂':>10} {'μₗ':>8} {'Status':>12}")
    print("-" * 40)

    rates_mean = []
    flux_mean = []
    for l in range(L_student):
        r = sum(all_rates[l]) / len(all_rates[l]) if all_rates[l] else 1.0
        f = sum(all_flux[l]) / len(all_flux[l]) if all_flux[l] else 0.0
        rates_mean.append(r)
        flux_mean.append(f)

        if f < 0.1:
            status = "passive"
        elif r < 1.0 - ALPHA / 2:
            status = "contracting ✓"
        elif r < 1.0:
            status = "mild contraction"
        else:
            status = "marginal"

        print(f"  {l:4d}   {r:8.4f}   {f:6.2f}   {status}")

    active_rates = [r for r, f in zip(rates_mean, flux_mean) if f >= 0.1]
    if active_rates:
        mean_active = sum(active_rates) / len(active_rates)
        print(f"\nMean ‖P⊥Jₗ‖₂ (active layers): {mean_active:.4f}")
        print(f"Theory prediction (1−α):       {1 - ALPHA:.4f}")
        print(f"Gap:                           {mean_active - (1-ALPHA):+.4f}")

    # Save results
    results = {
        "rates": rates_mean,
        "flux": flux_mean,
        "alpha": ALPHA,
        "theory_target": 1 - ALPHA,
        "num_projections": K,
    }
    results_path = os.path.join(OUTPUT_DIR, "contraction_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="CRF local verification pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "collect", "train", "measure"],
        help="Which stage to run",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.stage in ("all", "collect"):
        collect_trajectories()

    if args.stage in ("all", "train"):
        train_student()

    if args.stage in ("all", "measure"):
        measure_contraction()

    if args.stage == "all":
        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print(f"  Trajectories: {TRAJ_DIR}/")
        print(f"  Checkpoint:   {OUTPUT_DIR}/checkpoints/")
        print(f"  Results:      {OUTPUT_DIR}/contraction_results.json")
        print("=" * 60)


if __name__ == "__main__":
    main()
