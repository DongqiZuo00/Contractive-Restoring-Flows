"""Post-hoc measurement of ‖P⊥Jₗ‖₂ per layer."""

import argparse
import json
import torch
import numpy as np
import sys
sys.path.insert(0, "src")

from transformers import AutoModelForCausalLM
from jacobian_measure import measure_transversal_contraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--trajectories", type=str, required=True)
    parser.add_argument("--num_projections", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output", type=str, default="contraction_measurements.json")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Load trajectories and measure
    from pathlib import Path
    traj_files = sorted(Path(args.trajectories).glob("trajectory_*.pt"))[:args.num_samples]

    all_rates = []
    all_flux = []
    for tf in traj_files:
        data = torch.load(tf, map_location="cuda")
        result = measure_transversal_contraction(
            model, data["h_stars"], data["delta_stars"],
            num_projections=args.num_projections,
        )
        all_rates.append(result["contraction_rates"])
        all_flux.append(result["flux_magnitudes"])

    rates = np.stack(all_rates)
    output = {
        "rates": rates.mean(axis=0).tolist(),
        "rates_std": rates.std(axis=0).tolist(),
        "flux": np.stack(all_flux).mean(axis=0).tolist(),
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved measurements to {args.output}")


if __name__ == "__main__":
    main()
