"""
Collect teacher hidden-state trajectories for CRF training.

For each prompt, runs the teacher in autoregressive mode and stores:
  - h*_l: hidden states at each layer boundary
  - Δ*_l = h*_{l+1} - h*_l: residual flux
  - μ_l = ‖Δ*_l‖₂: flux magnitude

Storage: O(T × L × d) per example in bfloat16.
For 72B teacher (L=80, d=8192, T=1024): ~1.3 GB per example.
With activation checkpointing: ~40% of naive estimate.
"""

import argparse
import os
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def collect_trajectory(model, tokenizer, prompt: str, max_new_tokens: int = 1024):
    """Run teacher forward pass and collect hidden states at every layer.

    Returns:
        dict with keys:
          - h_stars: list of (T, d) tensors, one per layer
          - delta_stars: list of (T, d) tensors (residual flux)
          - tokens: generated token ids
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    # Collect hidden states from all generation steps
    # outputs.hidden_states is a tuple of (num_steps,) each containing
    # a tuple of (num_layers+1,) tensors of shape (B, seq_len, d)
    all_hidden_states = []
    for step_hidden in outputs.hidden_states:
        # step_hidden: tuple of (num_layers+1,) tensors
        # Each tensor shape: (1, 1, d) for autoregressive steps
        layer_states = [h.squeeze(0).squeeze(0) for h in step_hidden]
        all_hidden_states.append(layer_states)

    num_layers = len(all_hidden_states[0]) - 1  # exclude embedding layer
    T = len(all_hidden_states)

    # Reshape to per-layer trajectories
    h_stars = []
    delta_stars = []
    for l in range(num_layers):
        h_l = torch.stack([all_hidden_states[t][l] for t in range(T)])  # (T, d)
        h_l1 = torch.stack([all_hidden_states[t][l + 1] for t in range(T)])  # (T, d)
        delta = h_l1 - h_l  # Residual flux
        h_stars.append(h_l.to(torch.bfloat16))
        delta_stars.append(delta.to(torch.bfloat16))

    return {
        "h_stars": h_stars,
        "delta_stars": delta_stars,
        "tokens": outputs.sequences[0].cpu(),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect teacher trajectories for CRF")
    parser.add_argument("--teacher", type=str, required=True, help="Teacher model path")
    parser.add_argument("--data", type=str, required=True, help="JSONL with prompts")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading teacher model: {args.teacher}")
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)

    # Load prompts
    with open(args.data) as f:
        prompts = [json.loads(line) for line in f]

    print(f"Collecting trajectories for {len(prompts)} prompts...")
    for i, item in enumerate(tqdm(prompts)):
        prompt = item["prompt"] if "prompt" in item else item["question"]
        trajectory = collect_trajectory(model, tokenizer, prompt, args.max_new_tokens)

        save_path = os.path.join(args.output, f"trajectory_{i:06d}.pt")
        torch.save(trajectory, save_path)

    print(f"Done. Saved {len(prompts)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
