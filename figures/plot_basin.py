"""
Figure 4: AIME pass@1 under hidden-state noise injection.

1×2 layout: (a) 7B student, (b) 0.5B student.
Validates the Basin of Attraction theorem: CRF maintains the flattest
degradation curve as noise magnitude increases.
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import json


def plot_basin(data_path: str, save_path: str = "figure4_basin.pdf"):
    with open(data_path) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    noise_levels = np.array(data["noise_levels"])

    methods = {
        "SFT": ("gray", "--"),
        "CoT KD": ("tab:orange", "-."),
        "GKD": ("tab:brown", ":"),
        "DASD": ("tab:red", "-"),
        "CRF (ours)": ("tab:blue", "-"),
    }

    for panel, (ax, scale) in enumerate([(ax1, "7b"), (ax2, "0.5b")]):
        for label, (color, ls) in methods.items():
            key = f"{label.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{scale}"
            if key in data:
                scores = np.array(data[key])
                ax.plot(noise_levels, scores, color=color, ls=ls, lw=2.0,
                        marker="o", markersize=4, label=label)

        ax.set_xlabel(r"Noise magnitude $\sigma$", fontsize=11)
        ax.set_ylabel("AIME pass@1 (%)", fontsize=11)
        titles = {"7b": "(a) 7B student (10× compression)",
                  "0.5b": "(b) 0.5B student (140× compression)"}
        ax.set_title(titles[scale], fontsize=10)
        ax.set_xlim(0, 0.52)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="figure4_basin.pdf")
    args = parser.parse_args()
    plot_basin(args.data, args.output)
