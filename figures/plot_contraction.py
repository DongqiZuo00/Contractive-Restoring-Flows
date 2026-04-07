"""
Figure 3: Transversal contraction rate ‖P⊥Jₗ‖₂ and residual flux μₗ per layer.

Reproduces the main mechanistic validation figure from the paper.
Upper panel: ‖P⊥Jₗ‖₂ for SFT, hidden-state matching, CRF w/o restoring force,
             and CRF with α ∈ {0.05, 0.10, 0.20}.
Lower panel: Residual flux μₗ with active/passive layer classification.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import argparse
import json


def plot_contraction(data_path: str, save_path: str = "figure3_contraction.pdf"):
    """Plot contraction rates from measurement data.

    Args:
        data_path: Path to JSON with measurement results.
                   Expected keys: "sft", "hsm", "crf_no_restore", "crf_005",
                   "crf_010", "crf_020", each containing "rates" and "flux".
    """
    with open(data_path) as f:
        data = json.load(f)

    num_layers = len(data["sft"]["rates"])
    layers = np.arange(num_layers)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 5), height_ratios=[3, 1],
        sharex=True, gridspec_kw={"hspace": 0.08}
    )

    # ---- Upper panel: contraction rates ----
    methods = {
        "SFT": ("sft", "gray", "--", 1.5),
        "Hidden state matching": ("hsm", "tab:orange", "-.", 1.5),
        "CRF w/o restoring force": ("crf_no_restore", "tab:red", "-", 1.5),
        r"CRF $\alpha$=0.20": ("crf_020", "tab:green", "-", 2.0),
        r"CRF $\alpha$=0.10": ("crf_010", "tab:blue", "-", 2.0),
        r"CRF $\alpha$=0.05": ("crf_005", "tab:purple", "-", 2.0),
    }

    for label, (key, color, ls, lw) in methods.items():
        if key in data:
            rates = np.array(data[key]["rates"])
            if "rates_std" in data[key]:
                std = np.array(data[key]["rates_std"])
                ax1.fill_between(layers, rates - std, rates + std, alpha=0.15, color=color)
            ax1.plot(layers, rates, color=color, ls=ls, lw=lw, label=label)

    # Theory lines
    for alpha, color in [(0.05, "tab:purple"), (0.10, "tab:blue"), (0.20, "tab:green")]:
        ax1.axhline(1 - alpha, color=color, ls=":", lw=0.8, alpha=0.6)
        ax1.text(num_layers + 0.3, 1 - alpha, f"α={alpha:.2f}",
                 fontsize=7, color=color, va="center")

    ax1.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax1.text(num_layers + 0.3, 1.0, "γ=1\n(marginal)", fontsize=6, color="gray", va="center")

    ax1.set_ylabel(r"$\|P_\perp J_l\|_2$", fontsize=11)
    ax1.set_ylim(0.74, 1.16)
    ax1.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.9)

    # Shade passive layers
    if "flux" in data["sft"]:
        flux = np.array(data["sft"]["flux"])
        tau = np.percentile(flux, 5)
        passive_mask = flux < tau
        for l in layers[passive_mask]:
            ax1.axvspan(l - 0.5, l + 0.5, color="lightblue", alpha=0.15)

    # ---- Lower panel: residual flux ----
    if "flux" in data["sft"]:
        flux = np.array(data["sft"]["flux"])
        tau = np.percentile(flux, 5)
        colors = ["tab:blue" if f >= tau else "lightblue" for f in flux]
        ax2.bar(layers, flux, color=colors, width=0.7, edgecolor="none")
        ax2.axhline(tau, color="gray", ls="--", lw=1.0, label=f"Gate threshold τ")
        ax2.set_ylabel(r"Residual flux $\mu_l$", fontsize=10)
        ax2.set_xlabel("Layer index $l$", fontsize=11)

        # Legend for active/passive
        active_patch = mpatches.Patch(color="tab:blue", label="Active layer")
        passive_patch = mpatches.Patch(color="lightblue", label="Passive layer")
        ax2.legend(handles=[active_patch, passive_patch],
                   fontsize=7, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="figure3_contraction.pdf")
    args = parser.parse_args()
    plot_contraction(args.data, args.output)
