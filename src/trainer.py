"""
CRF Distillation Trainer.

Two-phase training procedure (Algorithm 1):
  Phase I (no gradient): Extract teacher geometry — residual flux, tangent direction,
                          kinetic gate weights.
  Phase II (gradient):   Subject student to stochastic stability test at each active
                          layer; accumulate CRF loss alongside SFT.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import os
import yaml
import logging

from crf_loss import GatedCRFLoss, compute_transversal_projection

logger = logging.getLogger(__name__)


class CRFTrainer:
    def __init__(self, config: dict):
        self.config = config

        # CRF hyperparameters
        self.alpha = config.get("alpha", 0.10)
        self.sigma = config.get("sigma", 0.01)
        self.lambda_crf = config.get("lambda_crf", 0.5)
        self.lr = config.get("lr", 1e-5)
        self.num_epochs = config.get("num_epochs", 3)
        self.grad_accum_steps = config.get("grad_accum_steps", 4)

        # Kinetic gate (calibrated from data)
        self.tau = config.get("tau", None)
        self.eps_gate = config.get("eps_gate", None)

        # Model paths
        self.student_path = config["student_model"]
        self.trajectory_dir = config["trajectory_dir"]
        self.output_dir = config["output_dir"]

        # CRF loss module
        self.crf_loss = GatedCRFLoss(
            alpha=self.alpha,
            sigma=self.sigma,
            tau=self.tau,
            eps_gate=self.eps_gate,
        )

    def load_student(self):
        """Load student model."""
        logger.info(f"Loading student model from {self.student_path}")
        self.student = AutoModelForCausalLM.from_pretrained(
            self.student_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.student_path)
        return self.student

    def get_student_residual_fns(self, layer_indices: list):
        """Extract residual block callables from the student model.

        Each callable computes f^(l)_θ(h) = attention(h) + ffn(attention(h) + h) - h,
        i.e. the residual function whose Jacobian we want to control.

        Returns:
            List of callables, one per layer index.
        """
        fns = []
        for l in layer_indices:
            layer = self.student.model.layers[l]

            def make_residual_fn(layer_module):
                def residual_fn(h):
                    # Standard Pre-LN transformer residual block
                    # f^(l)(h) = T^(l)(h) - h = attention + FFN
                    residual = h
                    # Self-attention sublayer
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
                    return attn_out + ffn_out  # f^(l)(h) = total residual
                return residual_fn
            fns.append(make_residual_fn(layer))
        return fns

    def calibrate_kinetic_gate(self, trajectory_dataset, num_samples: int = 200):
        """Calibrate τ from teacher flux magnitudes.

        Run over a subset of trajectories and set τ = P5({μ_l}).
        """
        all_mu = []
        for i, sample in enumerate(trajectory_dataset):
            if i >= num_samples:
                break
            delta_stars = sample["delta_stars"]  # list of (d,) tensors
            for ds in delta_stars:
                all_mu.append(ds.norm().item())

        all_mu = torch.tensor(all_mu)
        self.crf_loss.calibrate_gate(all_mu)
        self.tau = self.crf_loss.tau
        self.eps_gate = self.crf_loss.eps_gate

    def train_step(self, batch) -> dict:
        """One CRF training step (Algorithm 1).

        Args:
            batch: Dict with keys:
                - input_ids, attention_mask, labels: for SFT
                - h_stars: list of L tensors (B, d) — teacher hidden states
                - delta_stars: list of L tensors (B, d) — teacher residual fluxes

        Returns:
            Dict with loss components.
        """
        # Phase III: SFT loss
        outputs = self.student(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss_sft = outputs.loss

        # Phase I + II: CRF loss (teacher geometry + student stability test)
        h_stars = batch["h_stars"]
        delta_stars = batch["delta_stars"]
        L = len(h_stars)
        layer_indices = list(range(L))
        residual_fns = self.get_student_residual_fns(layer_indices)

        loss_crf = self.crf_loss(residual_fns, h_stars, delta_stars)

        # Total loss
        loss_total = loss_sft + self.lambda_crf * loss_crf

        return {
            "loss": loss_total,
            "loss_sft": loss_sft.item(),
            "loss_crf": loss_crf.item(),
        }

    def train(self, train_dataset, eval_dataset=None):
        """Full training loop."""
        self.load_student()
        self.student.train()
        self.student.cuda()

        # Calibrate kinetic gate
        if self.tau is None:
            logger.info("Calibrating kinetic gate from trajectory data...")
            self.calibrate_kinetic_gate(train_dataset)

        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.lr,
            weight_decay=0.01,
        )

        dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get("batch_size", 2),
            shuffle=True,
            num_workers=4,
        )

        global_step = 0
        for epoch in range(self.num_epochs):
            for step, batch in enumerate(dataloader):
                batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
                metrics = self.train_step(batch)

                loss = metrics["loss"] / self.grad_accum_steps
                loss.backward()

                if (step + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 100 == 0:
                        logger.info(
                            f"Step {global_step} | "
                            f"SFT: {metrics['loss_sft']:.4f} | "
                            f"CRF: {metrics['loss_crf']:.4f}"
                        )

            # Save checkpoint
            save_path = os.path.join(self.output_dir, f"checkpoint-epoch{epoch}")
            self.student.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    trainer = CRFTrainer(config)
    # trainer.train(train_dataset)  # requires dataset setup


if __name__ == "__main__":
    main()
