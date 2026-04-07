"""Entry point for CRF distillation training."""

import argparse
import yaml
import sys
sys.path.insert(0, "src")

from trainer import CRFTrainer
from dataset import TrajectoryDataset


def main():
    parser = argparse.ArgumentParser(description="CRF Distillation Training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Build dataset
    train_dataset = TrajectoryDataset(
        trajectory_dir=config["trajectory_dir"],
        tokenizer_name=config["student_model"],
        max_seq_len=config.get("max_seq_len", 4096),
    )

    # Train
    trainer = CRFTrainer(config)
    trainer.train(train_dataset)


if __name__ == "__main__":
    main()
