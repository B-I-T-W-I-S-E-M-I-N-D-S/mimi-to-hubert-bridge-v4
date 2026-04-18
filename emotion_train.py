"""
emotion_train.py — Entry point for multi-task emotion training
===============================================================
Single-GPU:
    python emotion_train.py --config config.yaml \
        --train-csv data/mead_train.csv \
        --val-csv   data/mead_val.csv

Multi-GPU (recommended):
    torchrun --nproc_per_node=4 emotion_train.py \
        --config config.yaml \
        --train-csv data/mead_train.csv \
        --val-csv   data/mead_val.csv

Resume:
    torchrun --nproc_per_node=4 emotion_train.py \
        --config config.yaml \
        --train-csv data/mead_train.csv \
        --val-csv   data/mead_val.csv \
        --resume checkpoints/emotion_bridge_best.pt

The CSV format expected is:
    filename,emotion
    ./audio/W037_happy_level_3_002.wav,happy
    ./audio/W040_sad_level_2_016.wav,sad

Split your CSV into train/val with:
    python preprocess_emotion.py --csv mead_labels.csv --val-frac 0.1

Override any config key:
    torchrun --nproc_per_node=4 emotion_train.py \
        --config config.yaml \
        --train-csv data/mead_train.csv \
        --overrides training.batch_size=16 emotion.num_classes=7
"""

import argparse
import logging
import os
import sys
import yaml

from emotion_trainer import EmotionTrainer


def override_cfg(cfg: dict, overrides: list):
    for kv in overrides:
        key, _, val = kv.partition("=")
        keys = key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                if val.lower() in ("true", "false"):
                    val = val.lower() == "true"
        node[keys[-1]] = val
        print(f"  Override: {key} = {val}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-task emotion training (HuBERT features + emotion labels)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",    default="config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--train-csv", required=True,
                        help="Path to training CSV (filename,emotion)")
    parser.add_argument("--val-csv",   default=None,
                        help="Path to validation CSV (optional)")
    parser.add_argument("--resume",    default=None,
                        help="Checkpoint path to resume from")
    parser.add_argument("--overrides", nargs="*", default=[],
                        help="Dot-path overrides, e.g. training.batch_size=16")
    args = parser.parse_args()

    global_rank = int(os.environ.get("RANK", 0))
    handlers    = [logging.StreamHandler(sys.stdout)]
    if global_rank == 0:
        handlers.append(logging.FileHandler("emotion_train.log"))

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  [rank%(process)d]  %(message)s",
        handlers= handlers,
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Inject defaults for emotion section if not already in config
    if "emotion" not in cfg:
        cfg["emotion"] = {
            "num_classes":     7,
            "dropout":         0.2,
            "hidden_dim":      256,
            "label_smoothing": 0.1,
        }
    if "strategy" not in cfg["training"]["loss_weights"]:
        cfg["training"]["loss_weights"]["strategy"] = "uncertainty"
    if "emotion" not in cfg["training"]["loss_weights"]:
        cfg["training"]["loss_weights"]["emotion"] = 1.0

    if args.overrides:
        override_cfg(cfg, args.overrides)

    trainer = EmotionTrainer(cfg, train_csv=args.train_csv, val_csv=args.val_csv)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
