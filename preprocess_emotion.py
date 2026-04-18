"""
preprocess_emotion.py — Prepare MEAD CSV for emotion training
=============================================================
Splits mead_labels.csv into train/val CSVs and pre-extracts
Mimi tokens + HuBERT features (optional) for faster training.

Usage:
    # Split only (fast, no extraction):
    python preprocess_emotion.py \
        --csv mead_labels.csv \
        --val-frac 0.1 \
        --out-dir  data

    # Split + pre-extract features (requires GPU, takes time):
    torchrun --nproc_per_node=4 preprocess_emotion.py \
        --csv      mead_labels.csv \
        --val-frac 0.1 \
        --out-dir  data \
        --config   config.yaml \
        --preextract \
        --device   cuda

Outputs:
    data/mead_train.csv
    data/mead_val.csv
    data/cache/*.pt        (if --preextract)
"""

import argparse
import csv
import logging
import os
import random
from pathlib import Path

logger = logging.getLogger(__name__)


def split_csv(csv_path: str, val_frac: float, out_dir: str, seed: int = 42):
    """Split a mead_labels.csv into train/val CSVs."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)

    random.seed(seed)
    random.shuffle(rows)

    n_val   = max(1, int(len(rows) * val_frac))
    val_rows = rows[:n_val]
    trn_rows = rows[n_val:]

    def write_csv(path, data):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "emotion"])
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"  Written {len(data):5d} rows → {path}")

    train_path = os.path.join(out_dir, "mead_train.csv")
    val_path   = os.path.join(out_dir, "mead_val.csv")
    write_csv(train_path, trn_rows)
    write_csv(val_path,   val_rows)
    return train_path, val_path


def preextract(csv_path: str, cfg: dict, device: str):
    """Pre-extract and cache Mimi tokens + HuBERT features for every row in csv_path."""
    from emotion_dataset import MEADDataset
    from torch.utils.data import DataLoader

    ds = MEADDataset(csv_path, cfg, split="train", device=device)
    logger.info(f"Pre-extracting {len(ds)} samples from {csv_path} …")
    loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=lambda x: x)
    for i, _ in enumerate(loader):
        if i % 50 == 0:
            logger.info(f"  {i}/{len(ds)}")
    logger.info("Pre-extraction complete.")


def main():
    parser = argparse.ArgumentParser(description="Prepare MEAD emotion CSV for training")
    parser.add_argument("--csv",         required=True,  help="mead_labels.csv")
    parser.add_argument("--val-frac",    type=float, default=0.1)
    parser.add_argument("--out-dir",     default="data")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--config",      default="config.yaml",
                        help="config.yaml — required only with --preextract")
    parser.add_argument("--preextract",  action="store_true",
                        help="Pre-extract Mimi + HuBERT features and cache them")
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_csv, val_csv = split_csv(args.csv, args.val_frac, args.out_dir, args.seed)

    if args.preextract:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if "emotion" not in cfg:
            cfg["emotion"] = {"num_classes": 7}
        for csv_path in [train_csv, val_csv]:
            preextract(csv_path, cfg, args.device)


if __name__ == "__main__":
    main()
