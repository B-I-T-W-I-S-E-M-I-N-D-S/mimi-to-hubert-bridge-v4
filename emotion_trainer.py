"""
emotion_trainer.py — Multi-Task Trainer (HuBERT Features + Emotion)
=====================================================================
Extends the original Trainer for the new MimiHuBERTBridgeWithEmotion model.
All DDP, AMP, gradient accumulation, checkpointing and TensorBoard logic
from the original trainer.py is preserved — only the model, criterion, and
batch handling are updated.

Launch:
    # Single-GPU
    python emotion_train.py --config config.yaml --csv mead_labels.csv

    # Multi-GPU (recommended)
    torchrun --nproc_per_node=4 emotion_train.py \
        --config config.yaml \
        --train-csv data/mead_train.csv \
        --val-csv   data/mead_val.csv

    # Resume
    torchrun --nproc_per_node=4 emotion_train.py \
        --config config.yaml \
        --train-csv data/mead_train.csv \
        --val-csv   data/mead_val.csv \
        --resume checkpoints/emotion_bridge_best.pt
"""

import logging
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model import MimiHuBERTBridgeWithEmotion, FeatureDiscriminator
from emotion_losses import EmotionBridgeLoss
from emotion_dataset import MEADDataset, emotion_collate_fn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# DDP helpers  (identical to trainer.py)
# ──────────────────────────────────────────────────────────────────────────────

def setup_ddp():
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK",       0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size


def teardown_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, (DDP, nn.DataParallel)) else model


@contextmanager
def _null_ctx():
    yield


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler  (identical to trainer.py)
# ──────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    t      = cfg["training"]
    warmup = t["warmup_steps"]
    total  = t["num_epochs"] * steps_per_epoch
    ws = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup)
    cs = CosineAnnealingLR(optimizer, T_max=max(1, total - warmup), eta_min=1e-6)
    return SequentialLR(optimizer, schedulers=[ws, cs], milestones=[warmup])


# ──────────────────────────────────────────────────────────────────────────────
# CUDA Prefetcher  (identical to trainer.py)
# ──────────────────────────────────────────────────────────────────────────────

class CUDAPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self._iter  = iter(self.loader)
        self._preload()

    def _preload(self):
        try:
            self._next = next(self._iter)
        except StopIteration:
            self._next = None
            return
        if self.stream is None:
            return
        with torch.cuda.stream(self.stream):
            self._next = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in self._next.items()
            }

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self):
        if self.stream:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self._next
        if batch is None:
            raise StopIteration
        self._preload()
        return batch

    def __len__(self):
        return len(self.loader)


# ──────────────────────────────────────────────────────────────────────────────
# Emotion Trainer
# ──────────────────────────────────────────────────────────────────────────────

class EmotionTrainer:
    """
    Multi-task trainer for MimiHuBERTBridgeWithEmotion.

    Key differences from Trainer (trainer.py):
      - Uses MimiHuBERTBridgeWithEmotion (outputs features + emotion_logits)
      - Uses EmotionBridgeLoss  (bridge losses + emotion CE + uncertainty weight)
      - Loads MEADDataset from CSV instead of JSONL manifests
      - Tracks emotion_acc in validation; saves best on combined val loss
      - Checkpoints save emotion_head weights separately for clean loading
    """

    def __init__(self, cfg: dict, train_csv: str, val_csv: Optional[str] = None):
        self.cfg      = cfg
        self.train_csv = train_csv
        self.val_csv   = val_csv
        t_cfg = cfg["training"]

        # ── DDP ──────────────────────────────────────────────────────────────
        self.local_rank, self.global_rank, self.world_size = setup_ddp()
        self.is_main = (self.global_rank == 0)

        # ── Device ───────────────────────────────────────────────────────────
        self.device = (
            torch.device(f"cuda:{self.local_rank}")
            if torch.cuda.is_available() else torch.device("cpu")
        )
        self.amp_device      = self.device.type
        self.mixed_precision = t_cfg.get("mixed_precision", True) and self.device.type == "cuda"

        torch.manual_seed(t_cfg.get("seed", 42) + self.global_rank)

        if self.is_main:
            logger.info(f"[EmotionTrainer] World={self.world_size} | Device={self.device} | AMP={self.mixed_precision}")

        self.accum_steps = int(t_cfg.get("accum_steps", 1))

        # ── Models ───────────────────────────────────────────────────────────
        bridge_raw = MimiHuBERTBridgeWithEmotion(cfg).to(self.device)
        disc_raw   = FeatureDiscriminator(
            input_dim  = cfg["model"]["output_dim"],
            hidden     = t_cfg["disc_hidden"],
            num_layers = t_cfg["disc_layers"],
        ).to(self.device)

        if self.world_size > 1:
            self.bridge = DDP(bridge_raw, device_ids=[self.local_rank],
                              output_device=self.local_rank, find_unused_parameters=False)
            self.disc   = DDP(disc_raw,   device_ids=[self.local_rank],
                              output_device=self.local_rank, find_unused_parameters=False)
        else:
            self.bridge = bridge_raw
            self.disc   = disc_raw

        # ── Loss ─────────────────────────────────────────────────────────────
        self.criterion = EmotionBridgeLoss(cfg).to(self.device)

        # ── Optimisers ────────────────────────────────────────────────────────
        # Include criterion parameters so uncertainty weights get updated.
        self.opt_g = AdamW(
            list(_unwrap(self.bridge).parameters()) + list(self.criterion.parameters()),
            lr=t_cfg["learning_rate"], weight_decay=t_cfg["weight_decay"],
            fused=(self.device.type == "cuda"),
        )
        self.opt_d = AdamW(
            _unwrap(self.disc).parameters(),
            lr=t_cfg["disc_lr"], weight_decay=t_cfg["weight_decay"],
            fused=(self.device.type == "cuda"),
        )

        # ── Data ──────────────────────────────────────────────────────────────
        self.train_loader, self.val_loader = self._build_loaders()
        steps_per_epoch = len(self.train_loader) // self.accum_steps

        # ── Schedulers ────────────────────────────────────────────────────────
        self.sched_g = build_scheduler(self.opt_g, cfg, steps_per_epoch)
        self.sched_d = CosineAnnealingLR(
            self.opt_d,
            T_max=t_cfg["num_epochs"] * steps_per_epoch, eta_min=1e-7
        )

        # ── AMP ───────────────────────────────────────────────────────────────
        self.scaler_g = GradScaler(device=self.amp_device, enabled=self.mixed_precision)
        self.scaler_d = GradScaler(device=self.amp_device, enabled=self.mixed_precision)

        # ── State ─────────────────────────────────────────────────────────────
        self.global_step  = 0
        self.epoch        = 0
        self.best_val     = math.inf   # tracks combined val loss (not just MSE)

        self.ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
        self.log_dir  = Path(cfg["paths"]["log_dir"])
        if self.is_main:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = None
        if self.is_main and cfg["paths"].get("tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                logger.warning("tensorboard not installed.")

        self.disc_start_step = t_cfg.get("disc_start_step", 5000)

        if self.is_main:
            p = _unwrap(self.bridge).get_param_count()
            logger.info(
                f"Bridge params: {p['bridge']:,}  |  Emotion head: {p['emotion_head']:,}  "
                f"|  Total trainable: {p['trainable']:,}"
            )

    # ── DataLoaders ───────────────────────────────────────────────────────────

    def _build_loaders(self):
        d_cfg = self.cfg["data"]
        t_cfg = self.cfg["training"]
        nw    = d_cfg.get("num_workers", 4)

        train_ds = MEADDataset(self.train_csv, self.cfg, "train")
        train_sampler = (
            DistributedSampler(train_ds, self.world_size, self.global_rank, shuffle=True, drop_last=True)
            if self.world_size > 1 else None
        )
        train_loader = DataLoader(
            train_ds, batch_size=t_cfg["batch_size"],
            sampler=train_sampler, shuffle=(train_sampler is None),
            num_workers=nw, collate_fn=emotion_collate_fn,
            pin_memory=True, drop_last=True,
            persistent_workers=(nw > 0),
            prefetch_factor=4 if nw > 0 else None,
        )
        self.train_sampler = train_sampler

        val_loader = None
        if self.val_csv and os.path.isfile(self.val_csv):
            val_ds = MEADDataset(self.val_csv, self.cfg, "val")
            val_sampler = (
                DistributedSampler(val_ds, self.world_size, self.global_rank, shuffle=False)
                if self.world_size > 1 else None
            )
            val_loader = DataLoader(
                val_ds, batch_size=t_cfg["batch_size"],
                sampler=val_sampler, shuffle=False,
                num_workers=max(nw // 2, 2), collate_fn=emotion_collate_fn,
                pin_memory=True,
                persistent_workers=(nw > 0),
                prefetch_factor=2 if nw > 0 else None,
            )
        return train_loader, val_loader

    # ── Device transfer ───────────────────────────────────────────────────────

    def _to_device(self, batch: dict) -> dict:
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    # ── Train step ────────────────────────────────────────────────────────────

    def _train_step(self, batch: dict, is_accum_step: bool) -> dict:
        tokens        = batch["tokens"]
        target        = batch["hubert"]
        emotion_labels= batch["emotion_label"]
        mask          = batch.get("mask", None)
        use_adv       = self.global_step >= self.disc_start_step

        # Discriminator step
        d_logs = {}
        if use_adv and not is_accum_step:
            self.opt_d.zero_grad(set_to_none=True)
            with autocast(device_type=self.amp_device, enabled=self.mixed_precision):
                pred_det, _, _ = self.bridge(tokens, mask)
                real_logits = self.disc(target)
                fake_logits = self.disc(pred_det.detach())
                d_loss, d_logs = self.criterion.adv.discriminator_loss(real_logits, fake_logits)
            self.scaler_d.scale(d_loss).backward()
            self.scaler_d.unscale_(self.opt_d)
            torch.nn.utils.clip_grad_norm_(
                _unwrap(self.disc).parameters(), self.cfg["training"]["grad_clip"]
            )
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()
            self.sched_d.step()

        # Generator / bridge step
        scale = 1.0 / self.accum_steps
        ctx = (
            self.bridge.no_sync()
            if (self.world_size > 1 and is_accum_step and isinstance(self.bridge, DDP))
            else _null_ctx()
        )
        with ctx:
            with autocast(device_type=self.amp_device, enabled=self.mixed_precision):
                pred, emotion_logits, _ = self.bridge(tokens, mask)
                fake_disc_logits = self.disc(pred) if use_adv else None
                g_loss, g_logs = self.criterion(
                    pred, emotion_logits, target, emotion_labels, batch, fake_disc_logits
                )
                g_loss = g_loss * scale
            self.scaler_g.scale(g_loss).backward()

        if not is_accum_step:
            self.scaler_g.unscale_(self.opt_g)
            torch.nn.utils.clip_grad_norm_(
                list(_unwrap(self.bridge).parameters()) + list(self.criterion.parameters()),
                self.cfg["training"]["grad_clip"],
            )
            self.scaler_g.step(self.opt_g)
            self.scaler_g.update()
            self.opt_g.zero_grad(set_to_none=True)
            self.sched_g.step()

        return {**g_logs, **d_logs}

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self) -> dict:
        if self.val_loader is None:
            return {}

        self.bridge.eval()
        agg, n = {}, 0

        for batch in self.val_loader:
            batch = self._to_device(batch)
            mask  = batch.get("mask", None)
            with autocast(device_type=self.amp_device, enabled=self.mixed_precision):
                pred, emotion_logits, _ = self.bridge(batch["tokens"], mask)

            pred_fp32   = pred.float()
            target_fp32 = batch["hubert"].float()
            _, logs = self.criterion(
                pred_fp32, emotion_logits,
                target_fp32, batch["emotion_label"], batch
            )
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + (v if isinstance(v, float) else float(v))
            n += 1

        agg = {k: v / max(n, 1) for k, v in agg.items()}

        # All-reduce across ranks
        if self.world_size > 1:
            for k in agg:
                t = torch.tensor(agg[k], device=self.device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                agg[k] = t.item()

        self.bridge.train()
        return agg

    # ── TensorBoard ───────────────────────────────────────────────────────────

    def _log(self, logs: dict, prefix: str = "train"):
        if self.is_main and self.writer:
            for k, v in logs.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, self.global_step)

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str, val_logs: Optional[dict] = None):
        if not self.is_main:
            return
        raw = _unwrap(self.bridge)
        ckpt = {
            "step":          self.global_step,
            "epoch":         self.epoch,
            "bridge":        raw.state_dict(),          # full model inc emotion head
            "emotion_head":  raw.emotion_head.state_dict(),  # convenience sub-dict
            "disc":          _unwrap(self.disc).state_dict(),
            "opt_g":         self.opt_g.state_dict(),
            "opt_d":         self.opt_d.state_dict(),
            "sched_g":       self.sched_g.state_dict(),
            "sched_d":       self.sched_d.state_dict(),
            "criterion":     self.criterion.state_dict(),  # includes unc log_vars
            "best_val":      self.best_val,
            "val_logs":      val_logs or {},
            "emotion_classes": self.cfg.get("emotion", {}).get("num_classes", 7),
        }
        path = self.ckpt_dir / f"emotion_bridge_{tag}.pt"
        torch.save(ckpt, path)
        logger.info(f"Saved → {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        _unwrap(self.bridge).load_state_dict(ckpt["bridge"], strict=False)
        _unwrap(self.disc).load_state_dict(ckpt["disc"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.sched_g.load_state_dict(ckpt["sched_g"])
        self.sched_d.load_state_dict(ckpt["sched_d"])
        if "criterion" in ckpt:
            self.criterion.load_state_dict(ckpt["criterion"], strict=False)
        self.global_step = ckpt["step"]
        self.epoch       = ckpt["epoch"]
        self.best_val    = ckpt.get("best_val", math.inf)
        logger.info(f"Resumed from step {self.global_step} (epoch {self.epoch})")

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, resume_from: Optional[str] = None):
        if resume_from:
            self.load_checkpoint(resume_from)

        t_cfg      = self.cfg["training"]
        num_epochs = t_cfg["num_epochs"]

        if self.is_main:
            logger.info(f"[EmotionTrainer] Starting — {num_epochs} epochs")

        self.bridge.train()
        prefetcher = CUDAPrefetcher(self.train_loader, self.device)

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            if self.world_size > 1 and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            t0, epoch_logs, n_steps, micro_step = time.time(), {}, 0, 0

            for batch in prefetcher:
                micro_step += 1
                is_accum    = (micro_step % self.accum_steps != 0)
                step_logs   = self._train_step(batch, is_accum)

                if not is_accum:
                    for k, v in step_logs.items():
                        epoch_logs[k] = epoch_logs.get(k, 0.0) + v
                    n_steps          += 1
                    self.global_step += 1

                    if self.is_main and self.global_step % 100 == 0:
                        avg = {k: v / n_steps for k, v in epoch_logs.items()}
                        lr  = self.opt_g.param_groups[0]["lr"]
                        logger.info(
                            f"Step {self.global_step:6d} | ep {epoch+1}/{num_epochs} | "
                            f"loss={avg.get('total', float('nan')):.4f} | "
                            f"emo_acc={avg.get('emotion_acc', 0):.3f} | "
                            f"lr={lr:.2e} | {time.time()-t0:.0f}s"
                        )
                        self._log(step_logs, "train")

            # Validation
            val_logs = self._val_epoch()
            if self.is_main:
                self._log(val_logs, "val")
                val_loss = val_logs.get("total", math.inf)
                acc      = val_logs.get("emotion_acc", 0.0)
                mse      = val_logs.get("recon_mse",   float("nan"))
                logger.info(
                    f"[Epoch {epoch+1:3d}] val_loss={val_loss:.4f} | "
                    f"recon_mse={mse:.5f} | emotion_acc={acc:.3f}"
                )
                self.save_checkpoint(f"epoch{epoch+1:03d}", val_logs)
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    self.save_checkpoint("best", val_logs)
                    logger.info(f"  ↳ New best val loss: {val_loss:.4f}")

            if self.world_size > 1:
                dist.barrier()

        if self.is_main and self.writer:
            self.writer.close()
        teardown_ddp()
        if self.is_main:
            logger.info("[EmotionTrainer] Training complete.")
