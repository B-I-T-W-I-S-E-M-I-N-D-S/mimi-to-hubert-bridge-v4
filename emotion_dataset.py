"""
emotion_dataset.py — MEAD Emotion Dataset + Collate
=====================================================
Loads audio + emotion labels from a CSV manifest (mead_labels.csv):

    filename,emotion
    ./audio/W037_disgusted_level_1_026.wav,disgust
    ./audio/W037_happy_level_3_002.wav,happy
    ./audio/W040_sad_level_2_016.wav,sad

Pipeline per sample:
  1. Load audio → resample to 24 kHz → Mimi tokens  (T_m, 8)
  2. Resample to 16 kHz → HuBERT features            (T_h, 1024)  [if ONNX available]
  3. Extract F0 + energy                              (T_h,)
  4. Return emotion label as integer index

Usage:
    from emotion_dataset import MEADDataset, emotion_collate_fn, EMOTION_CLASSES

Integrate with existing MimiHuBERTDataset infrastructure (caching, extractors)
by constructing with the same cfg dict passed to the bridge trainer.
"""

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Canonical emotion label mapping
# Add / remove classes to match your dataset. The ORDER matters — it sets the
# integer index used in CrossEntropyLoss. Keep this list in sync with
#   config.yaml:  emotion.num_classes
# ──────────────────────────────────────────────────────────────────────────────

EMOTION_CLASSES: List[str] = [
    "neutral",    # 0
    "happy",      # 1
    "sad",        # 2
    "angry",      # 3
    "disgust",    # 4
    "fear",       # 5
    "surprise",   # 6
    "contempt",   # 7  (remove if not in your dataset)
]

EMOTION2IDX: Dict[str, int] = {e: i for i, e in enumerate(EMOTION_CLASSES)}
# Aliases so "disgusted" maps to "disgust", "happiness" → "happy", etc.
_ALIASES: Dict[str, str] = {
    "disgusted": "disgust",
    "happiness": "happy",
    "sadness":   "sad",
    "anger":     "angry",
    "fearful":   "fear",
    "surprised": "surprise",
}


def parse_emotion(label: str) -> int:
    """Convert a raw emotion string → integer class index."""
    label = label.strip().lower()
    label = _ALIASES.get(label, label)
    if label not in EMOTION2IDX:
        raise ValueError(
            f"Unknown emotion '{label}'. "
            f"Known: {list(EMOTION2IDX.keys())}. "
            f"Add it to EMOTION_CLASSES in emotion_dataset.py."
        )
    return EMOTION2IDX[label]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class MEADDataset(Dataset):
    """
    MEAD-style emotion dataset.

    Each CSV row:  filename, emotion
    Each __getitem__ returns:
        tokens       : (T_m, 8)          int64  Mimi tokens at 12.5 Hz
        hubert       : (T_h, feat_dim)   float32 HuBERT features at 25 Hz
        f0           : (T_h,)            float32 log-F0
        energy       : (T_h,)            float32 log-energy
        voiced       : (T_h,)            bool
        emotion_label: scalar            int64  class index
        audio_path   : str

    Shares Mimi / HuBERT extractors and caching logic with MimiHuBERTDataset.
    """

    def __init__(
        self,
        csv_path: str,
        cfg: dict,
        split: str = "train",
        device: str = "cpu",
    ):
        self.cfg    = cfg
        self.split  = split
        self.device = device

        d_cfg = cfg["data"]
        self.sr         = d_cfg["sample_rate"]          # 16 kHz (HuBERT)
        self.mimi_sr    = d_cfg["mimi_sample_rate"]     # 24 kHz (Mimi)
        self.hop_length = cfg["training"]["hop_length"]
        self.max_secs   = d_cfg["max_audio_seconds"]
        self.cache_feat = d_cfg.get("cache_features", True)
        self.cache_dir  = Path(d_cfg.get("cache_dir", "data/cache"))

        self.samples = self._load_csv(csv_path)
        logger.info(f"[MEADDataset/{split}] {len(self.samples)} samples from {csv_path}")

        # Lazy-loaded extractors (shared across workers via __getattr__)
        self._mimi    = None
        self._hubert  = None

    # ── CSV loading ───────────────────────────────────────────────────────────

    def _load_csv(self, csv_path: str) -> List[dict]:
        samples = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fpath = row["filename"].strip()
                emotion_str = row["emotion"].strip()
                try:
                    emotion_idx = parse_emotion(emotion_str)
                except ValueError as e:
                    logger.warning(f"Skipping row — {e}")
                    continue
                if not os.path.isfile(fpath):
                    logger.warning(f"Audio not found, skipping: {fpath}")
                    continue
                samples.append({"audio_path": fpath, "emotion": emotion_idx})
        return samples

    # ── Lazy extractor accessors ──────────────────────────────────────────────

    def _get_mimi(self):
        if self._mimi is None:
            from dataset import MimiExtractor
            self._mimi = MimiExtractor(
                model_name=self.cfg["paths"]["mimi_model"], device=self.device
            )
        return self._mimi

    def _get_hubert(self):
        if self._hubert is None:
            from dataset import HuBERTExtractor
            self._hubert = HuBERTExtractor(
                onnx_path=self.cfg["paths"]["hubert_model"], device=self.device
            )
        return self._hubert

    # ── Cache helpers (mirrors MimiHuBERTDataset._get_or_cache) ──────────────

    def _cache_path(self, audio_path: str, kind: str) -> Path:
        import hashlib
        key = hashlib.md5(audio_path.encode()).hexdigest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{key}_{kind}.pt"

    def _get_or_cache(self, audio_path: str, kind: str, fn):
        if not self.cache_feat:
            return fn()
        cp = self._cache_path(audio_path, kind)
        if cp.exists():
            try:
                return torch.load(cp, weights_only=False)
            except Exception:
                pass
        result = fn()
        torch.save(result, cp)
        return result

    # ── Audio loading ─────────────────────────────────────────────────────────

    def _load_audio(self, path: str):
        import torchaudio
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        # Truncate to max_audio_seconds
        max_samples = int(self.max_secs * sr)
        if wav.shape[-1] > max_samples:
            wav = wav[:, :max_samples]
        return wav, sr  # (1, N)

    # ── Prosody extraction ────────────────────────────────────────────────────

    @staticmethod
    def _resample_array(arr: np.ndarray, target_len: int) -> np.ndarray:
        if len(arr) == target_len:
            return arr
        indices = np.linspace(0, len(arr) - 1, target_len)
        return np.interp(indices, np.arange(len(arr)), arr).astype(arr.dtype)

    def _extract_prosody(self, wav_np: np.ndarray, T_h: int):
        from dataset import extract_f0_energy
        f0_np, energy_np, voiced_np = extract_f0_energy(wav_np, self.sr, self.hop_length)
        f0     = torch.from_numpy(self._resample_array(f0_np,     T_h))
        energy = torch.from_numpy(self._resample_array(energy_np, T_h))
        voiced = torch.from_numpy(
            self._resample_array(voiced_np.astype(np.float32), T_h) > 0.5
        )
        return f0, energy, voiced

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample  = self.samples[idx]
        apath   = sample["audio_path"]
        emotion = sample["emotion"]

        wav, native_sr = self._load_audio(apath)
        wav_np = wav.squeeze(0).numpy()

        # Mimi tokens  (T_m, 8)  at 12.5 Hz
        tokens = self._get_or_cache(
            apath, "mimi",
            lambda: self._get_mimi().extract(wav, native_sr)
        )

        # HuBERT GT features  (T_h, feat_dim)  at 25 Hz
        hubert = self._get_or_cache(
            apath, "hubert",
            lambda: self._get_hubert().extract(wav, native_sr),
        )

        # Align token / feature lengths (2:1 ratio)
        T_m   = tokens.shape[0]
        T_h   = hubert.shape[0]
        T_min = min(T_m, T_h // 2)
        tokens = tokens[:T_min]
        hubert = hubert[:T_min * 2]
        T_h    = T_min * 2

        # Prosody
        def _prosody():
            # Resample wav to 16 kHz for prosody (same as HuBERT pipeline)
            import torchaudio.functional as TAF
            wav16 = TAF.resample(wav, native_sr, self.sr).squeeze(0).numpy()
            return self._extract_prosody(wav16, T_h)

        prosody = self._get_or_cache(apath + f"_L{T_h}", "prosody", _prosody)
        if isinstance(prosody, tuple) and len(prosody) == 3:
            f0, energy, voiced = prosody
        else:
            f0, energy, voiced = _prosody()

        return {
            "tokens":        tokens,                              # (T_m, 8)
            "hubert":        hubert,                              # (T_h, feat_dim)
            "f0":            f0,                                  # (T_h,)
            "energy":        energy,                              # (T_h,)
            "voiced":        voiced,                              # (T_h,)
            "emotion_label": torch.tensor(emotion, dtype=torch.long),  # scalar
            "audio_path":    apath,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────────────────────────────────────

def emotion_collate_fn(batch: List[dict]) -> dict:
    """
    Pads variable-length sequences; stacks scalar emotion labels.
    Returns all keys expected by EmotionTrainer._train_step().
    """
    batch = sorted(batch, key=lambda x: x["tokens"].shape[0], reverse=True)

    max_T_m   = max(b["tokens"].shape[0] for b in batch)
    max_T_h   = max_T_m * 2
    B         = len(batch)
    feat_dim  = batch[0]["hubert"].shape[-1]
    num_cb    = batch[0]["tokens"].shape[-1]

    tokens_out  = torch.zeros(B, max_T_m, num_cb, dtype=torch.long)
    hubert_out  = torch.zeros(B, max_T_h, feat_dim)
    f0_out      = torch.zeros(B, max_T_h)
    energy_out  = torch.zeros(B, max_T_h)
    voiced_out  = torch.zeros(B, max_T_h, dtype=torch.bool)
    mask_out    = torch.zeros(B, max_T_h, dtype=torch.bool)
    emotion_out = torch.zeros(B, dtype=torch.long)
    phone_out   = torch.full((B, max_T_h), -100, dtype=torch.long)

    token_lengths = []
    for i, s in enumerate(batch):
        T_m = s["tokens"].shape[0]
        T_h = T_m * 2
        tokens_out[i, :T_m]  = s["tokens"]
        hubert_out[i, :T_h]  = s["hubert"]
        f0_out[i, :T_h]      = s["f0"]
        energy_out[i, :T_h]  = s["energy"]
        voiced_out[i, :T_h]  = s["voiced"]
        mask_out[i, :T_h]    = True
        emotion_out[i]       = s["emotion_label"]
        token_lengths.append(T_m)

    token_lengths = torch.tensor(token_lengths, dtype=torch.long)
    frame_lengths = token_lengths * 2

    return {
        "tokens":         tokens_out,
        "hubert":         hubert_out,
        "f0":             f0_out,
        "energy":         energy_out,
        "voiced_mask":    voiced_out,
        "mask":           mask_out,
        "emotion_label":  emotion_out,           # (B,) int64 — NEW
        "phone_labels":   phone_out,
        "input_lengths":  frame_lengths,
        "ctc_targets":    None,
        "target_lengths": None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def build_emotion_dataloaders(
    cfg: dict,
    train_csv: str,
    val_csv: Optional[str] = None,
    world_size: int = 1,
    global_rank: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train (and optionally val) DataLoaders for emotion multi-task training.

    If val_csv is None, no val loader is returned.
    Supports DistributedSampler for multi-GPU training.
    """
    from torch.utils.data.distributed import DistributedSampler

    t_cfg = cfg["training"]
    d_cfg = cfg["data"]
    bsz   = t_cfg["batch_size"]
    nw    = d_cfg.get("num_workers", 4)

    train_ds = MEADDataset(train_csv, cfg, split="train")
    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=global_rank, shuffle=True, drop_last=True)
        if world_size > 1 else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size  = bsz,
        sampler     = train_sampler,
        shuffle     = (train_sampler is None),
        num_workers = nw,
        collate_fn  = emotion_collate_fn,
        pin_memory  = True,
        drop_last   = True,
        persistent_workers = (nw > 0),
        prefetch_factor    = 4 if nw > 0 else None,
    )

    val_loader = None
    if val_csv is not None and os.path.isfile(val_csv):
        val_ds = MEADDataset(val_csv, cfg, split="val")
        val_sampler = (
            DistributedSampler(val_ds, num_replicas=world_size, rank=global_rank, shuffle=False)
            if world_size > 1 else None
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = bsz,
            sampler     = val_sampler,
            shuffle     = False,
            num_workers = max(nw // 2, 2),
            collate_fn  = emotion_collate_fn,
            pin_memory  = True,
            persistent_workers = (nw > 0),
            prefetch_factor    = 2 if nw > 0 else None,
        )

    return train_loader, val_loader
