"""
emotion_inference.py — Inference: HuBERT features + Predicted Emotion
======================================================================
Given an input audio file and a trained emotion_bridge checkpoint, outputs:
  • Continuous HuBERT-like feature representation  (T_h, output_dim)
  • Predicted emotion label  (e.g. "happy")
  • Emotion probabilities for all classes

Usage:
    python emotion_inference.py \
        --checkpoint checkpoints/emotion_bridge_best.pt \
        --config     config.yaml \
        --audio      /path/to/audio.wav

    # Save features to disk:
    python emotion_inference.py \
        --checkpoint checkpoints/emotion_bridge_best.pt \
        --config     config.yaml \
        --audio      /path/to/audio.wav \
        --output     features.pt

    # Override number of emotion classes if different from config:
    python emotion_inference.py ... --num-classes 8
"""

import argparse
import logging
import torch
import torch.nn.functional as F
import yaml

from model import MimiHuBERTBridgeWithEmotion
from emotion_dataset import EMOTION_CLASSES

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loader  (handles both full trainer ckpt and bare state_dicts)
# ──────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(path: str, model: MimiHuBERTBridgeWithEmotion, device: torch.device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    # Full trainer checkpoint
    if isinstance(ckpt, dict) and "bridge" in ckpt:
        sd = ckpt["bridge"]
    else:
        sd = ckpt

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"Missing keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")


# ──────────────────────────────────────────────────────────────────────────────
# Inference class
# ──────────────────────────────────────────────────────────────────────────────

class EmotionBridgeInference:
    """
    End-to-end inference: audio file → HuBERT features + emotion label.

    Usage:
        infer = EmotionBridgeInference("checkpoints/emotion_bridge_best.pt", "config.yaml")
        features, emotion_label, probs = infer.from_audio("/path/to/audio.wav")
        print(f"Predicted emotion: {emotion_label}")
        print(f"Features shape: {features.shape}")  # (T_h, 1024)
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        device: str = None,
        emotion_classes: list = None,
    ):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Inject emotion section defaults if missing
        if "emotion" not in self.cfg:
            self.cfg["emotion"] = {"num_classes": 7, "dropout": 0.0, "hidden_dim": 256}

        self.model = MimiHuBERTBridgeWithEmotion(self.cfg).to(self.device)
        _load_checkpoint(checkpoint_path, self.model, self.device)
        self.model.eval()

        # Emotion class names — use custom list or fall back to default
        self.emotion_classes = emotion_classes or EMOTION_CLASSES
        num_classes_cfg = self.cfg["emotion"].get("num_classes", len(self.emotion_classes))
        self.emotion_classes = self.emotion_classes[:num_classes_cfg]

        logger.info(
            f"Loaded emotion bridge from {checkpoint_path} on {self.device} | "
            f"Classes: {self.emotion_classes}"
        )

    @torch.no_grad()
    def __call__(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        tokens : (T, num_codebooks) or (B, T, num_codebooks)
        mask   : (B, 2T) bool or None
        returns: features (T_h, output_dim), emotion_label str, probs dict
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)          # (1, T, C)

        tokens = tokens.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        features, emotion_logits, _ = self.model(tokens, mask)

        # Feature output: (1, 2T, D) → (2T, D)  float32 on CPU
        features = features.squeeze(0).float().cpu()

        # Emotion prediction
        probs = F.softmax(emotion_logits[0].float(), dim=-1).cpu()
        pred_idx   = probs.argmax().item()
        pred_label = (
            self.emotion_classes[pred_idx]
            if pred_idx < len(self.emotion_classes)
            else f"class_{pred_idx}"
        )
        probs_dict = {
            self.emotion_classes[i] if i < len(self.emotion_classes) else f"class_{i}": probs[i].item()
            for i in range(len(probs))
        }

        return features, pred_label, probs_dict

    def from_audio(self, audio_path: str):
        """
        Full pipeline: audio file → features + emotion label.
        Returns:
            features    : (T_h, output_dim) float32 on CPU
            pred_label  : str  e.g. "happy"
            probs_dict  : dict {emotion: probability}
        """
        import torchaudio
        from dataset import MimiExtractor

        wav, native_sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)   # mono

        extractor = MimiExtractor(self.cfg["paths"]["mimi_model"])
        tokens    = extractor.extract(wav, native_sr)  # (T, 8)

        return self(tokens)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Emotion Bridge Inference: audio → HuBERT features + emotion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config",     required=True, help="Path to config.yaml")
    parser.add_argument("--audio",      required=True, help="Input .wav / .flac file")
    parser.add_argument("--output",     default=None,  help="Save features to this .pt path")
    parser.add_argument("--device",     default=None,  help="cuda | cpu")
    parser.add_argument("--num-classes",type=int, default=None,
                        help="Override emotion.num_classes from config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    infer = EmotionBridgeInference(args.checkpoint, args.config, device=args.device)

    if args.num_classes is not None:
        infer.cfg["emotion"]["num_classes"] = args.num_classes

    features, pred_label, probs = infer.from_audio(args.audio)

    print(f"\n{'='*50}")
    print(f"Audio          : {args.audio}")
    print(f"Features shape : {features.shape}  (T_h={features.shape[0]}, dim={features.shape[1]})")
    print(f"\n▶  Predicted emotion : {pred_label.upper()}  ({probs[pred_label]*100:.1f}%)")
    print(f"\nAll class probabilities:")
    for emotion, prob in sorted(probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {emotion:12s}  {prob*100:5.1f}%  {bar}")
    print("=" * 50)

    if args.output:
        torch.save(features, args.output)
        logger.info(f"Features saved → {args.output}")


if __name__ == "__main__":
    main()
