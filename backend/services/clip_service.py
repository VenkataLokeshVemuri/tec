"""
clip_service.py — Multimodal CLIP embeddings (image + text)

Model: openai/clip-vit-base-patch32 (512-dim, CPU-friendly)
"""

import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from core.config import settings

_clip_model     = None
_clip_processor = None

def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        print(f"[CLIP] Loading model: {settings.CLIP_MODEL}")
        _clip_model     = CLIPModel.from_pretrained(settings.CLIP_MODEL)
        _clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
        _clip_model.eval()
    return _clip_model, _clip_processor


def embed_image(image_path: str) -> np.ndarray:
    """Return a L2-normalised (512,) float32 CLIP image embedding."""
    model, processor = _get_clip()
    image   = Image.open(image_path).convert("RGB")
    inputs  = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        feats = model.get_image_features(**inputs)

    vec = feats[0].cpu().numpy().astype(np.float32)
    return _l2_normalise(vec)


def embed_text_clip(text: str) -> np.ndarray:
    """Return a L2-normalised (512,) float32 CLIP text embedding."""
    try:
        model, processor = _get_clip()
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            feats = model.get_text_features(**inputs)

        vec = feats[0].cpu().numpy().astype(np.float32)
        return _l2_normalise(vec)
    except Exception as e:
        print(f"[CLIP] embed_text_clip failed: {e}")
        return np.zeros(512, dtype=np.float32)


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec
