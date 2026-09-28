"""ImgCompress — visual payload compression (deployable implementation).

The dissertation's variational-IB encoder (Section 4.4) is replaced here by a
deterministic crop/resize/re-encode path with per-model output resolution —
the same interface and per-model beta semantics, no training required.
Beta maps to output size: lower beta = more aggressive compression.
"""
from __future__ import annotations
import io
from PIL import Image

MODEL_BETA = {"gpt4o": 0.10, "claude_s46": 0.10, "deepseek": 0.05, "qwen35": 0.08}
_BETA_TO_SIZE = {0.10: (480, 320), 0.08: (400, 267), 0.05: (320, 214)}


def encode(image_bytes: bytes, receiver_model: str | None = None) -> tuple[bytes, dict]:
    beta = MODEL_BETA.get(receiver_model or "", 0.10)
    size = _BETA_TO_SIZE.get(beta, (480, 320))
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig = im.size
    im.thumbnail(size)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=70, optimize=True)
    out = buf.getvalue()
    report = {
        "beta": beta,
        "original_px": f"{orig[0]}x{orig[1]}",
        "compressed_px": f"{im.size[0]}x{im.size[1]}",
        "bytes_before": len(image_bytes),
        "bytes_after": len(out),
        "reduction_pct": round(100 * (1 - len(out) / max(len(image_bytes), 1)), 1),
    }
    return out, report
