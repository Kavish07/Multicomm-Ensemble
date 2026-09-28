"""Environment, model registry configuration, and mock-mode switch."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MODEL_POOL = ["gpt4o", "claude_s46", "deepseek", "qwen35"]

PROVIDERS = {
    "gpt4o": {
        "kind": "openai",
        "key_env": "OPENAI_API_KEY",
        "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        "base_url": None,
    },
    "claude_s46": {
        "kind": "anthropic",
        "key_env": "ANTHROPIC_API_KEY",
        "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        "base_url": None,
    },
    "deepseek": {
        "kind": "openai",
        "key_env": "DEEPSEEK_API_KEY",
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    },
    "qwen35": {
        "kind": "openai",
        "key_env": "QWEN_API_KEY",
        "model": os.getenv("QWEN_MODEL", "qwen2.5-coder-32b-instruct"),
        "base_url": os.getenv(
            "QWEN_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    },
}

# Aggregator for fusion (never part of the proposer pool)
AGGREGATOR = {"kind": "anthropic", "key_env": "ANTHROPIC_API_KEY",
              "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRIOR_DATASET = DATA_DIR / "weight_prior_dataset.csv"
WEIGHTNET_CKPT = DATA_DIR / "weightnet.pt"

WEIGHT_SOURCE = os.getenv("WEIGHT_SOURCE", "wif").lower()   # wif | static | weightnet
TAU = float(os.getenv("MULTICOMM_TAU", "0.1"))
WEIGHT_THRESHOLD = 0.05


def available_models() -> list[str]:
    """Pool members whose API key is configured."""
    return [m for m in MODEL_POOL if os.getenv(PROVIDERS[m]["key_env"])]


def mock_mode() -> bool:
    forced = os.getenv("MULTICOMM_MOCK", "").strip()
    if forced in ("1", "true", "yes"):
        return True
    if forced in ("0", "false", "no"):
        return False
    return len(available_models()) == 0     # auto-mock when no keys
