"""Weight derivation (dissertation Sections 4.9 / 4.12 / 4.11).

Three sources, selected by WEIGHT_SOURCE env:
  static    — hard-coded pilot matrix (Section 4.9)
  wif       — reference-statistics priors (Section 4.12)   [default]
  weightnet — neural per-task prediction if a checkpoint exists (Section 4.11)
Weights renormalise over the models actually available (API keys present),
unless mock mode is active, in which case the full pool is used.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from . import config
from .task import AgentTask

# Pilot performance matrix (Table 5B.1)
STATIC_MATRIX = {
    "planning":  {"gpt4o": 0.74, "claude_s46": 0.79, "deepseek": 0.61, "qwen35": 0.68},
    "implement": {"gpt4o": 0.70, "claude_s46": 0.72, "deepseek": 0.81, "qwen35": 0.75},
    "review":    {"gpt4o": 0.72, "claude_s46": 0.83, "deepseek": 0.64, "qwen35": 0.71},
    "ui_debug":  {"gpt4o": 0.75, "claude_s46": 0.80, "deepseek": 0.58, "qwen35": 0.63},
}


@dataclass
class WeightMap:
    weights: dict[str, float]
    tau: float
    source: str
    subtask_type: str

    def active(self, threshold: float = config.WEIGHT_THRESHOLD) -> dict[str, float]:
        return {m: w for m, w in self.weights.items() if w >= threshold}


def _softmax(scores: dict[str, float], tau: float) -> dict[str, float]:
    exp = {m: math.exp(p / tau) for m, p in scores.items()}
    z = sum(exp.values())
    return {m: round(v / z, 4) for m, v in exp.items()}


def _restrict(scores: dict[str, float]) -> dict[str, float]:
    if config.mock_mode():
        return scores
    avail = config.available_models()
    kept = {m: p for m, p in scores.items() if m in avail}
    return kept or scores


def derive_weights(task: AgentTask, tau: float | None = None) -> WeightMap:
    tau = tau if tau is not None else config.TAU
    source = config.WEIGHT_SOURCE

    if source == "weightnet" and config.WEIGHTNET_CKPT.exists():
        try:
            from .weightnet import predict_scores
            scores = predict_scores(task)
            return WeightMap(_softmax(_restrict(scores), tau), tau,
                             "weightnet", task.subtask_type)
        except Exception:
            source = "wif"          # graceful fallback

    if source == "wif":
        from .wif import prior_by_subtask
        try:
            scores = prior_by_subtask()[task.subtask_type]
            return WeightMap(_softmax(_restrict(scores), tau), tau,
                             "wif", task.subtask_type)
        except (FileNotFoundError, KeyError):
            source = "static"       # graceful fallback

    scores = STATIC_MATRIX[task.subtask_type]
    return WeightMap(_softmax(_restrict(scores), tau), tau,
                     "static", task.subtask_type)
