"""Weight Initialisation Framework (dissertation Section 4.12).

Loads the referenced-statistics dataset and aggregates it into the prior
success matrix P0[model][functionality], with tier weighting, recency decay,
and a cross-benchmark ratio estimator for missing cells.
"""
from __future__ import annotations
import csv, math, datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from .config import MODEL_POOL, PRIOR_DATASET

TIER_W = {"A": 1.0, "B": 0.7, "C": 0.4}

# Map dataset functionalities -> pipeline subtask types
FUNC_TO_SUBTASK = {
    "planning": "planning",
    "architecture": "planning",
    "algorithmic_implementation": "implement",
    "repo_scale_editing": "implement",
    "multilang_editing": "implement",
    "debugging": "implement",
    "code_review": "review",
    "ui_visual_development": "ui_debug",
}


@dataclass
class Stat:
    model: str
    functionality: str
    benchmark: str
    score: float
    tier: str
    date: str
    proxy: bool
    relevance: float = 1.0


def load_dataset(path=PRIOR_DATASET) -> list[Stat]:
    stats = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["tier"] not in TIER_W:          # skip estimator rows (tier E)
                continue
            stats.append(Stat(
                model=r["model_pool_id"],
                functionality=r["functionality"],
                benchmark=r["benchmark"],
                score=float(r["score"]),
                tier=r["tier"],
                date=r["date_accessed"],
                proxy=r["proxy_flag"] != "no",
            ))
    return stats


def recency(date_str: str, half_life_months: float = 12.0) -> float:
    try:
        d = dt.date.fromisoformat(date_str)
    except ValueError:
        return 1.0
    months = max((dt.date.today() - d).days, 0) / 30.4
    return math.exp(-months / half_life_months)


def ratio_estimate(model: str, func: str, stats: list[Stat],
                   ref_func: str = "repo_scale_editing") -> float | None:
    """Fill a missing cell via the mean target/reference ratio (Section 4.12.2)."""
    def cell(m, f):
        vals = [s.score for s in stats if s.model == m and s.functionality == f]
        return sum(vals) / len(vals) if vals else None

    ratios = []
    for m in MODEL_POOL:
        t, s = cell(m, func), cell(m, ref_func)
        if t is not None and s:
            ratios.append(t / s)
    ref = cell(model, ref_func)
    if not ratios or ref is None:
        return None
    pool_max = max((s.score for s in stats if s.functionality == func), default=1.0)
    return round(min(ref * sum(ratios) / len(ratios), pool_max), 3)


def prior_matrix(stats: list[Stat] | None = None) -> dict[str, dict[str, float]]:
    """P0[model][functionality] from the dataset."""
    stats = stats if stats is not None else load_dataset()
    funcs = sorted({s.functionality for s in stats})
    P0: dict[str, dict[str, float]] = defaultdict(dict)
    for func in funcs:
        for m in MODEL_POOL:
            rows = [s for s in stats if s.model == m and s.functionality == func]
            if rows:
                num = sum(s.score * TIER_W[s.tier] * recency(s.date) * s.relevance
                          for s in rows)
                den = sum(TIER_W[s.tier] * recency(s.date) * s.relevance
                          for s in rows)
                P0[m][func] = round(num / den, 3)
            else:
                est = ratio_estimate(m, func, stats)
                if est is not None:
                    P0[m][func] = est
    return dict(P0)


def prior_by_subtask(P0: dict[str, dict[str, float]] | None = None
                     ) -> dict[str, dict[str, float]]:
    """Collapse functionality-level P0 to subtask-level scores (mean)."""
    P0 = P0 if P0 is not None else prior_matrix()
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for subtask in ("planning", "implement", "review", "ui_debug"):
        funcs = [f for f, s in FUNC_TO_SUBTASK.items() if s == subtask]
        for m in MODEL_POOL:
            vals = [P0.get(m, {}).get(f) for f in funcs]
            vals = [v for v in vals if v is not None]
            if vals:
                out[subtask][m] = round(sum(vals) / len(vals), 3)
    return dict(out)
