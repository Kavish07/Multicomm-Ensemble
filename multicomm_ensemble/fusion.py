"""Weight-conditioned fusion (Section 4B.5)."""
from __future__ import annotations
import os
from . import config
from .proposers import Proposal

FUSION_SYSTEM = """You are a software engineering synthesis specialist.
Candidates below carry reliability weights (0-1) derived from benchmark
performance and test execution outcomes.
Rules: (1) Highest-weight candidates take precedence in all conflicts.
(2) Incorporate non-conflicting correct elements from lower-weight candidates.
(3) Discard elements appearing only in candidates with weight < 0.1 unless unique.
(4) Output must be complete and immediately usable - no placeholders."""


def _build_prompt(task_desc: str, active: list[Proposal]) -> str:
    body = "\n\n".join(
        f"--- Candidate [{p.model_id}] weight={p.weight:.3f} ---\n{p.content}"
        for p in active)
    return (f"Task:\n{task_desc}\n\nCandidates "
            f"(highest weight = most authoritative):\n{body}\n\nFused solution:")


def fuse(task_desc: str, proposals: list[Proposal]) -> tuple[str, str]:
    """Returns (fused_output, fusion_mode)."""
    active = sorted((p for p in proposals if p.weight > 0 and p.ok),
                    key=lambda p: p.weight, reverse=True)
    if not active:
        return "[fusion] no viable candidates survived the execution filter", "empty"
    if len(active) == 1 or config.mock_mode():
        return active[0].content, "top-weight (mock/single)"
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv(config.AGGREGATOR["key_env"]))
    r = client.messages.create(
        model=config.AGGREGATOR["model"], max_tokens=4096,
        system=FUSION_SYSTEM,
        messages=[{"role": "user", "content": _build_prompt(task_desc, active)}])
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    return text, "weight-conditioned aggregator"
