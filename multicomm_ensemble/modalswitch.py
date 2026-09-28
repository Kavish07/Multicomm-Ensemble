"""ModalSwitch — modality decision (heuristic implementation).

The dissertation's PPO policy (Section 4.3) requires the annotated log corpus
to train; this deployable version implements the same decision surface as
transparent rules, conditioned on the receiver's visual capability
(model-aware variant, Section 4B.6).
"""
from __future__ import annotations
from .task import AgentTask

PILOT_VISUAL = {"gpt4o": 0.78, "claude_s46": 0.81, "deepseek": 0.44, "qwen35": 0.61}


def decide(task: AgentTask, receiver_model: str | None = None) -> str:
    """Return 'text' | 'hybrid'. ('image'-only is never optimal in this pipeline.)"""
    if not task.has_visual_artefact:
        return "text"
    if task.context_utilisation > 0.8:          # protect the context window
        return "text"
    visual = PILOT_VISUAL.get(receiver_model or "", 0.70)
    if visual < 0.50:                           # weak visual reasoner: image adds noise
        return "text"
    return "hybrid"
