"""OpenCode model-agnostic backend adapter (Section 4.10).

Register:
  opencode --backend python:multicomm_ensemble.adapters.opencode:OpenCodeAdapter
"""
from __future__ import annotations
from ..task import AgentTask
from ..pipeline import MultiLLMPipeline


def _classify(prompt: str) -> str:
    p = prompt.lower()
    if any(k in p for k in ("plan", "design", "architecture", "decompose")):
        return "planning"
    if any(k in p for k in ("review", "audit", "check this")):
        return "review"
    if any(k in p for k in ("ui", "css", "layout", "render", "screenshot")):
        return "ui_debug"
    return "implement"


class OpenCodeAdapter:
    def __init__(self, pipeline: MultiLLMPipeline | None = None):
        self.pipeline = pipeline or MultiLLMPipeline()

    async def complete(self, request: dict) -> dict:
        prompt = request.get("prompt") or request.get("messages", [{}])[-1].get("content", "")
        task = AgentTask(description=prompt, subtask_type=_classify(prompt))
        result = await self.pipeline.run(task)
        return {
            "content": result.fused_output,
            "model_provenance": result.provenance,
            "fusion_mode": result.fusion_mode,
        }
