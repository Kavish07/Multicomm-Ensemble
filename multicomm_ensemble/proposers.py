"""Parallel proposers (Section 4B.3) with a zero-cost mock mode."""
from __future__ import annotations
import asyncio, os, time
from dataclasses import dataclass
from . import config
from .task import AgentTask

ROLE_PROMPTS = {
    "planning":  "You are a senior software planner. Decompose the task into a short, numbered implementation plan.",
    "implement": "You are an expert programmer. Solve the task. Reply with a single fenced Python code block and nothing else.",
    "review":    "You are a rigorous code reviewer. Review the provided work and give a verdict with concrete findings.",
    "ui_debug":  "You are a front-end debugging expert. Diagnose the visual defect and reply with a fenced code block containing the fix.",
}


@dataclass
class Proposal:
    model_id: str
    content: str
    weight: float
    latency_ms: int
    ok: bool = True
    error: str | None = None


# ── Mock proposals: exercise the whole pipeline without API keys ──────────────
_MOCK_SNIPPETS = {
    "gpt4o": '```python\nimport re\n\ndef slugify(text):\n    text = text.lower().strip()\n    text = re.sub(r"[^a-z0-9]+", "-", text)\n    return text.strip("-")\n```',
    "claude_s46": '```python\nimport re\n\ndef slugify(text: str) -> str:\n    """Lowercase, replace non-alphanumerics with single hyphens, trim."""\n    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")\n```',
    "deepseek": '```python\nimport re\n\ndef slugify(text):\n    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-")\n```',
    # deliberately broken candidate: demonstrates the execution filter
    "qwen35": '```python\ndef slugify(text):\n    return text.lower().replace(" ", "_")\n```',
}


async def _mock_call(model_id: str, task: AgentTask) -> str:
    await asyncio.sleep(0.05)
    if task.subtask_type == "implement":
        return _MOCK_SNIPPETS[model_id]
    return f"[mock:{model_id}] {task.subtask_type} response for: {task.description[:60]}"


# ── Real provider calls ───────────────────────────────────────────────────────
async def _openai_call(model_id: str, task: AgentTask, image: bytes | None) -> str:
    from openai import AsyncOpenAI
    p = config.PROVIDERS[model_id]
    client = AsyncOpenAI(api_key=os.getenv(p["key_env"]), base_url=p["base_url"])
    content: list | str = task.description
    if image is not None:
        import base64
        content = [
            {"type": "text", "text": task.description},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(image).decode()}},
        ]
    r = await client.chat.completions.create(
        model=p["model"], max_tokens=2048,
        messages=[{"role": "system", "content": ROLE_PROMPTS[task.subtask_type]},
                  {"role": "user", "content": content}])
    return r.choices[0].message.content or ""


async def _anthropic_call(model_id: str, task: AgentTask, image: bytes | None) -> str:
    from anthropic import AsyncAnthropic
    p = config.PROVIDERS[model_id]
    client = AsyncAnthropic(api_key=os.getenv(p["key_env"]))
    blocks: list = [{"type": "text", "text": task.description}]
    if image is not None:
        import base64
        blocks.insert(0, {"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(image).decode()}})
    r = await client.messages.create(
        model=p["model"], max_tokens=2048,
        system=ROLE_PROMPTS[task.subtask_type],
        messages=[{"role": "user", "content": blocks}])
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


async def _call_one(model_id: str, weight: float, task: AgentTask,
                    image: bytes | None) -> Proposal:
    t0 = time.monotonic()
    try:
        if config.mock_mode():
            content = await _mock_call(model_id, task)
        elif config.PROVIDERS[model_id]["kind"] == "anthropic":
            content = await _anthropic_call(model_id, task, image)
        else:
            content = await _openai_call(model_id, task, image)
        return Proposal(model_id, content, weight,
                        int((time.monotonic() - t0) * 1000))
    except Exception as e:                       # provider failure ≠ pipeline failure
        return Proposal(model_id, "", 0.0,
                        int((time.monotonic() - t0) * 1000),
                        ok=False, error=f"{type(e).__name__}: {e}")


async def run_proposers(task: AgentTask, weight_map, image: bytes | None = None
                        ) -> list[Proposal]:
    active = weight_map.active()
    props = await asyncio.gather(
        *[_call_one(m, w, task, image) for m, w in active.items()])
    return list(props)
