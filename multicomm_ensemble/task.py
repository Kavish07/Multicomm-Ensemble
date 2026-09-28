"""AgentTask — the unit of work flowing through the pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field

SUBTASK_TYPES = ("planning", "implement", "review", "ui_debug")


@dataclass
class AgentTask:
    description: str
    subtask_type: str = "implement"
    test_suite: list[str] = field(default_factory=list)  # pytest-compatible sources
    image: bytes | None = None                           # raw screenshot, if any
    context_utilisation: float = 0.0                     # 0..1
    task_id: str = "task-001"

    def __post_init__(self):
        if self.subtask_type not in SUBTASK_TYPES:
            raise ValueError(
                f"subtask_type must be one of {SUBTASK_TYPES}, got {self.subtask_type!r}"
            )

    @property
    def has_visual_artefact(self) -> bool:
        return self.image is not None

    @property
    def code_fraction(self) -> float:
        lines = self.description.splitlines() or [""]
        codeish = sum(
            1 for l in lines
            if l.strip().startswith(("def ", "class ", "import ", ">>>", "```"))
            or l.startswith(("    ", "\t"))
        )
        return codeish / len(lines)
