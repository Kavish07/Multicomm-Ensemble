"""Execution-grounded re-weighting (Section 4B.4).

Runs each candidate against the task's pytest suite in an isolated temp dir
subprocess (or Docker if EXEC_SANDBOX=docker and Docker is available), then
rescales weights by pass rate. Candidates with no code or 0 passes drop out.
"""
from __future__ import annotations
import os, re, subprocess, sys, tempfile
from dataclasses import dataclass
from .task import AgentTask
from .proposers import Proposal

TIMEOUT = 30


@dataclass
class TestResult:
    passed: int
    failed: int

    @property
    def total(self) -> int:
        return self.passed + self.failed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


def extract_code_block(content: str) -> str | None:
    m = re.search(r"```(?:\w+)?\n(.*?)```", content, re.DOTALL)
    return m.group(1).strip() if m else None


def _run_subprocess(code: str, tests: list[str]) -> TestResult:
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "solution.py"), "w").write(code)
        open(os.path.join(d, "test_solution.py"), "w").write(
            "from solution import *\n" + "\n".join(tests))
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "test_solution.py", "-q",
             "--no-header", "-p", "no:cacheprovider"],
            cwd=d, capture_output=True, text=True, timeout=TIMEOUT)
        out = r.stdout + r.stderr
        passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
        failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
        if r.returncode not in (0, 1):           # collection error etc.
            return TestResult(0, max(len(tests), 1))
        return TestResult(passed, failed)


def _run_docker(code: str, tests: list[str]) -> TestResult:
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "solution.py"), "w").write(code)
        open(os.path.join(d, "test_solution.py"), "w").write(
            "from solution import *\n" + "\n".join(tests))
        r = subprocess.run(
            ["docker", "run", "--rm", "--network=none",
             f"--volume={d}:/w", "--workdir=/w", "python:3.11-slim",
             "sh", "-c", "pip install pytest -q && pytest test_solution.py -q"],
            capture_output=True, text=True, timeout=120)
        out = r.stdout + r.stderr
        passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
        failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
        return TestResult(passed, failed)


def run_tests(code: str, tests: list[str]) -> TestResult:
    if os.getenv("EXEC_SANDBOX", "").lower() == "docker":
        try:
            return _run_docker(code, tests)
        except Exception:
            pass
    return _run_subprocess(code, tests)


def execution_reweight(proposals: list[Proposal], task: AgentTask
                       ) -> tuple[list[Proposal], dict[str, str]]:
    """Rescale weights by test pass rate. Returns (proposals, per-model report)."""
    report: dict[str, str] = {}
    if task.subtask_type not in ("implement", "ui_debug") or not task.test_suite:
        return proposals, {p.model_id: "skipped (no tests / wrong subtask)"
                           for p in proposals}
    for p in proposals:
        if not p.ok:
            p.weight = 0.0
            report[p.model_id] = f"provider error: {p.error}"
            continue
        code = extract_code_block(p.content)
        if code is None:
            p.weight = 0.0
            report[p.model_id] = "no code block -> excluded"
            continue
        res = run_tests(code, task.test_suite)
        p.weight *= res.pass_rate
        report[p.model_id] = f"{res.passed}/{res.total} tests passed"
    total = sum(p.weight for p in proposals)
    if total > 0:
        for p in proposals:
            p.weight = round(p.weight / total, 4)
    return proposals, report
