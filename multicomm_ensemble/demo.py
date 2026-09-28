"""CLI demo:  python -m multicomm_ensemble.demo  ["task description"]"""
from __future__ import annotations
import sys
from . import config
from .task import AgentTask
from .pipeline import MultiLLMPipeline

DEFAULT_TASK = AgentTask(
    description=(
        "Implement a Python function slugify(text) that lowercases the input, "
        "replaces every run of non-alphanumeric characters with a single hyphen, "
        "and strips leading/trailing hyphens."
    ),
    subtask_type="implement",
    test_suite=[
        "def test_basic():\n    assert slugify('Hello, World!') == 'hello-world'",
        "def test_spaces():\n    assert slugify('  A  B  ') == 'a-b'",
        "def test_symbols():\n    assert slugify('C++ & Python 3!') == 'c-python-3'",
        "def test_empty():\n    assert slugify('!!!') == ''",
    ],
)


def main() -> None:
    if len(sys.argv) > 1:
        task = AgentTask(description=" ".join(sys.argv[1:]), subtask_type="implement")
    else:
        task = DEFAULT_TASK

    mode = "MOCK (no API spend)" if config.mock_mode() else \
           f"REAL — providers: {', '.join(config.available_models())}"
    print(f"\n=== multicomm_ensemble demo — {mode} ===\n")
    print(f"Task [{task.subtask_type}]: {task.description[:90]}...\n")

    result = MultiLLMPipeline().run_sync(task)

    wm = result.weight_map
    print(f"1) Weights  (source: {wm.source}, tau={wm.tau}):")
    for m, w in sorted(wm.weights.items(), key=lambda kv: -kv[1]):
        print(f"     {m:<12} {w:.3f}")

    print(f"\n2) Modality: {result.modality}"
          + (f"   img: {result.img_report}" if result.img_report else ""))

    print("\n3) Proposals + execution filter:")
    for p in result.proposals:
        status = result.exec_report.get(p.model_id, "-")
        print(f"     {p.model_id:<12} {p.latency_ms:>5} ms   "
              f"final w={p.weight:.3f}   [{status}]")

    print(f"\n4) Fusion ({result.fusion_mode}) — output:\n")
    print("   " + result.fused_output.replace("\n", "\n   "))

    print(f"\n5) TextRoute message  seq={result.routed_message['seq']}  "
          f"sha={result.routed_message['body_sha256']}")
    print(f"   provenance: {result.provenance}\n")


if __name__ == "__main__":
    main()
