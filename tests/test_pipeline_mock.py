"""End-to-end pipeline in mock mode — exercises every stage without network."""
import pytest
from multicomm_ensemble.task import AgentTask
from multicomm_ensemble.pipeline import MultiLLMPipeline

TASK = AgentTask(
    description="Implement slugify(text): lowercase, hyphen-join alphanumerics, trim.",
    subtask_type="implement",
    test_suite=[
        "def test_basic():\n    assert slugify('Hello, World!') == 'hello-world'",
        "def test_symbols():\n    assert slugify('C++ & Py 3!') == 'c-py-3'",
    ],
)


@pytest.fixture(autouse=True)
def _mock(monkeypatch):
    monkeypatch.setenv("MULTICOMM_MOCK", "1")


def test_end_to_end_mock():
    result = MultiLLMPipeline().run_sync(TASK)
    # the deliberately broken qwen35 mock must be zeroed by the exec filter
    weights = {p.model_id: p.weight for p in result.proposals}
    assert weights["qwen35"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
    assert "slugify" in result.fused_output
    assert result.routed_message["schema_version"] == "1.3"
    assert all(e["model"] != "qwen35" for e in result.provenance)


def test_text_only_modality():
    result = MultiLLMPipeline().run_sync(TASK)
    assert result.modality == "text" and result.img_report is None
