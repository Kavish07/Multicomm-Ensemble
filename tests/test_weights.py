import pytest
from multicomm_ensemble.task import AgentTask
from multicomm_ensemble.weights import derive_weights, STATIC_MATRIX, _softmax


def test_softmax_normalises():
    w = _softmax(STATIC_MATRIX["ui_debug"], tau=0.1)
    assert abs(sum(w.values()) - 1.0) < 1e-3
    assert w["claude_s46"] > w["deepseek"]        # 0.80 vs 0.58 pilot


def test_derive_weights_wif_source(monkeypatch):
    monkeypatch.setenv("MULTICOMM_MOCK", "1")
    from multicomm_ensemble import config
    monkeypatch.setattr(config, "WEIGHT_SOURCE", "wif")
    wm = derive_weights(AgentTask(description="x", subtask_type="implement"))
    assert wm.source == "wif"
    assert abs(sum(wm.weights.values()) - 1.0) < 1e-3
    assert wm.weights["deepseek"] == max(wm.weights.values())   # LCB 0.935 dominates


def test_static_fallback(monkeypatch):
    monkeypatch.setenv("MULTICOMM_MOCK", "1")
    from multicomm_ensemble import config
    monkeypatch.setattr(config, "WEIGHT_SOURCE", "static")
    wm = derive_weights(AgentTask(description="x", subtask_type="review"))
    assert wm.source == "static"
    assert wm.weights["claude_s46"] == max(wm.weights.values())


def test_invalid_subtask_rejected():
    with pytest.raises(ValueError):
        AgentTask(description="x", subtask_type="banana")
