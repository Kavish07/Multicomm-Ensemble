from multicomm_ensemble.wif import load_dataset, prior_matrix, prior_by_subtask, recency


def test_dataset_loads():
    stats = load_dataset()
    assert len(stats) >= 20
    assert all(0.0 <= s.score <= 1.5 for s in stats)   # Elo cluster rows normalised to 1.0


def test_prior_matrix_covers_pool():
    P0 = prior_matrix()
    for m in ("gpt4o", "claude_s46", "deepseek", "qwen35"):
        assert m in P0 and len(P0[m]) >= 3


def test_ratio_estimator_fills_missing_cells():
    P0 = prior_matrix()
    # claude has no direct LiveCodeBench row in the dataset -> must be estimated
    assert "algorithmic_implementation" in P0["claude_s46"]
    assert 0.5 < P0["claude_s46"]["algorithmic_implementation"] <= 1.0


def test_subtask_collapse():
    ps = prior_by_subtask()
    for sub in ("planning", "implement", "review", "ui_debug"):
        assert sub in ps and len(ps[sub]) == 4


def test_recency_decay_monotone():
    assert recency("2026-07-01") > recency("2025-07-01")
