"""WeightNet — optional neural weight predictor (Section 4.11).
Requires:  pip install .[neural]
Pre-train on WIF priors:  python -m multicomm_ensemble.weightnet --pretrain
"""
from __future__ import annotations
from . import config
from .task import AgentTask, SUBTASK_TYPES

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:                                     # pragma: no cover
    _TORCH = False

_TEMPLATES = {
    "planning":  ["Decompose this feature request into implementation steps",
                  "Draft a delivery plan for the described software change"],
    "implement": ["Write a function that satisfies the following specification",
                  "Fix the failing behaviour described below and return a diff"],
    "review":    ["Review this pull request for correctness and style",
                  "Audit the following code for security and performance issues"],
    "ui_debug":  ["Diagnose why this UI element renders incorrectly",
                  "Fix the visual layout defect shown in the screenshot"],
}

if _TORCH:
    class WeightNet(nn.Module):
        def __init__(self, emb_dim=384, feat_dim=8, n_models=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(emb_dim + feat_dim, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, n_models))

        def forward(self, emb, feats):
            return self.net(torch.cat([emb, feats], dim=-1))

    def _feats(task: AgentTask) -> "torch.Tensor":
        onehot = [float(task.subtask_type == s) for s in SUBTASK_TYPES]
        return torch.tensor(onehot + [
            float(task.has_visual_artefact),
            task.context_utilisation,
            min(len(task.description) / 4000, 1.0),
            task.code_fraction], dtype=torch.float32)

    def _encoder():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")

    def predict_scores(task: AgentTask) -> dict[str, float]:
        model = WeightNet()
        model.load_state_dict(torch.load(config.WEIGHTNET_CKPT, weights_only=True))
        model.eval()
        emb = torch.tensor(_encoder().encode(task.description), dtype=torch.float32)
        with torch.no_grad():
            p = torch.sigmoid(model(emb, _feats(task)))
        return {m: float(p[i]) for i, m in enumerate(config.MODEL_POOL)}

    def pretrain(epochs: int = 30, smoothing: float = 0.05) -> None:
        """Pre-train on WIF priors (Section 4.12.3) and save the checkpoint."""
        from .wif import prior_by_subtask
        prior = prior_by_subtask()
        enc = _encoder()
        model, bce = WeightNet(), nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        rows = []
        for sub, texts in _TEMPLATES.items():
            y = torch.tensor([prior.get(sub, {}).get(m, 0.5)
                              for m in config.MODEL_POOL])
            y = y * (1 - smoothing) + 0.5 * smoothing
            for t in texts:
                task = AgentTask(description=t, subtask_type=sub)
                emb = torch.tensor(enc.encode(t), dtype=torch.float32)
                rows.append((emb, _feats(task), y))
        for _ in range(epochs):
            for emb, feats, y in rows:
                opt.zero_grad()
                loss = bce(model(emb, feats), y)
                loss.backward(); opt.step()
        torch.save(model.state_dict(), config.WEIGHTNET_CKPT)
        print(f"WeightNet pre-trained on WIF priors -> {config.WEIGHTNET_CKPT}")

else:
    def predict_scores(task: AgentTask) -> dict[str, float]:
        raise ImportError("WeightNet requires: pip install .[neural]")

    def pretrain(*a, **k):
        raise ImportError("WeightNet requires: pip install .[neural]")


if __name__ == "__main__":
    import sys
    if "--pretrain" in sys.argv:
        pretrain()
