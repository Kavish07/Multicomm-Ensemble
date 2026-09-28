# multicomm_ensemble

Multi-LLM weighted ensemble pipeline for agentic coding frameworks.
Implements the system from the dissertation *"Improving Multi-Agent Texts &
Images Communication in a Coding Framework"* (Khoodeeram, 2026):

- **WIF** — Weight Initialisation Framework: evidence-backed priors from
  `data/weight_prior_dataset.csv` (referenced leaderboard statistics)
- **Static weights** — pilot-matrix softmax (τ = 0.1)
- **WeightNet** *(optional)* — neural per-task weight prediction
- **Parallel proposers** — GPT-4o / Claude / DeepSeek / Qwen via async calls
- **Execution filter** — candidates run against the task's tests; weights rescale
- **Weight-conditioned fusion** — aggregator told which candidates to trust
- **TextRoute** — Pydantic schema validation (v1.3, `model_provenance`)
- **ModalSwitch / ImgCompress** — heuristic modality routing + Pillow compression

## Quickstart

```bash
py -m venv .venv 
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env                                  # add API keys (optional)
python -m multicomm_ensemble.demo                     # runs in MOCK mode w/o keys
pytest -q                                             # 15 tests, no network needed
```

**Mock mode** (default when no keys are set, or `MULTICOMM_MOCK=1`) runs the
entire pipeline with canned proposals — weights, execution filter, and fusion
all behave normally — so you can verify the flow at $0 before adding keys.

**Real mode**: set keys in `.env`. Any subset works; missing providers are
skipped and weights renormalise over the available pool.

**Neural extra** (WeightNet): `pip install -e .[neural]` then
`python -m multicomm_ensemble.weightnet --pretrain` to pre-train on WIF priors.

## Layout
```
multicomm_ensemble/
  config.py       env + model registry + mock switch
  task.py         AgentTask dataclass
  wif.py          prior dataset → P0 matrix (Section 4.12)
  weights.py      static matrix, softmax, source selection (Section 4.9)
  weightnet.py    optional MLP predictor (Section 4.11)
  textroute.py    schema validation (Section 4.5 / 4.9)
  modalswitch.py  heuristic modality decision
  imgcompress.py  Pillow-based visual compression
  proposers.py    async multi-model proposals (Section 4B.3)
  exec_filter.py  sandboxed test execution (Section 4B.4)
  fusion.py       weight-conditioned aggregation (Section 4B.5)
  pipeline.py     orchestrator
  demo.py         CLI demo
  adapters/       OpenCode backend + Claude Code hook
```
