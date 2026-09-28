"""MultiLLMPipeline — the orchestrator wiring every stage together."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from . import config
from .task import AgentTask
from .weights import derive_weights, WeightMap
from .modalswitch import decide as modal_decide
from .imgcompress import encode as img_encode
from .proposers import run_proposers, Proposal
from .exec_filter import execution_reweight
from .fusion import fuse
from .textroute import RoutedMessage, ProvenanceEntry, validate, enrich


@dataclass
class PipelineResult:
    task_id: str
    fused_output: str
    fusion_mode: str
    weight_map: WeightMap
    proposals: list[Proposal]
    exec_report: dict[str, str]
    modality: str
    img_report: dict | None
    routed_message: dict
    mock: bool = field(default_factory=config.mock_mode)

    @property
    def provenance(self) -> list[dict]:
        return [{"model": p.model_id, "weight": round(p.weight, 3)}
                for p in sorted(self.proposals, key=lambda x: -x.weight)
                if p.weight > 0]


class MultiLLMPipeline:
    def __init__(self):
        self._seq = 0

    async def run(self, task: AgentTask) -> PipelineResult:
        # 1. Weight derivation (WIF prior / static / WeightNet)
        wm = derive_weights(task)

        # 2. Modality decision + visual compression (per top-weight receiver)
        top_model = max(wm.weights, key=wm.weights.get)
        modality = modal_decide(task, receiver_model=top_model)
        image, img_report = None, None
        if modality == "hybrid" and task.image:
            image, img_report = img_encode(task.image, receiver_model=top_model)

        # 3. Parallel proposals
        proposals = await run_proposers(task, wm, image=image)

        # 4. Execution-grounded re-weighting
        proposals, exec_report = execution_reweight(proposals, task)

        # 5. Weight-conditioned fusion
        fused, fusion_mode = fuse(task.description, proposals)

        # 6. TextRoute-validated outgoing message with provenance
        self._seq += 1
        msg = validate({
            "sender_role": "ensemble",
            "receiver_role": "environment",
            "intent": "inform",
            "priority": 3,
            "body": fused[:8000],
            "expects_reply": False,
            "model_provenance": [
                ProvenanceEntry(model=p.model_id, weight=round(p.weight, 3)).model_dump()
                for p in proposals if p.weight > 0
            ],
        })
        routed = enrich(msg, self._seq)

        return PipelineResult(
            task_id=task.task_id, fused_output=fused, fusion_mode=fusion_mode,
            weight_map=wm, proposals=proposals, exec_report=exec_report,
            modality=modality, img_report=img_report, routed_message=routed)

    def run_sync(self, task: AgentTask) -> PipelineResult:
        return asyncio.run(self.run(task))
