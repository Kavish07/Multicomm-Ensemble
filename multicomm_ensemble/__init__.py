"""multicomm_ensemble — multi-LLM weighted ensemble pipeline."""
from .task import AgentTask
from .pipeline import MultiLLMPipeline, PipelineResult

__all__ = ["AgentTask", "MultiLLMPipeline", "PipelineResult"]
__version__ = "0.1.0"
