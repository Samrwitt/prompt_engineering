"""Metaheuristic prompt optimization: joint search over instructions and few-shot demos."""

__version__ = "1.0.0"

from src.config import RunConfig, load_run_config
from src.model import LLMConfig, MockLLM, OllamaLLM, create_llm
from src.prompt import build_prompt, extract_answer, load_blocks, load_jsonl

__all__ = [
    "__version__",
    "RunConfig",
    "load_run_config",
    "LLMConfig",
    "MockLLM",
    "OllamaLLM",
    "create_llm",
    "build_prompt",
    "extract_answer",
    "load_blocks",
    "load_jsonl",
]
