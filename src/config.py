"""Experiment configuration: YAML files plus built-in presets."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "max_llm_calls": 80,
        "fast_k": 6,
        "seeds": [0],
        "max_data": 20,
        "train_ratio": 0.5,
        "demo_pool_size": 8,
        "sa_iters": 12,
        "pop_size": 8,
        "de_iters": 6,
        "gwo_iters": 6,
        "hybrid_de": 4,
        "hybrid_sa": 8,
        "datasets": ["logic"],
        "backend": "mock",
        "skip_dspy": True,
    },
    "portfolio": {
        "max_llm_calls": 220,
        "fast_k": 16,
        "seeds": [0, 1, 2, 3, 4],
        "max_data": 100,
        "train_ratio": 0.5,
        "demo_pool_size": 16,
        "sa_iters": 28,
        "pop_size": 12,
        "de_iters": 12,
        "gwo_iters": 12,
        "hybrid_de": 8,
        "hybrid_sa": 16,
        "datasets": ["logic", "arithmetic"],
        "backend": "mock",
        "skip_dspy": True,
    },
    "balanced": {
        "max_llm_calls": 300,
        "fast_k": 10,
        "seeds": [0, 1, 2],
        "max_data": 40,
        "train_ratio": 0.7,
        "demo_pool_size": 12,
        "sa_iters": 30,
        "pop_size": 12,
        "de_iters": 15,
        "gwo_iters": 15,
        "hybrid_de": 10,
        "hybrid_sa": 20,
        "datasets": ["logic", "arithmetic"],
        "backend": "ollama",
        "skip_dspy": False,
    },
    "research": {
        "max_llm_calls": 1000,
        "fast_k": 20,
        "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "max_data": 100,
        "train_ratio": 0.5,
        "demo_pool_size": 16,
        "sa_iters": 40,
        "pop_size": 20,
        "de_iters": 30,
        "gwo_iters": 30,
        "hybrid_de": 20,
        "hybrid_sa": 40,
        "datasets": ["logic", "arithmetic", "gsm8k"],
        "backend": "ollama",
        "skip_dspy": False,
    },
}

DATASET_CATALOG: Dict[str, Dict[str, str]] = {
    "logic": {
        "path": "data/bbh_boolean_expressions.jsonl",
        "answer_type": "yesno",
        "blocks": "prompts/instruction_blocks_yesno.json",
        "label": "Boolean expressions (100 items)",
    },
    "arithmetic": {
        "path": "data/arithmetic.jsonl",
        "answer_type": "number",
        "blocks": "prompts/instruction_blocks_number.json",
        "label": "Arithmetic (100 items)",
    },
    "gsm8k": {
        "path": "data/gsm8k_sample.jsonl",
        "answer_type": "number",
        "blocks": "prompts/instruction_blocks_number.json",
        "label": "GSM8K (sample)",
    },
}


@dataclass
class RunConfig:
    name: str = "fast"
    max_llm_calls: int = 80
    fast_k: int = 6
    seeds: List[int] = field(default_factory=lambda: [0])
    max_data: Optional[int] = 20
    train_ratio: float = 0.5
    demo_pool_size: int = 8
    sa_iters: int = 12
    pop_size: int = 8
    de_iters: int = 6
    gwo_iters: int = 6
    hybrid_de: int = 4
    hybrid_sa: int = 8
    datasets: List[str] = field(default_factory=lambda: ["logic"])
    backend: str = "mock"
    skip_dspy: bool = True
    results_dir: str = "results"
    max_demos: int = 5
    early_stop: bool = True

    def dataset_specs(self) -> List[Tuple[str, Dict[str, str]]]:
        specs = []
        for name in self.datasets:
            if name not in DATASET_CATALOG:
                raise KeyError(f"Unknown dataset '{name}'. Choose from: {list(DATASET_CATALOG)}")
            specs.append((name, DATASET_CATALOG[name]))
        return specs


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load config files. pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return data


def load_run_config(
    preset: str = "fast",
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> RunConfig:
    if preset not in PRESETS:
        raise KeyError(f"Unknown preset '{preset}'. Choose from: {list(PRESETS)}")
    data: Dict[str, Any] = {"name": preset, **PRESETS[preset]}
    if config_path:
        data.update(_load_yaml(Path(config_path)))
        data["name"] = Path(config_path).stem
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                data[k] = v
    known = {f.name for f in RunConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known}
    return RunConfig(**filtered)
