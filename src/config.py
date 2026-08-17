"""Experiment configuration: YAML files plus built-in presets."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "max_llm_calls": 100,
        "fast_k": 5,
        "seeds": [0],
        "max_data": 10,
        "sa_iters": 20,
        "pop_size": 8,
        "de_iters": 10,
        "gwo_iters": 10,
        "hybrid_de": 10,
        "hybrid_sa": 10,
        "datasets": ["logic"],
    },
    "balanced": {
        "max_llm_calls": 300,
        "fast_k": 10,
        "seeds": [0, 1, 2],
        "max_data": 20,
        "sa_iters": 30,
        "pop_size": 12,
        "de_iters": 15,
        "gwo_iters": 15,
        "hybrid_de": 10,
        "hybrid_sa": 20,
        "datasets": ["logic", "arithmetic"],
    },
    "research": {
        "max_llm_calls": 1000,
        "fast_k": 20,
        "seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "max_data": 100,
        "sa_iters": 40,
        "pop_size": 20,
        "de_iters": 30,
        "gwo_iters": 30,
        "hybrid_de": 20,
        "hybrid_sa": 40,
        "datasets": ["logic", "arithmetic", "gsm8k"],
    },
}

DATASET_CATALOG: Dict[str, Dict[str, str]] = {
    "logic": {
        "path": "data/bbh_boolean_expressions.jsonl",
        "answer_type": "yesno",
        "blocks": "prompts/instruction_blocks_yesno.json",
        "label": "BBH Boolean Expressions",
    },
    "arithmetic": {
        "path": "data/arithmetic.jsonl",
        "answer_type": "number",
        "blocks": "prompts/instruction_blocks_number.json",
        "label": "Arithmetic Word Problems",
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
    max_llm_calls: int = 100
    fast_k: int = 5
    seeds: List[int] = field(default_factory=lambda: [0])
    max_data: Optional[int] = 10
    sa_iters: int = 20
    pop_size: int = 8
    de_iters: int = 10
    gwo_iters: int = 10
    hybrid_de: int = 10
    hybrid_sa: int = 10
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
