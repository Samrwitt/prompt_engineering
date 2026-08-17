from src.config import load_run_config
from src.experiment import config_from_args, parse_args


def test_fast_preset_defaults_to_mock():
    cfg = load_run_config("fast")
    assert cfg.backend == "mock"
    assert cfg.max_llm_calls == 100
    assert "logic" in cfg.datasets


def test_cli_fast_flag():
    args = parse_args(["--fast", "--backend", "mock", "--datasets", "logic"])
    cfg = config_from_args(args)
    assert cfg.name == "fast"
    assert cfg.backend == "mock"
    assert cfg.datasets == ["logic"]


def test_cli_balanced_seeds():
    args = parse_args(["--balanced", "--seeds", "0,2"])
    cfg = config_from_args(args)
    assert cfg.seeds == [0, 2]
    assert cfg.name == "balanced"
