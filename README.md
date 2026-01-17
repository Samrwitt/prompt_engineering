## prompt_engineering

Lightweight toolkit and experiments for prompt engineering and prompt-search algorithms.

This repository contains scripts, datasets and local-model artifacts used to evaluate and evolve prompts (GA/PSO/SA), measure prompt fitness, and run repeatable experiments.

### Contents

- `data/` — JSONL datasets used by experiments (arithmetic, logic, comparative_reasoning).
- `models/flan-t5-base/` — local model files/tokenizers used for offline evaluation.
- `prompts/` — example prompt blocks used by the experiments.
- `results/` — generated curves and final scores from past runs.
- `src/` — core library and experiment code:
  - `model.py`, `dspy_local.py` — model inference wrapper(s)
  - `fitness.py` — fitness/evaluation functions for prompts
  - `ga.py`, `pso.py`, `sa.py` — search/optimizers (genetic algorithm, particle swarm, simulated annealing)
  - `experiment.py`, `generate_data.py` — orchestration and dataset generation utilities
  - `baseline.py`, `sota_dspy.py` — baseline and SOTA evaluation helpers
- Top-level scripts:
  - `debug_model.py` — quick model debug/run script
  - `test_dspy.py` — small test harness used during development
  - `verify_fix.py` / `verify_debug.py` — verification helpers

### Requirements

- Python 3.10+ recommended.
- Install runtime deps from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: the repository includes a local `models/flan-t5-base/` folder; ensure any code that expects a local model path is pointed there or change the configuration in `src/model.py`.

### Quick start

- Run a simple debug or smoke test:

```bash
python debug_model.py
python test_dspy.py
```

- Run an experiment (example):

```bash
python -m src.experiment
```

Adjust entry points or pass arguments as needed — check the top of each script for CLI flags.

### Contract / expected I/O

- Inputs: dataset JSONL files in `data/`, local model files under `models/`, optional prompt JSON in `prompts/`.
- Outputs: numeric fitness values, CSV/JSON experiment results and plots written to `results/`.
- Error modes: missing model files, mismatched tokenizers, or malformed dataset JSONL will raise exceptions at startup — verify paths first.

### Tests & quality gates

- Fast checks: `python test_dspy.py` (small unit/functional smoke)
- Recommended: run linting and static-type checks if you add/modify code.

### Notes & next steps

- If you want a reproducible environment for experiments, pin exact package versions in a `requirements.lock` or use `pip-tools`/`poetry`.
- Consider adding a small CLI in `src/experiment.py` (argparse) to simplify common workflows.

If you'd like, I can also:
- add a simple `Makefile` or `tasks.json` for common commands;
- add a minimal test that validates model loading; or
- create a short tutorial `notebook/` that walks through a single experiment run.



