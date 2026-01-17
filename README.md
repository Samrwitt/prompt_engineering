# Prompt Engineering Toolkit

A lightweight research toolkit for experimenting with prompt engineering techniques and prompt-search algorithms.

This repository contains reusable scripts, datasets, and local-model artifacts designed to **systematically evaluate, optimize, and compare prompts** using classical optimization methods such as Genetic Algorithms (GA), Particle Swarm Optimization (PSO), and Simulated Annealing (SA).

The goal is to move beyond ad-hoc prompt tweaking and toward **repeatable, measurable prompt experimentation**.

---

## Motivation

Prompt engineering is often treated as a trial-and-error process driven by intuition.
This project explores a more structured approach: **searching the prompt space using optimization algorithms combined with explicit fitness functions**.

By formalizing prompt evaluation, the repository enables:

* reproducible experiments,
* objective comparison between prompt variants, and
* offline evaluation using local language models.

---

## Repository Structure

```
.
├── data/                  # JSONL datasets used for evaluation (arithmetic, logic, reasoning)
├── models/
│   └── flan-t5-base/       # Local model files and tokenizers for offline inference
├── prompts/               # Example prompt templates and prompt blocks
├── results/               # Generated experiment outputs (scores, curves, plots)
├── src/
│   ├── model.py            # Local model inference wrapper
│   ├── dspy_local.py       # DSPy-compatible local model interface
│   ├── fitness.py          # Prompt evaluation and scoring functions
│   ├── ga.py               # Genetic Algorithm optimizer
│   ├── pso.py              # Particle Swarm Optimization
│   ├── sa.py               # Simulated Annealing optimizer
│   ├── experiment.py       # Experiment orchestration logic
│   ├── generate_data.py    # Dataset generation utilities
│   ├── baseline.py         # Baseline evaluation helpers
│   └── sota_dspy.py        # SOTA/DSPy-based evaluation helpers
├── debug_model.py          # Quick sanity checks for model loading and inference
├── test_dspy.py            # Lightweight test harness used during development
├── verify_fix.py           # Verification helper scripts
├── verify_debug.py         # Debug verification utilities
└── requirements.txt
```

---

## Requirements

* Python **3.10+** recommended
* Install dependencies using:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Note: The repository includes a local `models/flan-t5-base/` directory.
> Ensure any code expecting a local model path points to this folder, or update the path configuration in `src/model.py`.

---

## Quick Start

### Run a smoke test

```bash
python debug_model.py
python test_dspy.py
```

### Run an experiment

```bash
python -m src.experiment
```

Most scripts expose configurable parameters at the top of the file.
Adjust dataset paths, prompt sources, or optimizer settings as needed.

---

## Typical Workflow

1. Prepare or generate datasets under `data/`
2. Define or modify prompt templates in `prompts/`
3. Select and run an optimizer (GA, PSO, or SA)
4. Evaluate prompt fitness using defined scoring functions
5. Inspect numeric results and plots generated in `results/`

---

## Inputs and Outputs

**Inputs**

* JSONL datasets in `data/`
* Local model files in `models/`
* Optional prompt definitions in `prompts/`

**Outputs**

* Numeric fitness scores
* CSV/JSON experiment logs
* Plots and evaluation artifacts written to `results/`

**Common error cases**

* Missing or misconfigured local model files
* Tokenizer/model mismatches
* Malformed JSONL datasets

These issues will typically surface at startup; verifying paths early is recommended.

---

## Project Status

This repository is an **experimental research playground**.
APIs, directory structure, and evaluation logic may evolve as experiments grow and new ideas are tested.

---

## Roadmap

* [ ] Add a small CLI (argparse) for common experiment workflows
* [ ] Improve reproducibility with pinned dependency versions
* [ ] Add a minimal unit test validating local model loading
* [ ] Extend support to additional local or quantized models

---

## Notes

If you’d like, this repository can be extended with:

* a `Makefile` or task runner for common commands,
* a short tutorial notebook demonstrating a full experiment run, or
* improved experiment configuration via YAML/JSON.

---

### Suggested commit message

```bash
git commit -m "Rewrite README to clarify motivation, workflow, and project scope"
```
