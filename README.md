# Prompt Engineering with Metaheuristic Optimization

A comprehensive research project comparing metaheuristic optimization algorithms for automatic prompt engineering. This project evaluates Genetic Algorithms (GA), Simulated Annealing (SA), Particle Swarm Optimization (PSO), and baseline methods (including DSPy SOTA) for selecting optimal prompt blocks to improve LLM performance.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Methodology](#methodology)

## 🎯 Overview

This project addresses the challenge of prompt engineering by treating it as a binary optimization problem. Each prompt is composed of multiple "blocks" (instructional components), and the goal is to find the optimal combination of blocks that maximizes LLM accuracy on a given task.

### Key Research Questions
- Which metaheuristic algorithm performs best for prompt optimization?
- Which prompt blocks are most effective for different task types?
- How do optimized prompts compare to state-of-the-art methods like DSPy?

## ✨ Features

- **Multiple Optimization Algorithms**: GA, SA, PSO, Random Search, Greedy Search
- **DSPy SOTA Baseline**: Integration with DSPy's COPRO optimizer for comparison
- **Block Effectiveness Analysis**: Automatic analysis of which prompt blocks contribute most to performance
- **Multiple Datasets**: Arithmetic, Logic, and Comparative Reasoning tasks
- **Comprehensive Evaluation**: Train/test splits with detailed metrics
- **Convergence Tracking**: Save and analyze optimization curves

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Navigate to Project

```bash
cd prompt_engineering
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first run will download model weights from HuggingFace:
- `google/flan-t5-small` (~250MB)
- `Qwen/Qwen2.5-Math-1.5B-Instruct` (~3GB)

These downloads happen automatically on first use.

### Optional: DSPy Installation

For DSPy SOTA baseline (optional but recommended):

```bash
pip install dspy-ai
```

## 📖 Usage

### Basic Usage: Run Full Experiment

Run the complete experiment across all datasets:

```bash
python -m src.experiment
```

This will:
1. Load all three datasets (arithmetic, logic, comparative)
2. Run all optimization algorithms
3. Perform block effectiveness analysis
4. Run DSPy SOTA baseline (if installed)
5. Save results to `results/final_scores.json` and `results/curves.json`

### Expected Runtime

- **Small models (CPU)**: ~30-60 minutes for full experiment
- **GPU available**: Significantly faster (set `device="cuda"` in `experiment.py`)

### Output Files

- `results/final_scores.json`: Complete results with train/test accuracies for all methods
- `results/curves.json`: Convergence curves for metaheuristic algorithms
- Console output: Real-time progress and analysis

### Example Output

```
======================================================================
                    Prompt Optimization Experiment                    
======================================================================
Comparing metaheuristic algorithms for prompt block selection
======================================================================

Initializing models...
  ✓ FLAN-T5-small loaded (for yes/no tasks)
  ✓ Qwen2.5-Math loaded (for arithmetic tasks)

======================================================================
                          Dataset: ARITHMETIC                          
======================================================================
Loaded 8 prompt blocks from prompts/blocks_number.json
Answer type: number
Dataset: 35 total | 28 train | 7 test

======================================================================
                    Block Effectiveness Analysis                      
======================================================================
...

[1/6] Deterministic Baseline (DSPy-inspired)...
  ✓ Train: 0.0000 | Test: 0.0000

[2/6] Random Baseline...
  ✓ Train: 0.0357 | Test: 0.0000
...
```

### Advanced Usage

#### Run Without Block Analysis

Edit `src/experiment.py` or modify the main function call:

```python
if __name__ == "__main__":
    main(run_block_analysis=False, run_dspy=True, save_curves=True)
```

#### Test Individual Components

**Test Model Loading:**
```bash
python debug_model.py
```

**Verify Prompt Extraction:**
```bash
python verify_fix.py
```

**Generate New Datasets:**
```bash
python -m src.generate_data
```

## 📁 Project Structure

```
prompt_engineering/
├── src/
│   ├── __init__.py
│   ├── experiment.py          # Main experiment runner
│   ├── model.py               # LLM wrapper (HFLLM)
│   ├── fitness.py             # Evaluation framework
│   ├── baseline.py            # Baseline methods (DSPy-inspired, random, greedy)
│   ├── ga.py                  # Genetic Algorithm
│   ├── sa.py                  # Simulated Annealing
│   ├── pso.py                 # Particle Swarm Optimization
│   ├── sota_dspy.py           # DSPy SOTA integration
│   ├── block_analysis.py      # Block effectiveness analysis
│   └── generate_data.py       # Dataset generation
├── data/
│   ├── arithmetic.jsonl       # Math problems (number answers)
│   ├── logic.jsonl            # Boolean logic (yes/no)
│   └── comparative_reasoning.jsonl  # Comparisons (yes/no)
├── prompts/
│   ├── blocks_number.json     # Prompt blocks for numeric tasks
│   └── blocks_yesno.json      # Prompt blocks for yes/no tasks
├── results/
│   ├── final_scores.json      # Experiment results
│   └── curves.json            # Convergence curves
├── requirements.txt
├── README.md
└── verify_fix.py              # Unit tests
```

## 📊 Results

Results are saved in `results/final_scores.json` with the following structure:

```json
{
  "arithmetic": {
    "baseline": {"train": 0.0, "test": 0.0, "x": [...]},
    "random": {"train": 0.04, "test": 0.0, "x": [...]},
    "greedy": {"train": 0.0, "test": 0.0, "x": [...]},
    "sa": {"train": 0.04, "test": 0.0, "x": [...]},
    "ga": {"train": 0.04, "test": 0.0, "x": [...]},
    "pso": {"train": 0.04, "test": 0.0, "x": [...]},
    "dspy_sota": {"train": 0.0, "test": 0.0},
    "best_method": "ga",
    "block_analysis": {...}
  },
  ...
}
```

### Interpreting Results

- **`x`**: Binary vector indicating which prompt blocks are selected (1 = included, 0 = excluded)
- **`train`/`test`**: Accuracy scores on train and test sets
- **`block_analysis`**: Individual block effectiveness scores
- **`best_method`**: Algorithm with highest test accuracy

## 🔬 Methodology

### Optimization Problem

Given:
- Set of prompt blocks: `B = {b₁, b₂, ..., bₙ}`
- Binary selection vector: `x ∈ {0,1}ⁿ`
- Fitness function: `f(x) = accuracy(LLM(prompt(x), dataset))`

Find: `x* = argmax f(x)`

### Algorithms

1. **Deterministic Baseline**: Rule-based selection (DSPy-inspired)
2. **Random Baseline**: Random search over prompt combinations
3. **Greedy Baseline**: Hill-climbing from deterministic baseline
4. **Simulated Annealing**: Probabilistic acceptance of worse solutions
5. **Genetic Algorithm**: Population-based evolution with crossover/mutation
6. **Particle Swarm Optimization**: Swarm intelligence with velocity updates
7. **DSPy SOTA**: COPRO optimizer (state-of-the-art prompt optimization)

### Evaluation

- **Train/Test Split**: 80/20 split with fixed seed (42)
- **Metrics**: Exact match accuracy (normalized by answer type)
- **Caching**: LLM outputs cached to avoid redundant calls

## 🛠️ Troubleshooting

### Model Download Issues

If models fail to download:
1. Check internet connection
2. Verify HuggingFace access (some models may require login)
3. Try downloading manually: `huggingface-cli download google/flan-t5-small`

### DSPy Import Errors

If you see `DSPy not available`:
```bash
pip install dspy-ai
```

### Memory Issues

For large models on CPU:
- Reduce `max_new_tokens` in `LLMConfig`
- Use smaller models
- Process datasets in batches

### Slow Performance

- Use GPU: Set `device="cuda"` in `experiment.py`
- Reduce iterations: Lower `iters`, `generations`, `pop_size` parameters
- Use smaller datasets: Modify `generate_data.py` to create fewer examples

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{prompt_engineering_metaheuristics,
  title={Prompt Engineering with Metaheuristic Optimization},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/prompt-engineering}
}
```

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

This is a research project. For questions or improvements, please open an issue or submit a pull request.

## 🙏 Acknowledgments

- HuggingFace for model hosting
- DSPy team for the COPRO optimizer
- The open-source community for optimization algorithm implementations

---

**Last Updated**: 2024
**Version**: 1.0

