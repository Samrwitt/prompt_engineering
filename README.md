Prompt Optimization Using Metaheuristics

This project presents a framework for optimizing LLM prompts—comprising instructions and few-shot examples—through metaheuristic methods such as Simulated Annealing, Differential Evolution, and Grey Wolf Optimizer (GWO). The performance of these methods is evaluated in comparison with DSPy and standard baseline approaches.

Setup

Install Dependencies

pip install -r requirements.txt


Install Ollama
Make sure Ollama is installed and running, then download the required model:

ollama pull llama3.2

Reproducing Results
1. Quick Debug Run

Use this option to confirm that the pipeline is functioning correctly (minimal budget, single seed):

python -m src.experiment --fast

2. Balanced Run (Recommended)

Provides a good trade-off between accuracy and runtime (200 calls, 3 seeds):

python -m src.experiment --balanced

3. Full Experimental Run

Intended for final research or paper-ready results (600 calls, strict budget, 5 or more seeds):

python -m src.experiment --research

Analysis

All run results are recorded in results/runs.jsonl, while validation outputs such as learning curves and scores are stored in the results/ directory.

Statistical Significance

To compute Wilcoxon signed-rank test statistics, run:

python src/stats.py


The results will be saved to results/significance.json.

Prompt Visualization

To review the highest-performing prompts discovered by the system:

python src/visualize.py
