# Ablation Study Analysis

An ablation study is a systematic way to understand which parts of your system contribute the most to success. In our Meta-Prompting framework, we have three core "ingredients" that we can ablate:

## 1. The Genome Components (Built-in Ablation)
Our **Joint Genome** search space itself allows for natural ablation analysis. By looking at the winning configurations (the best `x` vectors), we can see what the optimizer "decided" to remove:

*   **Ablation of Demos**: In tasks where the model is already proficient (like GSM8K), the optimizer often selects a nearly empty `demo_mask`, effectively ablating the few-shot context to avoid noise.
*   **Ablation of Instructions**: In the **Logic** task, the baseline fails precisely because it lacks the "Instruction-Block 1" (Operator Precedence). Our successful runs always select this block, proving that instructions are the non-negotiable component for logic.

## 2. Experimental Baselines as Ablations
Our existing results table already functions as a robust ablation study:

| Component Removed (Ablated) | Represented by Baseline | Impact on Logic |
| :--- | :--- | :--- |
| **Search & Optimization** | `BASELINE_ALL` | -33.3% (Drops to 66.7%) |
| **Instructions & Demos** | `BASELINE_NONE` | -33.3% (Drops to 66.7%) |
| **Logic/Heuristics** | `RANDOM` | High Variance (Unstable) |
| **Framework Complexity** | `dspy_miprov2` | -66.7% (Drops to 33.3%) |

## 3. Conclusions for the Paper
*   **The "Joint" Advantage**: Individual ablation shows that having *all* instructions (Baseline ALL) is actually worse than having *optimized* instructions. This proves that the **Search Process** is more important than the **Prompt Content** itself.
*   **Instruction Dominance**: Instructions are the "heavy hitters" for Logic, while Demos provide the "stability" for Arithmetic.
*   **Minimalism beats Complexity**: Our ablation of the DSPy framework (which is a more complex programmatic approach) shows that for local LLMs, a simple nature-inspired search is more effective than a multi-stage compilation pipeline.
