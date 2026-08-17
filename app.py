"""Streamlit dashboard for results, discovered prompts, and a live mock optimizer."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prompt Metaheuristics", layout="wide")

from src.config import DATASET_CATALOG, load_run_config
from src.methods import METHOD_HELP, build_methods
from src.model import create_llm
from src.plots import METHOD_ORDER, load_runs
from src.prompt import load_blocks, load_jsonl, split_train_test
from src.experiment import EvalConfig, run_budgeted_metaheuristic


@st.cache_data(show_spinner=False)
def _runs_df() -> pd.DataFrame:
    return load_runs(ROOT / "results")


def _calls(budget) -> float:
    if isinstance(budget, dict):
        return float(budget.get("llm_calls") or 0)
    return 0.0


def _method_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows = []
    for (dataset, method), g in df.groupby(["dataset", "method"]):
        rows.append({
            "dataset": dataset,
            "method": method,
            "n": int(len(g)),
            "mean_test": float(g["test_acc"].mean()),
            "std_test": float(g["test_acc"].std(ddof=0) if len(g) else 0.0),
            "mean_train": float(g["train_acc"].mean()) if "train_acc" in g else float("nan"),
            "mean_calls": float(g["budget"].map(_calls).mean()),
            "mean_seconds": float(g["wall_time_sec"].mean()) if "wall_time_sec" in g else float("nan"),
        })
    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(["dataset", "method"])


def page_overview(df: pd.DataFrame) -> None:
    st.title("Joint Prompt Optimization with Metaheuristics")
    st.markdown(
        "Search over **instruction blocks** and **few-shot demonstrations** as a single binary genome. "
        "Simulated Annealing, Differential Evolution, Grey Wolf Optimizer, and a DE→SA hybrid "
        "are compared under a shared LLM-call budget against greedy, random, and static baselines."
    )
    c1, c2, c3, c4 = st.columns(4)
    n_runs = 0 if df.empty else len(df)
    n_methods = 0 if df.empty else df["method"].nunique()
    n_ds = 0 if df.empty else df["dataset"].nunique()
    best = 0.0 if df.empty else float(df["test_acc"].max())
    c1.metric("Logged runs", n_runs)
    c2.metric("Methods", n_methods)
    c3.metric("Datasets", n_ds)
    c4.metric("Best test acc", f"{best:.0%}")

    st.subheader("Search genome")
    st.code(
        "x = [ instruction bits | demonstration bits ]  ∈ {0,1}^{B+E}\n"
        "system prompt  ← join(blocks[i] for i if x[i]=1)\n"
        "few-shot prefix ← up to K examples selected by the demo bits",
        language="text",
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**What this project shows**")
        st.markdown(
            "- Combinatorial prompt search, not free-text rewriting\n"
            "- Fair budgets, caching, and temperature-0 inference\n"
            "- Offline `mock` backend so the pipeline runs without a GPU\n"
            "- Wilcoxon tests and inspectable discovered prompts"
        )
    with right:
        st.markdown("**How to reproduce**")
        st.code(
            "python -m src experiment --fast --backend mock\n"
            "python -m src plots\n"
            "python -m src stats",
            language="bash",
        )


def page_results(df: pd.DataFrame) -> None:
    st.title("Experimental results")
    if df.empty:
        st.warning("No `results/runs.jsonl` yet. Run `python -m src experiment --fast`.")
        return

    datasets = sorted(df["dataset"].dropna().unique())
    dataset = st.selectbox("Dataset", datasets, index=0)
    sub = df[df["dataset"] == dataset]
    summary = _method_summary(sub)

    chart_df = summary.set_index("method")[["mean_test"]]
    st.bar_chart(chart_df)

    st.dataframe(
        summary.assign(
            mean_test=lambda d: d["mean_test"].map(lambda x: f"{x:.3f}"),
            std_test=lambda d: d["std_test"].map(lambda x: f"{x:.3f}"),
            mean_train=lambda d: d["mean_train"].map(lambda x: f"{x:.3f}"),
            mean_calls=lambda d: d["mean_calls"].map(lambda x: f"{x:.0f}"),
            mean_seconds=lambda d: d["mean_seconds"].map(lambda x: f"{x:.1f}"),
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Per-seed test accuracy")
    pivot = sub.pivot_table(index="seed", columns="method", values="test_acc")
    st.dataframe(pivot.style.format("{:.3f}"), use_container_width=True)

    if "curve" in sub.columns:
        st.subheader("Convergence (best accuracy vs calls)")
        import pandas as pd

        points = []
        for _, row in sub.iterrows():
            curve = row.get("curve")
            if not isinstance(curve, list):
                continue
            for item in curve:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    points.append({"method": row["method"], "calls": item[0], "acc": item[1], "seed": row["seed"]})
        if points:
            cdf = pd.DataFrame(points)
            mean_curve = cdf.groupby(["method", "calls"], as_index=False)["acc"].mean()
            st.line_chart(mean_curve.pivot_table(index="calls", columns="method", values="acc"))


def page_prompts(df: pd.DataFrame) -> None:
    st.title("Best discovered prompts")
    if df.empty:
        st.warning("No runs logged.")
        return
    dataset = st.selectbox("Dataset", sorted(df["dataset"].unique()))
    sub = df[df["dataset"] == dataset].sort_values("test_acc", ascending=False)
    method = st.selectbox("Method", ["(best overall)"] + sorted(sub["method"].unique()))
    view = sub if method == "(best overall)" else sub[sub["method"] == method]
    row = view.iloc[0]
    st.metric("Test accuracy", f"{row['test_acc']:.1%}")
    st.caption(f"{row['method']} · seed {row['seed']} · calls {_calls(row.get('budget')):.0f}")
    st.markdown("**Instruction blocks**")
    st.code(row.get("best_instruction_text") or "[none]", language="text")
    demos = row.get("selected_demo_indices") or []
    st.markdown(f"**Few-shot demo indices:** `{list(demos)}`")
    spec = DATASET_CATALOG.get(dataset)
    if spec and demos:
        path = ROOT / spec["path"]
        if path.exists():
            data = load_jsonl(path)
            train, _ = split_train_test(data, seed=42)
            for i in demos:
                if i < len(train):
                    st.markdown(f"- **Q:** {train[i]['q']}  \n  **A:** `{train[i]['a']}`")


def page_playground() -> None:
    st.title("Live optimizer playground")
    st.caption("Runs on the deterministic mock LLM — no Ollama required.")
    methods = list(METHOD_HELP)
    col_a, col_b, col_c = st.columns(3)
    method = col_a.selectbox("Method", methods, index=methods.index("SA+"))
    dataset = col_b.selectbox("Dataset", ["logic", "arithmetic"])
    seed = col_c.number_input("Seed", min_value=0, max_value=99, value=0)
    st.info(METHOD_HELP[method])

    if st.button("Run search", type="primary"):
        spec = DATASET_CATALOG[dataset]
        cfg = load_run_config("fast")
        cfg.max_data = 12
        cfg.max_llm_calls = 80
        cfg.sa_iters = 12
        cfg.de_iters = 6
        cfg.gwo_iters = 6
        cfg.hybrid_de = 4
        cfg.hybrid_sa = 8
        cfg.pop_size = 6
        blocks = load_blocks(ROOT / spec["blocks"])
        data = load_jsonl(ROOT / spec["path"])[: cfg.max_data]
        train, test = split_train_test(data, seed=42)
        oracle = {str(r["q"]): str(r["a"]) for r in train + test}
        llm = create_llm("mock", oracle=oracle)
        method_map = {name: (fn, kw) for name, fn, kw in build_methods(len(blocks), cfg)}
        algo_fn, kwargs = method_map[method]
        eval_cfg = EvalConfig(max_llm_calls=cfg.max_llm_calls, fast_k=5, early_stop=True, max_demos=3)
        with st.spinner(f"Searching with {method}…"):
            report = run_budgeted_metaheuristic(
                method=method,
                algo_fn=algo_fn,
                blocks=blocks,
                demo_candidates=train,
                answer_type=spec["answer_type"],
                llm=llm,
                train_data=train,
                test_data=test,
                seed=int(seed),
                eval_cfg=eval_cfg,
                algo_kwargs=kwargs,
                backend="mock",
            )
        m1, m2, m3 = st.columns(3)
        m1.metric("Train accuracy", f"{report.train_acc:.1%}")
        m2.metric("Test accuracy", f"{report.test_acc:.1%}")
        m3.metric("Mock LLM calls", report.budget.llm_calls)
        st.markdown("**Discovered instruction prompt**")
        st.code(report.best_instruction_text or "[empty system prompt]", language="text")
        st.markdown(f"Selected demo indices: `{report.selected_demo_indices}`")
        if report.curve:
            cdf = pd.DataFrame(report.curve, columns=["calls", "best_acc"])
            st.line_chart(cdf.set_index("calls"))


def main() -> None:
    df = _runs_df()
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Results", "Prompts", "Playground"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Mock backend for demos. Use `--backend ollama` for Llama 3.2 experiments.")
    if page == "Overview":
        page_overview(df)
    elif page == "Results":
        page_results(df)
    elif page == "Prompts":
        page_prompts(df)
    else:
        page_playground()


if __name__ == "__main__":
    main()
