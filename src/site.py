"""Build a static GitHub Pages demo from logged experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.plots import METHOD_ORDER, load_final_scores, load_runs

ROOT = Path(__file__).resolve().parents[1]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _rows_from_scores(scores: Dict[str, Any], dataset: str) -> List[Dict[str, Any]]:
    payload = scores.get(dataset) or {}
    rows = []
    for method in METHOD_ORDER:
        block = payload.get(method)
        if not isinstance(block, dict):
            continue
        if "mean_test" in block:
            rows.append({
                "method": method,
                "mean_test": float(block["mean_test"]),
                "std_test": float(block.get("std_test") or 0.0),
                "mean_train": float(block.get("mean_train") or 0.0),
                "mean_calls": float(block.get("mean_llm_calls") or 0.0),
                "n": int(len(block.get("runs") or [])),
            })
        elif "test" in block:
            rows.append({
                "method": method,
                "mean_test": float(block["test"]),
                "std_test": 0.0,
                "mean_train": float(block.get("train") or 0.0),
                "mean_calls": 0.0,
                "n": 1,
            })
    return rows


def _best_prompts(df, dataset: str, k: int = 6) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    sub = df[df["dataset"] == dataset].sort_values("test_acc", ascending=False)
    out = []
    seen = set()
    for _, row in sub.iterrows():
        method = row.get("method")
        if method in seen:
            continue
        seen.add(method)
        budget = row.get("budget") if isinstance(row.get("budget"), dict) else {}
        out.append({
            "method": method,
            "seed": int(row.get("seed") or 0),
            "test_acc": float(row.get("test_acc") or 0.0),
            "calls": budget.get("llm_calls"),
            "instruction": row.get("best_instruction_text") or "",
            "demos": list(row.get("selected_demo_indices") or []),
        })
        if len(out) >= k:
            break
    return out


def _table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<p class='muted'>No scores for this split.</p>"
    body = []
    best = max(r["mean_test"] for r in rows)
    for r in rows:
        klass = "best" if r["mean_test"] == best else ""
        body.append(
            "<tr class='{klass}'><td>{m}</td><td>{t}</td><td>±{s}</td>"
            "<td>{tr}</td><td>{c}</td><td>{n}</td></tr>".format(
                klass=klass,
                m=r["method"],
                t=_pct(r["mean_test"]),
                s=_pct(r["std_test"]),
                tr=_pct(r["mean_train"]),
                c=f"{r['mean_calls']:.0f}",
                n=r["n"],
            )
        )
    return (
        "<table><thead><tr><th>Method</th><th>Test</th><th>Std</th>"
        "<th>Train</th><th>Calls</th><th>Seeds</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _prompt_cards(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "<p class='muted'>Run the benchmark to populate discovered prompts.</p>"
    cards = []
    for item in items:
        instr = (item["instruction"] or "[empty system prompt]").replace("<", "&lt;")
        cards.append(
            f"""<article class="card">
              <h3>{item['method']}</h3>
              <p class="muted">test {_pct(item['test_acc'])} · seed {item['seed']} · demos {item['demos']}</p>
              <pre>{instr}</pre>
            </article>"""
        )
    return "<div class='grid'>" + "".join(cards) + "</div>"


def build_site(out_path: Path | None = None) -> Path:
    pilot_scores = load_final_scores(ROOT / "results")
    mock_dir = ROOT / "results" / "benchmark_mock"
    mock_scores = load_final_scores(mock_dir)
    mock_runs = load_runs(mock_dir)
    logic_rows = _rows_from_scores(mock_scores, "logic")
    arith_rows = _rows_from_scores(mock_scores, "arithmetic")
    pilot_rows = _rows_from_scores(pilot_scores, "logic")
    plot_payload = {
        "logic": logic_rows,
        "arithmetic": arith_rows,
        "pilot": pilot_rows,
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Joint Prompt Optimization with Metaheuristics</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0b1220;
      --card: #121a2b;
      --line: #243049;
      --text: #e8eef8;
      --muted: #9aa8c0;
      --accent: #7aa2ff;
      --good: #3dd68c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.5;
    }}
    a {{ color: var(--accent); }}
    header, main, footer {{ max-width: 1080px; margin: 0 auto; padding: 0 1.25rem; }}
    header {{ padding-top: 2.5rem; }}
    h1 {{ font-size: 2rem; margin: 0 0 0.5rem; letter-spacing: -0.03em; }}
    h2 {{ margin-top: 2.5rem; font-size: 1.35rem; }}
    .muted {{ color: var(--muted); }}
    .hero {{ display: grid; gap: 1rem; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin: 1.5rem 0 0.5rem; }}
    .metric, .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.1rem; }}
    .metric b {{ display: block; font-size: 1.35rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 0.9rem; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 12px; overflow: hidden; }}
    th, td {{ padding: 0.65rem 0.8rem; text-align: left; border-bottom: 1px solid var(--line); font-size: 0.95rem; }}
    th {{ color: var(--muted); font-weight: 600; }}
    tr.best td {{ color: var(--good); font-weight: 600; }}
    pre {{ white-space: pre-wrap; background: #0a101b; padding: 0.8rem; border-radius: 8px; font-size: 0.85rem; }}
    .callout {{ border-left: 3px solid var(--accent); padding: 0.2rem 0 0.2rem 1rem; }}
    nav {{ display: flex; gap: 1rem; margin: 1rem 0 0; font-size: 0.95rem; }}
    footer {{ padding: 3rem 1.25rem; color: var(--muted); }}
    @media (max-width: 800px) {{ .metrics {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <p class="muted">Prompt optimization · combinatorial search · budgeted evaluation</p>
    <h1>Joint instruction and few-shot search with metaheuristics</h1>
    <p class="hero muted">
      A prompt is a binary genome: which instruction blocks to keep, and which demonstrations to include.
      Simulated Annealing, Differential Evolution, Grey Wolf Optimizer, and a DE→SA hybrid compete under a
      shared call budget against greedy, random, and static baselines.
    </p>
    <nav>
      <a href="#benchmark">Benchmark</a>
      <a href="#pilot">Llama 3.2 pilot</a>
      <a href="#prompts">Prompts</a>
      <a href="#reproduce">Reproduce</a>
    </nav>
    <div class="metrics">
      <div class="metric"><span class="muted">Holdout size</span><b>50 / task</b></div>
      <div class="metric"><span class="muted">Seeds</span><b>5</b></div>
      <div class="metric"><span class="muted">Call budget</span><b>220</b></div>
      <div class="metric"><span class="muted">Backends</span><b>Mock + Ollama</b></div>
    </div>
  </header>
  <main>
    <section id="benchmark">
      <h2>Reproducible benchmark (MockLLM)</h2>
      <p class="muted">
        100 items per task, 50/50 split, 16-example demo pool, 5 seeds, 220-call budget.
        The mock model’s accuracy depends on which instructions and demos are selected, so the search is real.
        Anyone can rerun this in under a minute with no GPU.
        On both tasks, DE, GWO, and Hybrid DE-SA beat BASELINE_ALL by Wilcoxon signed-rank
        (<em>p</em> = 0.031, n = 5). Greedy search does not.
      </p>
      <h3>Boolean expressions</h3>
      {_table(logic_rows)}
      <div id="chart-logic" style="height:360px;margin-top:1rem;"></div>
      <p><img alt="Boolean convergence" src="figures/convergence_logic_mock.png" style="width:100%;border-radius:12px;border:1px solid var(--line);" /></p>
      <h3>Arithmetic</h3>
      {_table(arith_rows)}
      <div id="chart-arith" style="height:360px;margin-top:1rem;"></div>
    </section>

    <section id="pilot">
      <h2>Local Llama 3.2 pilot</h2>
      <div class="callout">
        <p>
          Real LLM numbers from Llama 3.2 via Ollama. This was a small-n study
          (3 seeds, test set of 3 logic items). Accuracy therefore jumps in thirds.
          It is a <strong>pilot</strong>, not a claim of statistical significance.
        </p>
      </div>
      {_table(pilot_rows)}
      <p><img alt="Pilot accuracy" src="figures/accuracy_logic.png" style="width:100%;border-radius:12px;border:1px solid var(--line);" /></p>
    </section>

    <section id="prompts">
      <h2>Discovered instruction prompts</h2>
      <p class="muted">Best test configuration per method on the mock boolean split.</p>
      {_prompt_cards(_best_prompts(mock_runs, "logic"))}
    </section>

    <section id="reproduce">
      <h2>Reproduce</h2>
      <pre>pip install -r requirements.txt
python -m src experiment --portfolio
python -m src stats --path results/benchmark_mock/runs.jsonl
python -m src site</pre>
      <p class="muted">Live LLM: <code>ollama pull llama3.2 && python -m src experiment --balanced --backend ollama</code></p>
      <p class="muted">Source: <a href="https://github.com/Samrwitt/prompt_engineering">github.com/Samrwitt/prompt_engineering</a></p>
    </section>
  </main>
  <footer>MIT license. Reported mock scores are generated by this repository; the Llama 3.2 table is the original local pilot.</footer>
  <script>
    const DATA = {json.dumps(plot_payload)};
    function bar(div, rows, title) {{
      if (!rows || !rows.length) return;
      Plotly.newPlot(div, [{{
        type: "bar",
        x: rows.map(r => r.method),
        y: rows.map(r => r.mean_test),
        error_y: {{ type: "data", array: rows.map(r => r.std_test), visible: true }},
        marker: {{ color: "#7aa2ff" }}
      }}], {{
        title, paper_bgcolor: "#0b1220", plot_bgcolor: "#0b1220",
        font: {{ color: "#e8eef8" }},
        yaxis: {{ range: [0, 1], title: "Mean test accuracy", gridcolor: "#243049" }},
        xaxis: {{ tickangle: -25 }},
        margin: {{ t: 40, r: 10, l: 50, b: 80 }}
      }}, {{displayModeBar: false, responsive: true}});
    }}
    bar("chart-logic", DATA.logic, "Boolean — mean test accuracy");
    bar("chart-arith", DATA.arithmetic, "Arithmetic — mean test accuracy");
  </script>
</body>
</html>
"""
    dest = out_path or (ROOT / "docs" / "index.html")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(html, encoding="utf-8")
    (dest.parent / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Wrote {dest}")
    return dest


if __name__ == "__main__":
    build_site()
