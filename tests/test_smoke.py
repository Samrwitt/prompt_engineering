from src.config import DATASET_CATALOG, load_run_config
from src.experiment import EvalConfig, run_budgeted_metaheuristic
from src.methods import build_methods
from src.model import create_llm
from src.prompt import load_blocks, load_jsonl, split_train_test


def test_sa_mock_experiment_completes():
    spec = DATASET_CATALOG["logic"]
    cfg = load_run_config("fast")
    cfg.max_data = 8
    cfg.max_llm_calls = 40
    cfg.sa_iters = 8
    blocks = load_blocks(spec["blocks"])
    data = load_jsonl(spec["path"])[: cfg.max_data]
    train, test = split_train_test(data, seed=42)
    llm = create_llm("mock", oracle={str(r["q"]): str(r["a"]) for r in train + test})
    methods = {name: (fn, kw) for name, fn, kw in build_methods(len(blocks), cfg)}
    fn, kw = methods["SA+"]
    report = run_budgeted_metaheuristic(
        method="SA+",
        algo_fn=fn,
        blocks=blocks,
        demo_candidates=train,
        answer_type=spec["answer_type"],
        llm=llm,
        train_data=train,
        test_data=test,
        seed=0,
        eval_cfg=EvalConfig(max_llm_calls=40, fast_k=4, max_demos=2),
        algo_kwargs=kw,
        backend="mock",
    )
    assert 0.0 <= report.test_acc <= 1.0
    assert report.budget.llm_calls <= 40
    assert report.best_x
