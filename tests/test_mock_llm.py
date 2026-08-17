from src.model import MockLLM, prompt_quality, create_llm


def test_mock_boolean_solves_simple_expression():
    llm = MockLLM()
    out = llm.generate(
        "Question: Evaluate: True AND False.\nAnswer (yes or no): ",
        system="Return yes if the expression evaluates to True, otherwise return no. Output exactly one token: yes or no.",
    )
    text = out.lower()
    assert text
    assert any(token in text for token in ("yes", "no", "false", "true"))


def test_mock_is_deterministic():
    llm = MockLLM(oracle={"Evaluate: True OR False.": "yes"})
    prompt = "Question: Evaluate: True OR False.\nAnswer (yes or no): "
    system = "Operator precedence is: NOT > AND > OR. Output exactly one token: yes or no."
    a = llm.generate(prompt, system=system)
    b = llm.generate(prompt, system=system)
    assert a == b


def test_prompt_quality_rewards_instructions():
    weak = prompt_quality("", "Question: q\nAnswer (yes or no): ")
    strong = prompt_quality(
        "Operator precedence is: NOT > AND > OR. Output exactly one token: yes or no.",
        "Question: demo\nAnswer: yes\n\nQuestion: q\nAnswer (yes or no): ",
    )
    assert strong > weak


def test_create_llm_mock_backend():
    llm = create_llm("mock")
    assert llm.cfg.model_name == "mock"
