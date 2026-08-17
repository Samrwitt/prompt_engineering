from src.prompt import build_prompt, decode_genome, extract_answer, normalize_gt


def test_extract_yesno_prefers_first_token():
    assert extract_answer("yes", "yesno") == "yes"
    assert extract_answer("The answer is no.", "yesno") == "no"
    assert extract_answer("true", "yesno") == "yes"
    assert extract_answer("false", "yesno") == "no"


def test_extract_number_from_answer_prefix():
    assert extract_answer("Reasoning...\nAnswer: 42", "number") == "42"
    assert extract_answer("I get 3 then 11", "number") == "11"
    assert extract_answer("Answer: -7", "number") == "-7"


def test_normalize_ground_truth():
    assert normalize_gt("True", "yesno") == "yes"
    assert normalize_gt("0", "yesno") == "no"
    assert normalize_gt("Yes", "yesno") == "yes"


def test_build_prompt_joins_selected_blocks():
    blocks = ["A", "B", "C"]
    assert build_prompt(blocks, [1, 0, 1]) == "A\nC"
    assert build_prompt(blocks, [0, 0, 0]) == ""


def test_decode_genome_caps_demos():
    x = [1, 0, 1, 1, 1, 1, 1, 1]
    instr, demos = decode_genome(x, n_blocks=2, n_demos=6, max_demos=3)
    assert instr == [1, 0]
    assert demos == [0, 1, 2]
