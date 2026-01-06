from src.model import HFLLM

llm = HFLLM()

prompts = [
    "Translate english to german: How old are you?",
    "Please answer the following question. What is the capital of France?",
    "What is 10 + 2?",
    "Answer the following math question: 7 + 5",
]

for p in prompts:
    print(f"Input: {p}")
    print(f"Output: {llm.generate(p)}")
    print("-" * 20)
