.PHONY: test plots demo dashboard experiment-mock experiment-ollama

test:
	python -m pytest

plots:
	python -m src plots

demo:
	python -m src.demo

dashboard:
	python -m streamlit run app.py

experiment-mock:
	python -m src experiment --fast --backend mock

experiment-ollama:
	python -m src experiment --balanced --backend ollama
