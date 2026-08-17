.PHONY: test plots demo dashboard experiment-mock portfolio site data

test:
	python -m pytest

data:
	python -m src data

portfolio:
	python -m src experiment --portfolio

plots:
	python -m src plots

site:
	python -m src site

demo:
	python -m src.demo

dashboard:
	python -m streamlit run app.py

experiment-mock:
	python -m src experiment --fast --backend mock

experiment-ollama:
	python -m src experiment --balanced --backend ollama
