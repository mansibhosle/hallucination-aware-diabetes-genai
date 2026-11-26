.PHONY: setup demo api ui test eval
setup:
	python -m pip install -r requirements.txt
demo:
	python scripts/bootstrap_demo.py
api:
	uvicorn src.api.main:app --reload --port 8000
ui:
	streamlit run src/ui/streamlit_app.py
test:
	pytest -q
eval:
	python -m src.evaluation.run_eval --file data/evaluation/sample_eval.jsonl
