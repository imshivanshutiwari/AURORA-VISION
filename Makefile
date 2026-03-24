.PHONY: install fetch run api evaluate export optimize test lint notebook clean

install:
	pip install -r requirements.txt

fetch:
	python data/processors/metadata_parser.py --fetch

run:
	python dashboard/app.py

api:
	uvicorn api.server:app --reload --port 8000

evaluate:
	python evaluation/benchmark_runner.py

export:
	python deployment/onnx_exporter.py

optimize:
	python deployment/tensorrt_optimizer.py

test:
	pytest tests/ -v --color=yes

lint:
	black . --line-length 100 && flake8 . --max-line-length=100

notebook:
	jupyter notebook notebooks/

clean:
	find . -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	find . -name "*.pyo" -delete 2>/dev/null; \
	true
