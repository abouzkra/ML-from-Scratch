install:
	uv sync
	uv pip install -e .
	uv run python -m ipykernel install --user --name=ML_venv --display-name="ML-from-scratch"

clean:
	rm -rf .venv/
	rm -rf $$(find . -name "__pycache__" -o -name ".mypy_cache" -o -name "*.egg-info")
	rm -rf uv.lock

.PHONY: install clean
