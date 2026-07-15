install:
	UV_CACHE_DIR=~/.goinfre/.cache uv sync
	UV_CACHE_DIR=~/.goinfre/.cache uv pip install -e .
	UV_CACHE_DIR=~/.goinfre/.cache uv run python -m ipykernel install --user --name=ML_venv --display-name="ML-from-scratch"

clean:
	rm -rf .venv/
	rm -rf $$(find . -name "__pycache__" -o -name ".mypy_cache" -o -name "*.egg-info")
	rm -rf uv.lock

.PHONY: install clean
