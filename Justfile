# List all available recipes in the Justfile (Default task)
default:
	@just --list

# Bootstrap the environment (install Pixi dependencies, pre-commit hooks, and VS Code extensions)
setup:
	pixi install -e dev
	pixi run --frozen -e dev pre-commit install
	@just install-extensions

# Alias target to run bootstrap setup
install:
	@just setup

init:
	pixi shell --manifest-path ./pyproject.toml -e dev

# Run the pytest test suite in the dev environment
test:
	pixi run --frozen -e dev pytest

# Run linting checks (Ruff, Mypy) and auto-fix simple styling discrepancies
lint:
	pixi run --frozen -e dev ruff check --fix .
	pixi run --frozen -e dev ruff format .
	pixi run --frozen -e dev mypy app/

# Run all pre-commit hooks manually against all files in the repository
check:
	pixi run --frozen -e dev pre-commit run --all-files

# Format all code blocks inside the workspace using Ruff formatter
format:
	pixi run --frozen -e dev ruff format .

# Kill any running FastAPI/Streamlit instances (frees ports 8000 and 8501)
stop:
    @echo "Stopping FastAPI and Streamlit..."
    @fuser -k 8000/tcp 2>/dev/null || true
    @fuser -k 8501/tcp 2>/dev/null || true
    @echo "Done."

# Start FastAPI (uvicorn) and Streamlit frontend concurrently
run:
    #!/usr/bin/env bash
    set -euo pipefail

    # Release ports if previous run was killed uncleanly
    fuser -k 8000/tcp 2>/dev/null || true
    fuser -k 8501/tcp 2>/dev/null || true
    sleep 0.3

    echo "Starting FastAPI and Streamlit..."
    pixi run --frozen -e dev api &
    API_PID=$!
    pixi run --frozen -e dev ui &
    UI_PID=$!

    # Propagate SIGINT/SIGTERM to both child processes and wait cleanly
    trap 'echo; echo "Shutting down gracefully..."; kill "$API_PID" "$UI_PID" 2>/dev/null; wait "$API_PID" "$UI_PID" 2>/dev/null; exit 0' INT TERM

    wait

# Clean all temporary files, cache folders, compilation files, and local environments
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .cache site build dist .pytest_cache .mypy_cache .ruff_cache .hypothesis .coverage htmlcov .pixi
	@just clean-notebooks

# Build the documentation site statically using Zensical
docs:
	pixi run --frozen -e dev zensical build

# Build and serve the Zensical documentation website locally on http://localhost:9000
serve:
	pixi run --frozen -e dev zensical build
	pixi run --frozen -e dev zensical serve -a localhost:9000

# Start a local JupyterLab server inside the dev environment
lab:
	pixi run --frozen -e dev jupyter lab --ip=127.0.0.1 --port=8888
# Download the raw MVTec AD assets from the remote Hugging Face Hub repository
download-data:
	pixi run --frozen -e dev python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='foersben/mvtec-ad', repo_type='dataset', local_dir='data/raw/mvtec_ad')"

# Pull the MVTec AD dataset and benchmark files into the local data directory
fetch-data: download-data
	@echo "Downloading pre-computed AUPIMO curves..."
	mkdir -p data/external
	git clone --depth 1 https://github.com/jpcbertoldo/aupimo.git data/external/aupimo_repo
	mkdir -p data/external/aupimo_benchmarks
	mv data/external/aupimo_repo/data/experiments/benchmark/* data/external/aupimo_benchmarks/
	rm -rf data/external/aupimo_repo

# Run the dummy classifier evaluation to demonstrate the accuracy paradox (supports theoretical or real mode)
# Example: just run-dummy mode=real data_root=data/raw/mvtec_ad category=bottle
run-dummy *args:
	pixi run --frozen -e dev python -m app.main dummy {{args}}

# Run the Patchcore baseline on the MVTec AD dataset
# Example: just run-baseline category="bottle"
run-baseline *args:
	pixi run --frozen -e dev python -m app.main baseline {{args}}

# Extract the downloaded MVTec AD tar.xz package locally (if downloaded manually from the official site)
extract-data:
	@echo "Extracting MVTec AD package..."
	tar -xf data/raw/mvtec_ad/mvtec_anomaly_detection.tar.xz -C data/raw/mvtec_ad/

# Log in to Hugging Face Hub (queries KeePassXC Secret Service via D-Bus, falls back to interactive prompt)
hf-login:
	#!/usr/bin/env bash
	token=$(secret-tool lookup Title "[HuggingFace Access Token] [write] workspace-upload" 2>/dev/null)
	if [ -n "$token" ]; then
		echo "Token successfully retrieved from KeePassXC Secret Service."
		pixi run --frozen -e dev hf auth login --token "$token"
	else
		echo "Could not find token in KeePassXC Secret Service. Falling back to interactive prompt..."
		pixi run --frozen -e dev hf auth login
	fi

# Upload a local directory to the Hugging Face Hub dataset repository (defaults to raw data dir)
upload-data local_dir='data/raw/mvtec_ad':
	pixi run --frozen -e dev hf upload foersben/mvtec-ad {{local_dir}} . --repo-type dataset

# Clean Jupyter notebook checkpoint caches under the notebooks directory
clean-notebooks:
	find notebooks/ -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

# Run the CI pipeline locally using GitHub 'act' tool
act-ci:
	act -W .github/workflows/ci.yml

# Install the recommended VS Code extensions list
install-extensions:
	@jq -r '.recommendations[]' .vscode/extensions.json | while read -r ext; do \
		code --install-extension "$ext"; \
	done
