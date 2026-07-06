# List all available recipes in the Justfile (Default task)
default:
    @just --list

# Bootstrap the environment (install Pixi dependencies, pre-commit hooks, .venv symlink, and VS Code extensions)
setup:
    pixi install -e dev
    pixi run -e dev pre-commit install
    rm -rf .venv && ln -s .pixi/envs/dev .venv
    @just install-extensions

# Delete the local .venv folder and recreate it as a symbolic link pointing to .pixi/envs/dev
link-venv:
    rm -rf .venv && ln -s .pixi/envs/dev .venv

# Alias target to run bootstrap setup
install:
    @just setup

# Run the pytest test suite in the dev environment
test:
    pixi run -e dev pytest

# Run linting checks (Ruff, Mypy) and auto-fix simple styling discrepancies
lint:
    pixi run -e dev ruff check --fix .
    pixi run -e dev ruff format .
    pixi run -e dev mypy app/

# Run all pre-commit hooks manually against all files in the repository
check:
    pixi run -e dev pre-commit run --all-files

# Format all code blocks inside the workspace using Ruff formatter
format:
    pixi run -e dev ruff format .

# Start the production FastAPI web server
run:
    pixi run fastapi run app/main.py

# Clean all temporary files, cache folders, compilation files, and local environments
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    rm -rf .cache site build dist .pytest_cache .mypy_cache .ruff_cache .hypothesis .coverage htmlcov .pixi
    @just clean-notebooks

# Build the documentation site statically using Zensical
docs:
    pixi run -e dev zensical build

# Build and serve the Zensical documentation website locally on http://localhost:9000
serve:
    pixi run -e dev zensical build
    pixi run -e dev zensical serve -a localhost:9000

# Start a local JupyterLab server inside the dev environment
lab:
    pixi run -e dev jupyter lab --ip=127.0.0.1 --port=8888
# Download the raw MVTec ITODD assets from the remote Hugging Face Hub repository
download-data:
    pixi run -e dev python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='foersben/mvtec-itodd', repo_type='dataset', local_dir='data/raw/mvtec_itodd')"

# Extract the downloaded MVTec ITODD tar.xz packages locally
extract-data:
    @echo "Extracting base package..."
    tar -xf data/raw/mvtec_itodd/base_package.tar.xz -C data/raw/mvtec_itodd/
    @echo "Extracting 3D short baseline data..."
    tar -xf data/raw/mvtec_itodd/3d_short_baseline.tar.xz -C data/raw/mvtec_itodd/

# Log in to Hugging Face Hub (queries KeePassXC Secret Service via D-Bus, falls back to interactive prompt)
hf-login:
    #!/usr/bin/env bash
    token=$(secret-tool lookup Title "[HuggingFace Access Token] [write] workspace-upload" 2>/dev/null)
    if [ -n "$token" ]; then
        echo "Token successfully retrieved from KeePassXC Secret Service."
        pixi run -e dev hf auth login --token "$token"
    else
        echo "Could not find token in KeePassXC Secret Service. Falling back to interactive prompt..."
        pixi run -e dev hf auth login
    fi

# Upload a local directory to the Hugging Face Hub dataset repository (defaults to raw data dir)
upload-data local_dir='data/raw/mvtec_itodd':
    pixi run -e dev hf upload foersben/mvtec-itodd {{local_dir}} . --repo-type dataset

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
