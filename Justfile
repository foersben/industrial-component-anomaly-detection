default:
    @just --list

setup:
    pixi install
    pixi run -e dev pre-commit install
    @just install-extensions

install:
    @just setup

test:
    pixi run -e dev pytest

lint:
    pixi run -e dev ruff check --fix .
    pixi run -e dev ruff format .
    pixi run -e dev mypy app/

check:
    pixi run -e dev pre-commit run --all-files

format:
    pixi run -e dev ruff format .

run:
    pixi run fastapi run app/main.py

clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    rm -rf .cache site build dist .pytest_cache .mypy_cache .ruff_cache .hypothesis .coverage htmlcov .pixi
    @just clean-notebooks

docs:
    pixi run -e dev zensical build

serve:
    pixi run -e dev zensical build
    pixi run -e dev zensical serve -a localhost:9000

lab:
    pixi run -e dev jupyter lab --ip=127.0.0.1 --port=8888

clean-notebooks:
    find notebooks/ -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

act-ci:
    act -W .github/workflows/ci.yml

install-extensions:
    @jq -r '.recommendations[]' .vscode/extensions.json | while read -r ext; do \
        code --install-extension "$ext"; \
    done
