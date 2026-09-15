#!/usr/bin/env bash
set -e

echo "Running Patchcore Evaluations..."
pixi run python scripts/evaluate.py --model patchcore

echo "Running Tuned Patchcore Evaluations..."
pixi run python scripts/evaluate.py --model patchcore --tuned

echo "Running Keras CAE Evaluations..."
pixi run python scripts/evaluate.py --model keras

echo "Running Tuned Keras CAE Evaluations..."
pixi run python scripts/evaluate.py --model keras --tuned

echo "All evaluations completed!"
