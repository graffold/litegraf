#!/usr/bin/env bash
# Run the litegraf benchmark suite.
# Usage: ./bench.sh --docs 10
cd "$(dirname "$0")"
PYTHONPATH=src exec python -m pipeline.benchmarks.compare_providers "$@"
