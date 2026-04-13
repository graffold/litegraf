#!/usr/bin/env bash
# Publish latest benchmark results to docs/ for GitHub Pages.
# Usage: ./publish_dashboard.sh
cd "$(dirname "$0")"
RESULTS_DIR="src/pipeline/benchmarks/results"
DOCS_DIR="docs"

# Find latest result JSON
LATEST=$(ls -t "$RESULTS_DIR"/*.json 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
  echo "No results found in $RESULTS_DIR"
  exit 1
fi

echo "Publishing: $LATEST"
echo "var BENCHMARK_DATA = $(cat "$LATEST");" > "$DOCS_DIR/data.js"

# Inject data script into index.html if not already present
if ! grep -q 'data.js' "$DOCS_DIR/index.html"; then
  sed -i '' 's|</head>|<script src="data.js"></script></head>|' "$DOCS_DIR/index.html"
fi

echo "Dashboard ready: open $DOCS_DIR/index.html or push to GitHub Pages"
