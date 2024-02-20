#!/bin/sh
src="fed"

# Exit immediately if a pipeline exits with a non-zero status.
set -e

# no buffering of log messages, all goes straight to stdout
export PYTHONUNBUFFERED=1

echo "========================= ruff format =========================="
ruff format $src --config pyproject.toml --preview
echo "========================= ruff linting =========================="
ruff check $src --fix --show-fixes --config pyproject.toml --preview
echo "syntax ok"
