#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
exec uv run --frozen --no-sync python "$repo_dir/scripts/launch_glm53.py" "$@"
