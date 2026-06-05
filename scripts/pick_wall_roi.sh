#!/usr/bin/env bash
# Pick a wall ROI from a synced overhead video (needs display / ssh -X).
#
# Usage:
#   export SESSION_ROOT=/home/scholl-lab/ferret_recordings/session_...
#   ./scripts/pick_wall_roi.sh --camera 24676894 --wall-id north
#   ./scripts/pick_wall_roi.sh --video /path/to/24676894_....mp4 --wall-id east

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARGS=()
while [[ $# -gt 0 ]]; do
	ARGS+=("$1")
	shift
done

if [[ -n "${SESSION_ROOT:-}" ]]; then
	ARGS=(--session "$SESSION_ROOT" "${ARGS[@]}")
fi

uv run python -m src.arena_render.pick_wall_roi "${ARGS[@]}"
