#!/usr/bin/env bash
# Arena-only virtual render pipeline (no ferret POV, no DLC).
#
# Usage:
#   export SESSION_ROOT=/home/scholl-lab/ferret_recordings/session_2026-03-19_psychopy_trial_1_ferret411
#   ./scripts/lab_arena_render.sh validate
#   ./scripts/lab_arena_render.sh sync
#   ./scripts/lab_arena_render.sh rerun
#   ./scripts/lab_arena_render.sh extract-textures --max-frames 30
#   ./scripts/lab_arena_render.sh export-unreal-bundle

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSION_ROOT="${SESSION_ROOT:-}"
CONFIG="${CONFIG:-configs/arena_render.json}"
GEOMETRY="${GEOMETRY:-}"
MAX_FRAMES="${MAX_FRAMES:-}"
FRAME_STRIDE="${FRAME_STRIDE:-}"

if [[ -z "$SESSION_ROOT" ]]; then
	echo "Set SESSION_ROOT to your session folder."
	exit 1
fi

CMD="${1:-validate}"
shift || true

ARGS=(--config "$CONFIG" --session "$SESSION_ROOT")
if [[ -n "$GEOMETRY" ]]; then
	ARGS+=(--geometry "$GEOMETRY")
fi
if [[ -n "$MAX_FRAMES" ]]; then
	ARGS+=(--max-frames "$MAX_FRAMES")
fi
if [[ -n "$FRAME_STRIDE" ]]; then
	ARGS+=(--frame-stride "$FRAME_STRIDE")
fi

while [[ $# -gt 0 ]]; do
	case "$1" in
		--geometry) GEOMETRY="$2"; ARGS+=(--geometry "$GEOMETRY"); shift 2 ;;
		--max-frames) MAX_FRAMES="$2"; ARGS+=(--max-frames "$MAX_FRAMES"); shift 2 ;;
		--frame-stride) FRAME_STRIDE="$2"; ARGS+=(--frame-stride "$FRAME_STRIDE"); shift 2 ;;
		--preview-frame) ARGS+=(--preview-frame "$2"); shift 2 ;;
		*) echo "Unknown arg: $1"; exit 1 ;;
	esac
done

case "$CMD" in
	validate|sync|rerun|extract-textures|export-unreal-bundle) ;;
	*) echo "Unknown command: $CMD"; exit 1 ;;
esac

uv run python -m src.arena_render.run_arena_render "${ARGS[@]}" "$CMD"
