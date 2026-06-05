#!/usr/bin/env bash
# Run the offline ferret POV pipeline on the Scholl lab machine.
#
# Usage:
#   export SESSION_ROOT=/home/scholl-lab/ferret_recordings/session_YYYY-MM-DD_ferret_XXX
#   ./scripts/lab_offline_pov.sh validate
#   ./scripts/lab_offline_pov.sh pipeline
#   ./scripts/lab_offline_pov.sh gaze --clip 0m_37s-1m_37s
#   ./scripts/lab_offline_pov.sh all --clip 0m_37s-1m_37s

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SESSION_ROOT="${SESSION_ROOT:-}"
CLIP_NAME="${CLIP_NAME:-}"
CONFIG="${CONFIG:-configs/offline_pipeline.json}"

if [[ -z "$SESSION_ROOT" ]]; then
	echo "Set SESSION_ROOT to your session folder, e.g.:"
	echo "  export SESSION_ROOT=/home/scholl-lab/ferret_recordings/session_2025-10-22_ferret_420_EO13"
	exit 1
fi

CMD="${1:-validate}"
shift || true

case "$CMD" in
	validate)
		uv run python -m src.offline_pov.run_offline_pov \
			--config "$CONFIG" \
			--session "$SESSION_ROOT" \
			--validate-only
		;;
	pipeline)
		uv run python -m src.offline_pov.run_offline_pov \
			--config "$CONFIG" \
			--session "$SESSION_ROOT" \
			--skip-gaze \
			--skip-copy
		;;
	gaze)
		CLIP="${CLIP_NAME:-}"
		while [[ $# -gt 0 ]]; do
			case "$1" in
				--clip) CLIP="$2"; shift 2 ;;
				*) echo "Unknown arg: $1"; exit 1 ;;
			esac
		done
		if [[ -z "$CLIP" ]]; then
			echo "Provide --clip <name> or set CLIP_NAME"
			exit 1
		fi
		uv run python -m src.offline_pov.run_offline_pov \
			--config "$CONFIG" \
			--session "$SESSION_ROOT" \
			--clip "$CLIP" \
			--skip-pipeline
		;;
	all)
		CLIP="${CLIP_NAME:-}"
		while [[ $# -gt 0 ]]; do
			case "$1" in
				--clip) CLIP="$2"; shift 2 ;;
				*) echo "Unknown arg: $1"; exit 1 ;;
			esac
		done
		ARGS=(--config "$CONFIG" --session "$SESSION_ROOT")
		if [[ -n "$CLIP" ]]; then
			ARGS+=(--clip "$CLIP")
		fi
		uv run python -m src.offline_pov.run_offline_pov "${ARGS[@]}"
		;;
	*)
		echo "Usage: SESSION_ROOT=... $0 {validate|pipeline|gaze|all} [--clip NAME]"
		exit 1
		;;
esac
