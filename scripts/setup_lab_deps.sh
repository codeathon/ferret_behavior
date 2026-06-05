#!/usr/bin/env bash
# Bootstrap ferret_behavior and external subprocess repos on the lab machine.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPS_ROOT="${DEPS_ROOT:-$HOME/git_repos}"

echo "==> Syncing ferret_behavior environment"
cd "$REPO_ROOT"
uv sync

echo "==> Cloning external tools under $DEPS_ROOT"
mkdir -p "$DEPS_ROOT"

clone_if_missing() {
	local url="$1"
	local dest="$2"
	local branch="${3:-}"
	if [[ -d "$dest/.git" ]]; then
		echo "  exists: $dest"
		return
	fi
	if [[ -n "$branch" ]]; then
		git clone --branch "$branch" "$url" "$dest"
	else
		git clone "$url" "$dest"
	fi
}

clone_if_missing "https://github.com/freemocap/skellyclicker.git" "$DEPS_ROOT/skellyclicker"
clone_if_missing "https://github.com/philipqueen/freemocap_playground.git" "$DEPS_ROOT/dlc_to_3d" "philip/bs"
clone_if_missing "https://github.com/freemocap/freemocap.git" "$DEPS_ROOT/freemocap"

echo "==> Verify imports"
uv run python -c "import cv2, torch, pandas, rerun; print('ferret_behavior deps OK')"

cat <<EOF

Next steps (manual on lab machine):
1. Install skellyclicker env: conda create -n skellyclicker python=3.11 && ...
2. Install dlc_to_3d venv in $DEPS_ROOT/dlc_to_3d
3. Install freemocap env (fmc) for headless calibration
4. Update paths in configs/offline_pipeline.json if your clone locations differ
5. export SESSION_ROOT=/home/scholl-lab/ferret_recordings/session_...
6. ./scripts/lab_offline_pov.sh validate

EOF
