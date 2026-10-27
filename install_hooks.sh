#!/usr/bin/env bash
# Install/Update local git hooks: commit-msg & pre-commit
# Copyright©2023-2025 Shanghai Biren Technology Co., Ltd. All rights reserved.

set -Eeuo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: not inside a Git work tree."
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
SRC_DIR="$REPO_ROOT/git-hooks"

HOOKS_DIR="$(git config --path core.hooksPath || true)"
if [[ -z "${HOOKS_DIR}" ]]; then
  HOOKS_DIR="$(git rev-parse --path-format=absolute --git-path hooks)"
fi
mkdir -p "${HOOKS_DIR}"

install_one() {
  local name="$1"
  local src="${SRC_DIR}/${name}"
  local dst="${HOOKS_DIR}/${name}"

  if [[ ! -f "${src}" ]]; then
    echo "WARN: ${src} not found, skip."
    return 0
  fi

  if [[ ! -f "${dst}" ]]; then
    cp -f "${src}" "${dst}"
    chmod +x "${dst}"
    echo "Installed: ${name} -> ${dst}"
    return 0
  fi

  if cmp -s "${src}" "${dst}"; then
    chmod +x "${dst}" || true
    echo "Up-to-date: ${name}"
  else
    cp -f "${src}" "${dst}"
    chmod +x "${dst}"
    echo "Updated: ${name} -> ${dst}"
  fi
}

echo "Using hooks dir: ${HOOKS_DIR}"
install_one "commit-msg"
install_one "pre-commit"

echo "Done."
