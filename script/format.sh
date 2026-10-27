#!/usr/bin/env bash
# Format Python (isort + black) and C/C++ (clang-format) files.
# Copyright©2023-2025 Shanghai Biren Technology Co., Ltd. All rights reserved.

set -Eeuo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

SKIP_FILES_REL=(
)
SKIP_GLOBS=(
  # eg:'vendor/**' 'generated/**'
)

[[ "${DEBUG:-0}" == "1" ]] && set -x

install_format_tools() {
  # --- clang-format ---
  if ! command -v clang-format &>/dev/null; then
    echo "Installing clang-format..."
    apt-get update -y
    apt-get install -y clang-format || {
      echo "Failed to install clang-format."
      exit 1
    }
  else
    echo "clang-format is already installed: $(clang-format --version | head -n1)"
  fi

  # isort
  if ! python3 -m pip show isort &>/dev/null; then
    echo "Installing isort..."
    python3 -m pip install isort || {
      echo "Failed to install isort."
      exit 1
    }
  else
    echo "isort is already installed"
  fi

  # black
  if ! python3 -m pip show black &>/dev/null; then
    echo "Installing black..."
    python3 -m pip install black || {
      echo "Failed to install black."
      exit 1
    }
  else
    black_ver=$(python3 -m pip show black | awk -F': ' '/Version/{print $2}')
    black_yy="${black_ver%%.*}"
    curr_yy="$(date +%y)"
    if [[ "$black_yy" != "$curr_yy" ]]; then
      echo "Updating black (version mismatch)..."
      python3 -m pip install --upgrade black || {
        echo "Failed to update black."
        exit 1
      }
    else
      echo "black is up to date: $black_ver"
    fi
  fi

  # flake8
  if ! python3 -m pip show flake8 &>/dev/null; then
    echo "Installing flake8..."
    python3 -m pip install flake8 || {
      echo "Failed to install flake8."
      exit 1
    }
  else
    echo "flake8 is already installed"
  fi

  echo "Format & lint tools installation done."
}

is_supported_ext() {
  case "$1" in
    *.py|*.pyi|*.pyx|*.pxd|*.cpp|*.hpp|*.h) return 0 ;;
    *) return 1 ;;
  esac
}

is_in_skip_list() {
  local rel="$1"
  for s in "${SKIP_FILES_REL[@]}"; do
    [[ "$rel" == "$s" ]] && return 0
  done
  for g in "${SKIP_GLOBS[@]}"; do
    [[ "$rel" == $g ]] && return 0
  done
  return 1
}

build_isort_skip_args() {
  ISORT_SKIP_ARGS=()
  for s in "${SKIP_FILES_REL[@]}"; do
    ISORT_SKIP_ARGS+=( "--skip" "$REPO_ROOT/$s" )
  done
  for g in "${SKIP_GLOBS[@]}"; do
    ISORT_SKIP_ARGS+=( "--skip-glob" "$g" )
  done
}

build_black_force_exclude_regex() {
  local parts=()
  for s in "${SKIP_FILES_REL[@]}"; do
    local esc="${s//\./\\.}"; esc="${esc//\//\/}"
    parts+=( "$esc" )
  done
  if ((${#parts[@]})); then
    BLACK_FORCE_EXCLUDE="(^|/)($(IFS='|'; echo "${parts[*]}"))$"
  else
    BLACK_FORCE_EXCLUDE=""
  fi
}

to_rel() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "${p#"$REPO_ROOT"/}"
  else
    printf '%s\n' "$p"
  fi
}

gather_from_dir() {
  local dir="$1"
  local have=0
  while IFS= read -r -d '' p; do
    have=1
    local rel; rel="$(to_rel "$p")"
    is_supported_ext "$rel" && CANDIDATES["$rel"]=1
  done < <(git ls-files -z -- "$dir")
  if [[ $have -eq 0 ]]; then
    while IFS= read -r -d '' p; do
      local rel; rel="$(to_rel "$p")"
      is_supported_ext "$rel" && CANDIDATES["$rel"]=1
    done < <(find "$dir" -type f \( \
                -name '*.py'  -o -name '*.pyi' -o -name '*.pyx' -o -name '*.pxd' -o \
                -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \
              \) -print0)
  fi
}

format_file() {
  local file="$1"
  if is_in_skip_list "$file"; then
    echo "  - (skip) $file"
    return 0
  fi

  case "$file" in
    *.py|*.pyi|*.pyx|*.pxd)
      isort --profile black -l 120 "$file" --atomic "${ISORT_SKIP_ARGS[@]}"   # :contentReference[oaicite:4]{index=4}
      if [[ -n "$BLACK_FORCE_EXCLUDE" ]]; then
        black -S -l 120 "$file" --force-exclude "$BLACK_FORCE_EXCLUDE"        # :contentReference[oaicite:5]{index=5}
      else
        black -S -l 120 "$file"
      fi
      ;;
    *.cpp|*.hpp|*.h)
      clang-format -style=file -i "$file"
      ;;
  esac
}

install_format_tools
build_isort_skip_args
build_black_force_exclude_regex

declare -A CANDIDATES=()
declare -a FILES=()

if [[ $# -gt 0 ]]; then
  for arg in "$@"; do
    if [[ -d "$arg" ]]; then
      gather_from_dir "$arg"
    elif [[ -f "$arg" ]]; then
      CANDIDATES["$(to_rel "$arg")"]=1
    else
      echo "WARN: not found: $arg" >&2
    fi
  done
elif [[ "${ALL:-0}" == "1" ]]; then
  while IFS= read -r f; do FILES+=("$f"); done < <(git ls-files -m)
else
  while IFS= read -r f; do FILES+=("$f"); done < <(git diff --cached --name-only --diff-filter=ACM)
fi

if ((${#CANDIDATES[@]})); then
  for k in "${!CANDIDATES[@]}"; do FILES+=("$k"); done
fi

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No files to format."
  exit 0
fi

echo "Formatting files:"
for f in "${FILES[@]}"; do
  is_supported_ext "$f" || continue
  echo "  - $f"
  format_file "$f"
  if [[ "${AUTO_STAGE:-0}" == "1" ]]; then
    git add -- "$f"
  fi
done

echo "Formatting done."
[[ "${AUTO_STAGE:-0}" != "1" ]] && echo "Tip: run 'git add -u' to restage formatted changes."
