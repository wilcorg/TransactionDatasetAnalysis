#!/usr/bin/env sh
set -e

cd "$(dirname "$0")"

tmpfile=$(mktemp)
trap 'rm -f "$tmpfile"' EXIT

git diff --cached --name-only --diff-filter=ACMR --relative -z -- '*.py' '*.pyi' '*.ipynb' \
  | xargs -0 -I{} sh -c '[ -f "$1" ] && printf "%s\0" "$1"; :' _ {} > "$tmpfile"

if [ -s "$tmpfile" ]; then
  xargs -0 uv run ruff check --fix < "$tmpfile"
  xargs -0 uv run ruff format < "$tmpfile"
  xargs -0 git add -- < "$tmpfile"
fi
