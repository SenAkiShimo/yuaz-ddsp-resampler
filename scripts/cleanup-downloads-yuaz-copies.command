#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DOWNLOADS="$HOME/Downloads"
TRASH="$HOME/.Trash"

mode="${1:-}"

if [ ! -d "$DOWNLOADS" ]; then
  echo "Downloads folder not found: $DOWNLOADS" >&2
  exit 1
fi

mkdir -p "$TRASH"

CURRENT_ROOT="$(cd "$ROOT" && pwd -P)"
CANONICAL="$DOWNLOADS/yuaz-ddsp-resampler"

candidates=()
while IFS= read -r path; do
  [ -n "$path" ] || continue
  resolved="$(cd "$path" 2>/dev/null && pwd -P || printf '%s' "$path")"

  if [ "$resolved" = "$CURRENT_ROOT" ]; then
    continue
  fi
  if [ "$path" = "$CANONICAL" ]; then
    continue
  fi

  base="$(basename "$path")"
  case "$base" in
    yuaz-ddsp-resampler*) candidates+=("$path") ;;
  esac
done < <(find "$DOWNLOADS" -mindepth 1 -maxdepth 1 -type d -name 'yuaz-ddsp-resampler*' -print | LC_ALL=C sort)

if [ "${#candidates[@]}" -eq 0 ]; then
  echo "No old Yuaz working copies found in $DOWNLOADS"
  echo "Kept current repo: $CURRENT_ROOT"
  [ -d "$CANONICAL" ] && echo "Kept canonical repo: $CANONICAL"
  exit 0
fi

echo "Old Yuaz copies found in Downloads:"
echo
for path in "${candidates[@]}"; do
  size="$(du -sh "$path" 2>/dev/null | awk '{print $1}' || true)"
  printf '  %-10s %s\n' "${size:-?}" "$path"
done

echo
echo "Protected:"
echo "  current repo:   $CURRENT_ROOT"
[ -d "$CANONICAL" ] && echo "  canonical repo: $CANONICAL"
echo

if [ "$mode" = "--list" ]; then
  exit 0
fi

if [ "$mode" != "--yes" ]; then
  printf "Move all listed copies to macOS Trash? [y/N] "
  read -r answer
  case "$answer" in
    y|Y|yes|YES) ;;
    *) echo "Cancelled."; exit 0 ;;
  esac
fi

moved=0
for path in "${candidates[@]}"; do
  base="$(basename "$path")"
  dest="$TRASH/$base"
  if [ -e "$dest" ]; then
    stamp="$(date +%Y%m%d-%H%M%S)"
    dest="$TRASH/${base}-${stamp}-$moved"
  fi
  mv "$path" "$dest"
  echo "Moved to Trash: $path"
  moved=$((moved + 1))
done

echo
echo "Done. Moved $moved Yuaz working copies to Trash."
echo "Nothing was permanently deleted."
