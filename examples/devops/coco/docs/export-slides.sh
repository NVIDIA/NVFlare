#!/usr/bin/env bash
set -Eeuo pipefail

[[ $# == 1 ]] || { echo "Usage: $0 NEW-OUTPUT-DIRECTORY" >&2; exit 2; }
DOCS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname -- "$DOCS_DIR")"
OUTPUT_DIR="$(realpath -m -- "$1")"
[[ "$OUTPUT_DIR" != "$PACKAGE_DIR" && "$OUTPUT_DIR" != "$PACKAGE_DIR/"* ]] \
    || { echo 'Export outside the public source package' >&2; exit 1; }
[[ ! -e "$1" && ! -L "$1" ]] || { echo 'Refusing to overwrite existing output' >&2; exit 1; }
MARP_BIN="${MARP_BIN:-marp}"
version="$("$MARP_BIN" --version)"
[[ "$version" == *'@marp-team/marp-cli v4.5.1 '* ]] \
    || { echo 'Use the documented Marp CLI 4.5.1 toolchain' >&2; exit 1; }
browser_args=()
if [[ -n "${BROWSER_BIN:-}" ]]; then
    browser_args=(--browser chrome --browser-path "$BROWSER_BIN")
fi
mkdir -m 0755 -- "$OUTPUT_DIR"
printf '%s\n' "$version" > "$OUTPUT_DIR/toolchain.txt"
node --version >> "$OUTPUT_DIR/toolchain.txt"
[[ -z "${BROWSER_BIN:-}" ]] || "$BROWSER_BIN" --version >> "$OUTPUT_DIR/toolchain.txt"
SOURCE="$DOCS_DIR/coco-security-design-3-slides.md"
for format in html pdf pptx; do
    "$MARP_BIN" "$SOURCE" "${browser_args[@]}" --output "$OUTPUT_DIR/coco-security-design-3-slides.$format"
done
sha256sum "$SOURCE" > "$OUTPUT_DIR/source-sha256.txt"
printf 'Generated slide exports in %s\nReview the rendering before publishing as documentation/release artifacts.\n' "$OUTPUT_DIR"
