#!/bin/bash
# Renders figure PDFs to PNG for a visual check (in .temp/figures/) and verifies that none contains a raster image (Poppler's pdfimages). Arguments: PDF files or folders (default: every figures/ folder of the experiments and docs/figures).
# Usage: .claude/skills/figures/scripts/review.sh [<pdf or folder> ...] [--dpi 110]
cd "$(dirname "$0")/../../../.." || exit 1
dpi=110; args=()
while [ $# -gt 0 ]; do [ "$1" = --dpi ] && { dpi=$2; shift 2; continue; }; args+=("$1"); shift; done
[ ${#args[@]} -eq 0 ] && args=(experiments/*/figures docs/figures)
mkdir -p .temp/figures
status=0
for a in "${args[@]}"; do
  for pdf in $( [ -d "$a" ] && ls "$a"/*.pdf 2>/dev/null || echo "$a" ); do
    name=$(echo "${pdf%.pdf}" | tr '/' '_')
    pdftoppm -png -r "$dpi" -singlefile "$pdf" ".temp/figures/$name"
    rasters=$(pdfimages -list "$pdf" 2>/dev/null | tail -n +3 | wc -l | tr -d ' ')
    [ "$rasters" != 0 ] && { echo "RASTER $pdf: $rasters images"; status=1; }
  done
done
echo "rendered to .temp/figures/"
exit $status
