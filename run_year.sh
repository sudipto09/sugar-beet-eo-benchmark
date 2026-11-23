#!/usr/bin/env bash
# Full pipeline for one year, in order.
#
#   bash run_year.sh 2021 <url-or-path-to-DLR-CropTypes-V021-GeoTIFF-for-that-year>
#
# Steps: polygons (non-2022 years, if missing) -> scene bands ->
# processing-baseline check -> targets -> Prithvi extraction in all five
# modes -> temporal probing -> clustering benchmark.
#
# BEST_LAYER defaults to 5, the layer chosen on 2022. Using it unchanged for
# another year avoids picking the layer on the data it is then scored on;
# experiment A in the probe output still prints that year's own best layer
# for comparison. Override with BEST_LAYER=<n>.
set -euo pipefail

YEAR="${1:?usage: bash run_year.sh YEAR [RASTER]}"
RASTER="${2:-}"
BEST_LAYER="${BEST_LAYER:-5}"
LIKE="Data/sugar_beet_fields_franconia_hires.geojson"
POLY="Data/sugar_beet_fields_${YEAR}.geojson"

if [[ "$YEAR" != "2022" && ! -f "$POLY" ]]; then
  [[ -n "$RASTER" ]] || { echo "need the DLR raster for $YEAR to cut polygons"; exit 1; }
  echo "== polygons ($YEAR)"
  python make_polygons_from_croptypes.py --year "$YEAR" --raster "$RASTER" --like "$LIKE"
fi

echo "== scene bands ($YEAR)";           python download_prithvi_bands.py --year "$YEAR"
echo "== processing baseline (all years)"; python check_processing_baseline.py --verify
echo "== targets ($YEAR)";               python compute_targets.py --year "$YEAR"

for MODE in none shuffle reverse repeat_first mask_last; do
  echo "== extract $MODE ($YEAR)"
  python extract_prithvi_v4.py --year "$YEAR" --mode "$MODE"
done

echo "== probing ($YEAR)";   python probe_temporal.py --year "$YEAR" --best-layer "$BEST_LAYER"
echo "== clustering ($YEAR)"; python run_seeds_v2.py --year "$YEAR" --best-layer "$BEST_LAYER"
echo "Done. Compare Data/probe_*_${YEAR}.csv with the 2022 files."