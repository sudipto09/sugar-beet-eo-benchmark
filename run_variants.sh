#!/usr/bin/env bash
# Does the input resolution / image size change the conclusions?
#
#   bash run_variants.sh 2022            # 30 m test (recommended)
#   FULL=1 bash run_variants.sh 2022     # also shuffle/reverse/repeat_first and
#                                        # the full probing at 30 m
#   S224=1 bash run_variants.sh 2022     # also the 224 px test (slow)
#
# Needs the baseline extractions for the year (modes none and mask_last).
# The 30 m run pools crops to Prithvi's pretraining resolution; the 224 px
# run feeds the pretraining image size. Each writes tagged files, so the
# baseline pickles are never overwritten, then compare_variants.py reports
# the change against the 0.03 noise floor.
set -euo pipefail

YEAR="${1:?usage: bash run_variants.sh YEAR}"
LAYER="${LAYER:-5}"

for F in Data/prithvi_enc_${YEAR}_none.pkl Data/prithvi_enc_${YEAR}_mask_last.pkl; do
  [[ -f "$F" ]] || { echo "missing $F - run the baseline extraction first"; exit 1; }
done

echo "== smoke test (20 fields, 30 m)"
python extract_prithvi_v4.py --year "$YEAR" --mode none --input-gsd 30 --tag _gsd30 --limit 20

for MODE in none mask_last; do
  echo "== 30 m: $MODE"
  python extract_prithvi_v4.py --year "$YEAR" --mode "$MODE" --input-gsd 30 --tag _gsd30
done
echo "== compare 30 m vs baseline"
python compare_variants.py --year "$YEAR" --tag _gsd30 --layer "$LAYER"

if [[ "${FULL:-0}" == "1" ]]; then
  # Re-run the perturbation experiments at 30 m: is order-invariance, the
  # headline finding, still there at Prithvi's pretraining resolution?
  for MODE in shuffle reverse repeat_first; do
    echo "== 30 m: $MODE"
    python extract_prithvi_v4.py --year "$YEAR" --mode "$MODE" --input-gsd 30 --tag _gsd30
  done
  echo "== full probing at 30 m (writes probe_*_${YEAR}_gsd30.csv)"
  python probe_temporal.py --year "$YEAR" --best-layer "$LAYER" --tag _gsd30
fi

if [[ "${S224:-0}" == "1" ]]; then
  echo "== smoke test (20 fields, 224 px)"
  python extract_prithvi_v4.py --year "$YEAR" --mode none --img-size 224 --batch-size 2 --tag _s224 --limit 20
  for MODE in none mask_last; do
    echo "== 224 px: $MODE"
    python extract_prithvi_v4.py --year "$YEAR" --mode "$MODE" --img-size 224 --batch-size 2 --tag _s224
  done
  echo "== compare 224 px vs baseline"
  python compare_variants.py --year "$YEAR" --tag _s224 --layer "$LAYER"
fi