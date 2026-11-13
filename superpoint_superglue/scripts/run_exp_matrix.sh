#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/build/superpointglue_image"
WEIGHTS="$ROOT/weights"
PAIR_ROOT="$ROOT/image/exp_pairs"
OUT_ROOT="${1:-$ROOT/build/exp_results}"
TAG="${2:-baseline}"
PAIRS_TSV="${3:-$PAIR_ROOT/pairs.tsv}"

mkdir -p "$OUT_ROOT/$TAG"
RESULT_CSV="$OUT_ROOT/$TAG/results.csv"
echo "tag,scenario,config,width,height,kpts0,kpts1,matches,sp0_ms,sp1_ms,sg_ms,total_ms,match_png" > "$RESULT_CSV"

export SPSG_WARMUP="${SPSG_WARMUP:-5}"
export SPSG_ITERS="${SPSG_ITERS:-20}"

run_one() {
  local scenario="$1"
  local config_name="$2"
  local img0="$3"
  local img1="$4"
  local stem="${scenario}__${config_name%.*}"
  local out_png="$OUT_ROOT/$TAG/${stem}.png"
  local log="$OUT_ROOT/$TAG/${stem}.log"
  echo "=== [$TAG] $scenario / $config_name ==="
  set +e
  "$BIN" "$ROOT/config/$config_name" "$WEIGHTS" "$img0" "$img1" "$out_png" >"$log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "FAILED rc=$rc, see $log"
    echo "$TAG,$scenario,$config_name,,,,,,,FAIL,,,$out_png" >> "$RESULT_CSV"
    return 0
  fi
  local line
  line="$(grep '^RESULT' "$log" | tail -n1 || true)"
  if [[ -z "$line" ]]; then
    echo "NO RESULT line, see $log"
    echo "$TAG,$scenario,$config_name,,,,,,,NO_RESULT,,,$out_png" >> "$RESULT_CSV"
    return 0
  fi
  local size kpts0 kpts1 matches sp0 sp1 sg total
  size="$(echo "$line" | sed -n 's/.* size=\([^ ]*\).*/\1/p')"
  kpts0="$(echo "$line" | sed -n 's/.* kpts0=\([^ ]*\).*/\1/p')"
  kpts1="$(echo "$line" | sed -n 's/.* kpts1=\([^ ]*\).*/\1/p')"
  matches="$(echo "$line" | sed -n 's/.* matches=\([^ ]*\).*/\1/p')"
  sp0="$(echo "$line" | sed -n 's/.* sp0_ms=\([^ ]*\).*/\1/p')"
  sp1="$(echo "$line" | sed -n 's/.* sp1_ms=\([^ ]*\).*/\1/p')"
  sg="$(echo "$line" | sed -n 's/.* sg_ms=\([^ ]*\).*/\1/p')"
  total="$(echo "$line" | sed -n 's/.* total_ms=\([^ ]*\).*/\1/p')"
  local w="${size%x*}"
  local h="${size#*x}"
  echo "$TAG,$scenario,$config_name,$w,$h,$kpts0,$kpts1,$matches,$sp0,$sp1,$sg,$total,$out_png" >> "$RESULT_CSV"
  echo "$line"
}

# Fix campus config if needed
sed -i 's/I_campus\tconfig_640x480.yaml/I_campus\tconfig_640x512.yaml/' "$PAIRS_TSV" || true

while IFS=$'\t' read -r scenario config desc; do
  [[ "$scenario" == "scenario" ]] && continue
  [[ -z "$scenario" ]] && continue
  # 320 reference uses A_indoor_near images
  if [[ "$scenario" == *_320 ]]; then
    src_scenario="${scenario%_320}"
  else
    src_scenario="$scenario"
  fi
  d="$PAIR_ROOT/$src_scenario"
  img0="$(ls "$d"/image0.* | head -n1)"
  img1="$(ls "$d"/image1.* | head -n1)"
  run_one "$scenario" "$config" "$img0" "$img1"
done < "$PAIRS_TSV"

echo "Wrote $RESULT_CSV"
