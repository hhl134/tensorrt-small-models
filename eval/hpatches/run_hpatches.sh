#!/bin/bash
set -u
ROOT=/media/hhl/data/dataset/hpatches-sequences-release
SPSG=/home/hhl/workspace/deploy/tensorrt/SuperPoint-SuperGlue-TensorRT
ALG=/home/hhl/workspace/deploy/tensorrt/ALIKED-LightGlue-TensorRT
OUT=/home/hhl/workspace/deploy/tensorrt/hpatches_eval/out
PY=/home/hhl/miniconda3/envs/3d_recon/bin/python
mkdir -p "$OUT"

cp -a "$SPSG/weights/superglue_indoor_sim_int32.engine" "$SPSG/weights/superglue_indoor_fp32_eval.engine"
python3 - << 'PY'
from pathlib import Path
src = Path("/home/hhl/workspace/deploy/tensorrt/SuperPoint-SuperGlue-TensorRT/config/config_640x480.yaml").read_text()
src = src.replace("superglue_indoor_fp16_stable.engine", "superglue_indoor_fp32_eval.engine")
Path("/home/hhl/workspace/deploy/tensorrt/hpatches_eval/out/config_640x480_fp32.yaml").write_text(src)
PY

run_spsg() {
  local name="$1"
  local cfg="$2"
  local gpu="$3"
  local watch="$4"
  local before=""
  if [[ -n "$watch" ]]; then
    before=$(stat -c %Y "$watch")
  fi
  echo "START $name $(date +%T)"
  SPSG_GPU_SAMPLE="$gpu" "$SPSG/build/superpointglue_hpatches" \
    "$cfg" "$SPSG/weights" "$ROOT" "$OUT/${name}.txt" \
    > "$OUT/${name}.log" 2>&1
  local status=$?
  echo "DONE $name status=$status $(date +%T)"
  if [[ -n "$watch" ]]; then
    local after
    after=$(stat -c %Y "$watch")
    if [[ "$before" != "$after" ]]; then
      echo "ENGINE REBUILT $watch" | tee "$OUT/${name}.REBUILT"
    fi
  fi
  return $status
}

run_spsg spsg_fp32 "$OUT/config_640x480_fp32.yaml" 1 "$SPSG/weights/superglue_indoor_fp32_eval.engine"
run_spsg spsg_fp16_sk100 "$SPSG/config/config_640x480.yaml" 1 ""
run_spsg spsg_fp16_cpu_sample "$SPSG/config/config_640x480.yaml" 0 ""
run_spsg spsg_sk50 "$SPSG/config/config_640x480_sk50.yaml" 1 ""
run_spsg spsg_sk20 "$SPSG/config/config_640x480_sk20.yaml" 1 ""
run_spsg spsg_outdoor "$SPSG/config/config_640x480_outdoor.yaml" 1 ""
run_spsg spsg_k256 "$SPSG/config/config_640x480_k256.yaml" 1 ""
run_spsg spsg_k128 "$SPSG/config/config_640x480_k128.yaml" 1 ""
run_spsg spsg_sp_int8 "$SPSG/config/config_640x480_spint8.yaml" 1 "$SPSG/weights/superpoint_v1_int8.engine"

echo "START aliked_trt $(date +%T)"
"$ALG/build/aliked_lightglue_hpatches" \
  "$ALG/config/config.yaml" "$ALG/weights" "$ROOT" "$OUT/aliked_trt.txt" \
  > "$OUT/aliked_trt.log" 2>&1
echo "DONE aliked_trt status=$? $(date +%T)"

for variant in pytorch_default pytorch_k256_dynamic pytorch_k256_static; do
  echo "START $variant $(date +%T)"
  "$PY" "$ALG/../hpatches_eval/dump_aliked_pytorch.py" "$ROOT" "$variant" "$OUT/${variant}.txt" \
    > "$OUT/${variant}.log" 2>&1
  echo "DONE $variant status=$? $(date +%T)"
done

echo "SCORING $(date +%T)"
"$PY" /home/hhl/workspace/deploy/tensorrt/hpatches_eval/score_matches.py "$ROOT" "$OUT"/*.txt \
  | tee "$OUT/summary.txt"
echo "ALL DONE $(date +%T)"
