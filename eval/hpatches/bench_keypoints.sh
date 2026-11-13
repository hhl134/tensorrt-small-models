#!/usr/bin/env bash
# Time and GPU-memory sweep for ALIKED top-k and the matching LightGlue profile.
set -euo pipefail

ROOT=/home/hhl/workspace/deploy/tensorrt/ALIKED-LightGlue-TensorRT
BIN="$ROOT/build/aliked_lightglue_image"
HP="$ROOT/build/aliked_lightglue_hpatches"
WEIGHTS="$ROOT/weights"
CFG_DIR="$ROOT/config/bench"
OUT=/home/hhl/workspace/deploy/tensorrt/hpatches_eval/out
IMG0=/home/hhl/workspace/deploy/tensorrt/SuperPoint-SuperGlue-TensorRT/image/image0.png
IMG1=/home/hhl/workspace/deploy/tensorrt/SuperPoint-SuperGlue-TensorRT/image/image1.png
HP_ROOT=/media/hhl/data/dataset/hpatches-sequences-release
PY=/home/hhl/miniconda3/envs/3d_recon/bin/python

mkdir -p "$CFG_DIR" "$OUT"

write_cfg() {
    local k=$1
    local threshold=$2
    local glue_max=$3
    local glue_opt=$4
    local glue_engine=$5
    local cfg=$6
    cat > "$cfg" <<EOF
aliked:
  image_width: 640
  image_height: 480
  max_keypoints: ${k}
  desc_dim: 128
  detection_threshold: ${threshold}
  input_tensor_names:
    - "image"
  output_tensor_names:
    - "keypoints"
    - "scores"
    - "descriptors"
    - "peaks"
  onnx_file: "aliked_n16_k${k}.onnx"
  engine_file: "aliked_n16_k${k}_fp16.engine"

lightglue:
  match_threshold: 0.1
  min_keypoints: 1
  opt_keypoints: ${glue_opt}
  max_keypoints: ${glue_max}
  desc_dim: 128
  fp16: true
  input_tensor_names:
    - "kpts0"
    - "desc0"
    - "kpts1"
    - "desc1"
  output_tensor_names:
    - "scores"
  onnx_file: "aliked_lightglue_dyn.onnx"
  engine_file: "${glue_engine}"
EOF
}

run_pair() {
    local tag=$1
    local cfg=$2
    local keep=${3:-}
    echo "----- ${tag} -----"
    if [[ -n "$keep" ]]; then
        ALG_KEEP="$keep" ALG_WARMUP=5 ALG_ITERS=30 "$BIN" "$cfg" "$WEIGHTS" "$IMG0" "$IMG1" /tmp/alg_bench_match.png
    else
        ALG_WARMUP=5 ALG_ITERS=30 "$BIN" "$cfg" "$WEIGHTS" "$IMG0" "$IMG1" /tmp/alg_bench_match.png
    fi
}

# Pipeline sized for exactly K points: extractor top-k = K, matcher profile max = K.
# Threshold -1 keeps every extracted point, so both stages see K.
for k in 128 256 512 1024 2048 3840; do
    cfg="$CFG_DIR/k${k}.yaml"
    if [[ "$k" == "3840" ]]; then
        glue_engine="aliked_lightglue_dyn_fp16.engine"
        glue_opt=1024
    else
        glue_engine="aliked_lightglue_dyn_k${k}_fp16.engine"
        glue_opt=$k
    fi
    write_cfg "$k" -1 "$k" "$glue_opt" "$glue_engine" "$cfg"
    run_pair "cap=${k} threshold=-1 glue_max=${k}" "$cfg"
done

# Same 3840 engines, then keep only the strongest 512. Matcher buffers stay at 3840.
cfg3840="$CFG_DIR/k3840.yaml"
run_pair "cap=3840 keep=512 glue_max=3840" "$cfg3840" 512

# Realistic threshold on the sample pair: 3840 graph, drop peaks <= 0.2.
write_cfg 3840 0.2 3840 1024 "aliked_lightglue_dyn_fp16.engine" "$CFG_DIR/k3840_thr.yaml"
run_pair "cap=3840 threshold=0.2 glue_max=3840" "$CFG_DIR/k3840_thr.yaml"

# Extract only 512, and also drop anything below 0.2.
write_cfg 512 0.2 512 512 "aliked_lightglue_dyn_k512_fp16.engine" "$CFG_DIR/k512_thr.yaml"
run_pair "cap=512 threshold=0.2 glue_max=512" "$CFG_DIR/k512_thr.yaml"

# Accuracy: threshold 0.2, cap is the top-k baked into ALIKED.
for k in 256 512 1024 2048; do
    cfg="$CFG_DIR/k${k}_thr.yaml"
    write_cfg "$k" 0.2 "$k" "$k" "aliked_lightglue_dyn_k${k}_fp16.engine" "$cfg"
    echo "----- hpatches k=${k} -----"
    "$HP" "$cfg" "$WEIGHTS" "$HP_ROOT" "$OUT/aliked_trt_k${k}.txt"
    "$PY" /home/hhl/workspace/deploy/tensorrt/hpatches_eval/score_matches.py \
        "$HP_ROOT" "$OUT/aliked_trt_k${k}.txt"
done
