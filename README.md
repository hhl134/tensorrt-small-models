# tensorrt-small-models

面向部署的小模型 TensorRT 工程合集：特征匹配、视觉定位描述子、目标检测。代码在本仓库；推荐权重与引擎在 Hugging Face。

- 代码：https://github.com/hhl134/tensorrt-small-models
- 权重：https://huggingface.co/hhl134/tensorrt-small-models-weights

## 环境（实测）

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080 16GB |
| 驱动 / CUDA | 595.x / 工具包 12.8 |
| TensorRT | 10.8.0.43 |
| OpenCV | 4.6 |
| Python（导出） | conda `3d_recon`，PyTorch 2.12 cu128 |

## 目录

| 目录 | 内容 |
| --- | --- |
| `superpoint_superglue/` | SuperPoint + SuperGlue TensorRT |
| `aliked_lightglue/` | ALIKED-n16 + LightGlue TensorRT |
| `mixvpr/` | MixVPR（ResNet50 → 4096）TensorRT |
| `yolo/` | YOLOv8n：ONNX 路径 + tensorrtx wts 路径改动说明 |
| `eval/hpatches/` | HPatches 评测辅助脚本 |

权重默认放各子目录 `weights/`，从 HF 下载后放入即可。

## 上游署名

本仓在以下开源工作之上做 TensorRT 10 部署、精度对齐与工程优化，保留各自许可证要求：

| 模块 | 上游 |
| --- | --- |
| SuperPoint / SuperGlue | [yulunyue/SuperPoint-SuperGlue-TensorRT](https://github.com/yulunyue/SuperPoint-SuperGlue-TensorRT)、[magicleap/SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) |
| ALIKED / LightGlue | [cvg/LightGlue](https://github.com/cvg/LightGlue)、[cubestudio/ALIKED](https://github.com/cubestudio) 相关实现、本仓独立 C++ TensorRT 管线 |
| MixVPR | [amaralibey/MixVPR](https://github.com/amaralibey/MixVPR) |
| YOLO | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)、[wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)（wts 路径） |

## 推荐配置（默认）

| 模型 | 推荐 | 不推荐作默认 |
| --- | --- | --- |
| SuperPoint + SuperGlue | Indoor SuperGlue；SuperPoint FP16；SG 混合精度（Sinkhorn 的 Exp/Log/ReduceSum 留 FP32）；`enqueueV3` / 异步；GPU `grid_sample`；Buffer 复用 | SuperPoint INT8；点数压到 128；整网硬 FP16（曾导致匹配为 0） |
| ALIKED + LightGlue | 检测阈值 0.2；提取器 / LG 上限 **1024**；FP16；Exp/Log/ReduceSum 留 FP32；关闭 Flash / 早停 / 剪枝 | 固定 256 点；默认开到 3840（显存与匹配耗时平方涨） |
| MixVPR | **4096** 维；FP16；L2 的 ReduceSum 留 FP32；输入 320、ImageNet 归一化 | 仅用 128 维当主配置（召回明显低） |
| YOLO | **ONNX FP16** + GPU letterbox/NMS；填充 114；部署阈值 0.25 / NMS 0.45 | INT8 PTQ（ONNX mAP 掉到 ~23；wts INT8 ~31） |

YOLO 若更看重速度、可接受约 0.5 mAP 差距，可用 wts FP16。

---

## SuperPoint + SuperGlue

### 主要改动

- 适配 TensorRT 10；SuperGlue 混合精度：敏感层留 FP32，避免 FP16 匹配崩溃。
- 前后处理向量化、Buffer 复用、`enqueueV3`、异步拷贝、可选双图并行提点、GPU `grid_sample`。
- SuperPoint INT8 PTQ 做过：匹配变少，不当默认。

### HPatches（640×480，580 对）

| 改动 | 平均匹配 | Acc@1/3/5 | AUC@5 |
| --- | --- | --- | --- |
| 整网 FP32 SuperGlue | 307.4 | 26.9 / 67.2 / 80.3 | 53.3 |
| 混合精度 sk100 + GPU 采样（推荐） | 307.3 | 28.6 / 66.4 / 80.3 | 53.2 |
| Outdoor 权重 | 269.0 | 29.0 / 70.5 / 84.8 | 55.9 |
| k=128 | 58.6 | 24.5 / 56.6 / 70.9 | 45.2 |
| SuperPoint INT8 | 259.6 | 29.3 / 65.5 / 81.7 | 53.1 |

视角子集 295 对上，推荐混合精度 AUC@5 ≈ 38.2（FP32 基线 39.1）。

### HF 权重（建议）

- `superpoint_v1_sim_int32_trt10.onnx` / 对应 FP16 engine  
- `superglue_indoor_fp16_stable.engine`（推荐）及 indoor/outdoor ONNX  

---

## ALIKED + LightGlue

### 主要改动

- 可变形卷积 → 偏移截断后的 `grid_sample` + 1×1（与 torchvision 误差约 1e-6）。
- 关掉 Flash / 早停 / 剪枝以便静态导出；阈值检测 + 动态点数 LightGlue。
- 默认上限从实验中的 3840 收到 **1024**（精度几乎不掉，显存与耗时大降）。

### HPatches（阈值 0.2，580 对）

| 版本 | 平均匹配 | Acc@5 | AUC@5 |
| --- | --- | --- | --- |
| PyTorch 原版 | 706.7 | 85.7 | 60.0 |
| TensorRT 上限 3840 FP16 | 707.4 | 85.0 | 59.1 |
| TensorRT 上限 1024（推荐） | — | — | **59.0** |
| TensorRT 固定 256 | 132.5 | 81.7 | 55.7 |

### 点数上限与资源（样例对）

| 上限 | 合计耗时 | 合计显存 |
| --- | --- | --- |
| 512 | 6.41 ms | 242 MiB |
| **1024** | ~8–10 ms | **279 MiB** |
| 3840 | 83.40 ms | 897 MiB |

### HF 权重（建议）

- `aliked_n16_k1024.onnx` / `aliked_n16_k1024_fp16.engine`  
- `aliked_lightglue_dyn.onnx` / `aliked_lightglue_dyn_k1024_fp16.engine`  

---

## MixVPR

### 主要改动

- 按官方结构重写：ResNet50（去最后一级）+ MixVPR 聚合 → L2 归一化 4096 维。
- TensorRT FP16；L2 中 ReduceSum 留 FP32。前处理与官方一致（RGB、320、ImageNet mean/std）。

### 样例描述子

| 版本 | 与 PyTorch 余弦 | 两图耗时 |
| --- | --- | --- |
| PyTorch FP32 | — | 3.476 ms |
| TensorRT FP16（推荐） | 0.999990 | 1.348 ms |

### Nordland 召回（Patch-NetVLAD 划分）

前 1369 对：

| 版本 | R@1 | R@5 | R@10 |
| --- | --- | --- | --- |
| PyTorch 4096 | 98.32 | 99.42 | 99.56 |
| TensorRT FP16 4096 | 98.32 | 99.42 | 99.56 |

全长 27592 对：

| 版本 | R@1 | R@5 | R@10 |
| --- | --- | --- | --- |
| PyTorch 4096 | 76.20 | 86.89 | 90.37 |
| TensorRT FP16 4096 | 76.20 | 86.94 | 90.37 |
| TensorRT 512 FP16 | 67.06 | 81.06 | 85.88 |
| TensorRT 128 FP16 | 47.93 | 66.51 | 73.96 |

### HF 权重（建议）

- `mixvpr_resnet50_4096.onnx`  
- `mixvpr_resnet50_4096_fp16.engine`  

---

## YOLO（YOLOv8n）

### 主要改动

- ONNX 与 wts 两条路径；填充对齐 **114**；部署 conf/NMS **0.25 / 0.45**；框映回原图。
- ONNX：GPU letterbox + NMS；端到端约 **0.67 ms** 量级（RTX 5080）。
- wts：插件低分框阈值改为 0.001、输出上限 4096，便于 COCO 协议评 mAP。
- INT8 PTQ（ONNX + wts）已测；掉点大，**不当默认**。QAT 路线见子目录文档/部署笔记（进行中）。

### COCO val2017 mAP（conf 0.001，NMS 0.7，maxDets=300）

| 路径 | mAP50-95 | mAP50 | mAP75 |
| --- | --- | --- | --- |
| PyTorch | 36.73 | 51.68 | 39.88 |
| ONNX FP16（推荐） | 36.77 | 51.74 | 39.96 |
| wts FP16 | 36.20 | 51.48 | 39.38 |
| ONNX INT8 PTQ | 22.99 | 34.23 | 24.84 |
| wts INT8 PTQ | 31.30 | 45.15 | 34.14 |

### 耗时（bus.jpg，约）

| 路径 | 网络+NMS | 引擎大小 |
| --- | --- | --- |
| ONNX FP16 | ~0.60 ms | ~11 MB |
| ONNX INT8 | ~0.47 ms | ~7.3 MB |
| wts FP16 | ~0.58 ms | ~9.8 MB |

### HF 权重（建议）

- `yolov8n.pt` / `yolov8n.onnx` / `yolov8n.wts`  
- `yolov8n_fp16.engine` / `yolov8n_wts_fp16.engine`  
- INT8 引擎可选上传，仅作实验对照  

wts 完整构建仍依赖 [tensorrtx](https://github.com/wang-xinyu/tensorrtx) 的 YOLOv8 工程；本仓 `yolo/` 侧重 ONNX 管线与对照脚本，关键改动说明见子目录。

---

## 下载权重

```bash
# 需要已 hf auth login
huggingface-cli download hhl134/tensorrt-small-models-weights --local-dir /tmp/tsm-weights
# 再按子目录拷到各工程 weights/
```

或在浏览器打开权重页按文件下载。

## 构建提示

各子目录均为独立 CMake 工程。需本机安装 TensorRT 10、CUDA、OpenCV，并在 CMake 中指向 TensorRT 路径。具体目标名见各目录 `CMakeLists.txt`。

## 许可

本仓库代码以 MIT 发布。上游模型权重与第三方代码遵循其原许可证；使用前请自行核对。
