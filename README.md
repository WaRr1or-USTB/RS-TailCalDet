# RS-TailCalDet

RS-TailCalDet（Remote Sensing Tail-Aware and Calibration-Enhanced Detector）是面向光学遥感超大幅面影像的 25 类目标检测方法。该方法以 YOLO26x 为基础检测器，集成长尾数据组织、滑窗推理、全图坐标重构、跨切片逐类 NMS、类别阈值校准和 V1.6 评分审计。

## 项目概览

RS-TailCalDet 用于舰船、飞机和车辆目标检测：

- 舰船 4 类：HM、LQS、QHS、MS，类别 ID 0–3。
- 飞机 20 类：A1_SU-35 至 A20_SU-24，类别 ID 4–23。
- 车辆 1 类：FSC，类别 ID 24。

核心能力：

- 60% 训练、10% 校准、30% 锁定验证的场景级互斥数据协议。
- 面向 10000×10000 影像的无遗漏滑窗推理和原图坐标回映。
- 跨切片逐类别 Hard NMS 与 25 类独立置信度阈值。
- 综合指标、严格 25 类指标、大类指标和 V1.6 小类宏平均审计。
- 模型、阈值、数据划分及锁定评测结果的可复现记录。

## 基准结果

最终模型使用固定权重和固定 r925 类别阈值，在 30.0156% 锁定验证集构建的三种 10000×10000 代理大图布局上评测。

| 布局 | 图像数 | 综合 Recall | 综合 FDR | Strict25 Recall | Strict25 FDR | 平均耗时 | 最大耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `val_scene_grouped` | 10 | 95.31% | 14.43% | 93.45% | 16.10% | 7.49 s | 8.32 s |
| `val_mixed_stress` | 10 | 95.19% | 13.77% | 93.38% | 15.40% | 7.47 s | 8.04 s |
| `val_sparse` | 3 | 99.33% | 16.76% | 98.67% | 17.32% | 7.53 s | 7.98 s |

报告用性能图、精确汇总数据及证据来源见 [docs/results/README.md](docs/results/README.md)。

三种布局均满足当前评分协议的硬性要求：

- 综合 Recall ≥ 85%。
- 综合 FDR ≤ 20%，`FDR = FP / (TP + FP)`。
- 单张 10000×10000 影像处理时间 ≤ 20 秒。

以上结果来自项目自建锁定验证协议，不代表官方隐藏测试集成绩。

## 正式模型

| 资产 | 路径 | SHA-256 |
| --- | --- | --- |
| 模型权重 | `runs/detect/yolo26x_report60_cal10_e120_s2026_b6/weights/best.pt` | `72f7fbe5d531faa9a0a6e21b869b2c54a4d0de512b347b0e8259ca13e4e2ddd8` |
| 类别阈值 | `runs/detect/report_phase1_thresholds_r925_s2026/class_thresholds.json` | `b9122b09266b0f5d9c88c9866670e76cd28949f978bdff5f21b28f3202c55d49` |
| 训练参数 | `runs/detect/yolo26x_report60_cal10_e120_s2026_b6/args.yaml` | 见文件 |
| 训练曲线 | `runs/detect/yolo26x_report60_cal10_e120_s2026_b6/results.csv` | 见文件 |

模型权重由 Git LFS 管理。克隆后必须确认 LFS 文件已完整下载：

```bash
git lfs pull
sha256sum \
  runs/detect/yolo26x_report60_cal10_e120_s2026_b6/weights/best.pt \
  runs/detect/report_phase1_thresholds_r925_s2026/class_thresholds.json
```

## 环境要求

已核验环境：

```text
Ubuntu 22.04.1 LTS
Python 3.10.20
PyTorch 2.5.1+cu121
CUDA 12.1
NVIDIA GeForce RTX 3090 24 GiB
```

服务器环境初始化：

```bash
cd /root/autodl-tmp/TZB2026_RS-CalYOLO26x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate torch_env

export OMP_NUM_THREADS=8
export PYTHONPATH="$PWD/ultralytics-main"
export YOLO_CONFIG_DIR="$PWD/.ultralytics_config"
```

项目依赖见 `requirements.txt`；服务器关键运行时版本以本节记录为准。

## 数据协议

官方数据按原始场景组划分，避免同一场景的相邻裁剪图跨集合出现。外部数据和派生增强数据仅用于训练。

| 数据部分 | 图像数 | 目标数 | 用途 |
| --- | ---: | ---: | --- |
| 训练集 | 2976 | 13269 | 模型训练 |
| 校准集 | 448 | 2163 | 权重选择和类别阈值校准 |
| 锁定验证集 | 1345 | 6317 | 最终报告评测 |

官方原始图像共 4481 张，训练、校准和验证比例分别为 59.9866%、9.9978% 和 30.0156%。三个集合之间无场景组交叉。

Git 仓库仅保存可复现的数据协议，不保存原始竞赛影像：

```text
data_v2/report_split_v16/
├── dataset_report.yaml
├── split_summary.json
└── assignments.csv
```

原始数据应通过团队内部存储单独分发，并恢复到 `data_v2/` 对应路径。

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd RS-TailCalDet
git lfs pull
```

### 2. 检查环境与资产

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
PYTHONPATH="$PWD/ultralytics-main" python -c "import ultralytics; print(ultralytics.__file__)"

python -c "from rs_calvision.manifest import MethodRegistry; MethodRegistry().get('rs-tailcaldet-v1').verify_assets(); print('ASSETS=VERIFIED')"
```

### 3. 运行大图评测

```bash
ROOT="$PWD"
MODEL="$ROOT/runs/detect/yolo26x_report60_cal10_e120_s2026_b6/weights/best.pt"
THRESH="$ROOT/runs/detect/report_phase1_thresholds_r925_s2026/class_thresholds.json"

python scripts/evaluate_big_images_sliding.py \
  --model "$MODEL" \
  --data "$ROOT/data_big_report_validation_s2026/dataset.yaml" \
  --ultralytics-root "$ROOT/ultralytics-main" \
  --class-thresholds "$THRESH" \
  --split val_scene_grouped \
  --output-dir "$ROOT/runs/detect/report_recheck_val_scene_grouped" \
  --device 0 \
  --batch 1 \
  --imgsz 1024 \
  --tile-size 800 \
  --stride 800 \
  --pred-conf 0.001 \
  --pred-iou 0.70 \
  --global-iou 0.70 \
  --max-det 300 \
  --target-recall 0.85 \
  --max-fdr 0.20 \
  --time-limit 20
```

评测其他布局时，仅修改 `--split` 和 `--output-dir`。

## 训练复现

正式模型从 `yolo26x.pt` 初始化，训练输入尺寸为 1024，batch=6，随机种子为 2026，最多训练 120 个 epoch，patience=30。早停后共记录 99 个 epoch，训练耗时 20987.3 秒，即 5 小时 49 分 47.3 秒。

```bash
bash scripts/run_report_retrain.sh phase1
```

校准集最优记录：

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.92247 | 0.83434 | 0.87238 | 0.68773 |

脚本不会覆盖已有同名训练目录。进行消融实验时应设置新的运行名称，并保持数据划分、随机种子和评测协议一致。

## 阈值校准

正式阈值文件包含 25 个独立类别阈值。`r925` 表示校准优化的总体召回率约束为 0.925，不表示统一置信度阈值为 0.925。

校准集上，逐类别阈值将 FP 从 3214 降至 370，FDR 从 60.57% 降至 15.61%；Recall 从 96.72% 调整为 92.51%，满足预设校准约束。

完整数据构建、校准和锁定评测流程：

```bash
bash scripts/run_report_evaluation.sh
```

该脚本使用新目录保存复现实验，不覆盖冻结结果。

## 推理协议

```text
imgsz=1024
tile_size=800
stride=800
pred_conf=0.001
pred_iou=0.70
global_iou=0.70
max_det=300
batch=1
precision=FP32
```

10000×10000 影像生成 13×13 共 169 个切片。切片预测恢复至切片坐标后叠加窗口偏移，再执行跨切片逐类别 Hard NMS 和类别阈值过滤。

## 测试

```bash
python -m py_compile \
  scripts/evaluate_big_images_sliding.py \
  scripts/evaluate_official_thresholds.py \
  scripts/optimize_class_thresholds.py

python -m unittest discover -s tests -p "test_*.py" -v
```

当前自动化测试共 29 项，覆盖数据划分、评分口径、滑窗规划、全图 NMS、平台清单、结果追溯和 Docker 输出接口。

## 项目结构

```text
configs/              正式方案与平台配置
app/                  Docker 推理入口与 RS-TailCalDet 推理实现
data_v2/              本地数据及可提交的数据协议
docs/report/          研究报告与图表素材
frontend/             Web 前端
models/               Docker 构建使用的冻结模型与类别阈值
rs_calvision/         平台后端与领域逻辑
runs/detect/          冻结模型、阈值和评测证据
scripts/              数据构建、训练、推理和评测脚本
tests/                自动化测试
third_party/          第三方来源说明
ultralytics-main/     项目使用的 Ultralytics 源码
```

## 团队协作

- 使用私有 Git 仓库保存代码、配置、数据协议、报告材料和最终模型。
- 使用 Git LFS 管理 `.pt` 权重，不将原始数据或训练缓存提交到 Git。
- 每项消融实验使用独立分支和独立运行名称。
- 实验结论必须记录模型哈希、阈值哈希、数据划分、随机种子和评测输出目录。
- 正式基线以 `configs/final_solution.yaml` 为准。

## Docker 状态

仓库已实现赛事统一 Docker 接口。容器接收 `--input` 和 `--output`，仅扫描输入目录第一层的 JPG、JPEG、PNG、BMP 文件，并将结果写入 `/output/result.json`。推理固定使用冻结的 RS-TailCalDet 权重、25 类阈值、800×800 无重叠滑窗、1024×1024 网络输入和 IoU 0.70 的跨切片逐类 Hard NMS。

在 Linux x86_64 的已验证 Conda 环境中执行以下准备命令：

```bash
bash scripts/prepare_docker_delivery.sh
```

该脚本复制冻结权重和阈值到 `models/`，并按赛事要求由当前 Linux 环境导出不含 `prefix` 的 `environment.yml`。随后在项目根目录构建 `linux/amd64` 镜像，并使用 NVIDIA Container Toolkit 挂载 `/input` 与 `/output` 完成 GPU 自测。镜像推送和 tag 以赛事评测管理系统当次生成的地址为准。

## 许可与数据

项目尚未声明仓库级开源许可证，默认仅限团队内部使用。竞赛数据、外部数据、Ultralytics 源码和模型权重分别受其来源条款约束；公开仓库前必须完成数据授权、第三方许可证和敏感文件审查。
