# RS-TailCalDet 30% 验证结果证据

本目录汇总报告数据划分后的训练过程、阈值校准结果和 30% 锁定验证集评测结果。所有图表均由仓库内冻结的 CSV/JSON 结果生成，可通过 `scripts/generate_report_results.py` 复现。

## 数据划分与冻结对象

- 官方数据划分：60% 训练集、10% 校准集、30% 锁定验证集。
- 官方图像数：训练集 2688 张、校准集 448 张、锁定验证集 1345 张。
- 外部与派生样本仅用于训练，不进入校准集和锁定验证集。
- 锁定模型：`runs/detect/yolo26x_report60_cal10_e120_s2026_b6/weights/best.pt`
- 模型 SHA-256：`72f7fbe5d531faa9a0a6e21b869b2c54a4d0de512b347b0e8259ca13e4e2ddd8`
- 锁定阈值：`runs/detect/report_phase1_thresholds_r925_s2026/class_thresholds.json`
- 阈值 SHA-256：`b9122b09266b0f5d9c88c9866670e76cd28949f978bdff5f21b28f3202c55d49`

## 报告图表

| 文件 | 内容 | 建议用途 |
|---|---|---|
| `training_curves.png` | 60% 训练集上的训练损失与 10% 校准集上的检测指标 | 训练过程与收敛分析 |
| `threshold_calibration.png` | 10% 校准集上不同召回约束阈值策略的 Recall–FDR 权衡 | 阈值选择依据 |
| `locked_overall_metrics.png` | 三种 30% 锁定验证构造下的总体 Recall、FDR | 综合性能与硬门槛对照 |
| `locked_category_metrics.png` | 舰船、飞机、车辆三大类的合并匹配指标 | 大类性能分析 |
| `locked_subclass_metrics.png` | 25 个细分类别的严格匹配 Recall、FDR | 细粒度与长尾性能分析 |
| `inference_timing.png` | RTX 3090 上 10000×10000 图像的平均、最大推理时间 | 工程效率分析 |
| `performance_summary.csv` | 所有绘图数据的精确数值 | 制表、复核和二次绘图 |

## 数据集图表

| 文件 | 内容 | 建议用途 |
|---|---|---|
| `dataset_split.png` | 60%训练、10%校准、30%锁定验证的场景级互斥划分 | 数据集划分说明 |
| `class_distribution.png` | 60%官方训练子集的25类长尾分布 | 类别不均衡分析 |
| `broad_category_distribution.png` | 三个数据子集中的舰船、飞机、车辆标注数量 | 三大类分布分析 |
| `object_scale_distribution.png` | 按COCO面积阈值统计的小、中、大目标占比 | 小目标与尺度分析 |
| `dataset_statistics_overview.png` | 类别共现、25类长尾和相对目标尺度三联图 | 数据集统计主图 |
| `representative_training_samples.png` | 舰船、飞机、FSC及复杂背景真实训练样本 | 数据样例与标注展示 |
| `dataset_statistics.csv` | 数据划分、大类数量和目标尺度的精确统计 | 数据表与复核 |

## 30% 锁定验证结果

| 验证构造 | 总体 Recall | 总体 FDR | 严格 25 类 Recall | 严格 25 类 FDR | 平均时间 | 最大时间 |
|---|---:|---:|---:|---:|---:|---:|
| Scene-grouped | 95.31% | 14.43% | 93.45% | 16.10% | 7.488 s | 8.318 s |
| Mixed-stress | 95.19% | 13.77% | 93.38% | 15.40% | 7.474 s | 8.039 s |
| Sparse | 99.33% | 16.76% | 98.67% | 17.32% | 7.525 s | 7.981 s |

比赛硬门槛以合并大类后的总体指标判定：Recall 不低于 85%，FDR 不高于 20%，10000×10000 单图时间不超过 20 秒。三种内部锁定验证构造均满足上述门槛。时间统计使用 RTX 3090，当前记录不包含图像读取时间。

## 报告章节建议

- 结果分析中的训练设置与收敛过程：使用 `training_curves.png`。
- 阈值选择说明：使用 `threshold_calibration.png`，同时明确阈值仅由 10% 校准集确定。
- 综合性能：使用 `locked_overall_metrics.png`、`locked_category_metrics.png` 和 `locked_subclass_metrics.png`。
- 工程效率：使用 `inference_timing.png`，并注明计时口径。

## 尚需额外实验才能形成的结果图

以下内容不能从现有汇总 CSV/JSON 中可靠还原，若报告需要，必须重新评测并保存逐框预测、匹配关系或对应实验结果：

- 同一 30% 锁定验证集上的基线模型与 RS-TailCalDet 对比表和可视化。
- 各技术模块逐项启用/关闭的消融实验表。
- PR 曲线、F1–Confidence 曲线和混淆矩阵。
- 典型漏检、误检及其原因分析图。
- 超大幅面原图上的最终检测框叠加图及结果追溯图。

这些图应由固定模型、固定阈值和固定 30% 验证清单重新导出，不能使用训练集结果或不同划分上的旧实验替代。

## 复现图表

在项目根目录执行：

```bash
python scripts/generate_report_results.py
python scripts/generate_dataset_report_figures.py
```

脚本仅读取已冻结的结果文件，并覆盖本目录内的图表与 `performance_summary.csv`。

## 原始证据位置

- 数据划分摘要：`data_v2/report_split_v16/split_summary.json`
- 训练日志：`runs/detect/yolo26x_report60_cal10_e120_s2026_b6/results.csv`
- 阈值校准：`runs/detect/report_phase1_thresholds_r925_s2026/`
- 锁定验证：`runs/detect/report_v16_locked_r925_val_*_s2026/`

本目录记录的是内部锁定验证结果，不等同于主办方未公开测试集成绩。
