# RS-CalVision 演示数据

本目录用于生成和运行一张类别更均衡的 `10000×10000` 演示大图。它复用项目现有 `data_v2` 验证图像与 YOLO 标签，不改变训练集、模型、阈值或正式代理评测集。

演示图由 `12×12` 共144个遥感裁剪图拼接而成，其中舰船60张、飞机60张、车辆24张。每个单元严格使用算法审计参数中的 `800×800` 尺寸，不再缩放到769像素，因此与推理网格完全对齐。它仍然是代理拼接图，不是连续卫星场景。

`data_v2` 验证集中只有7张车辆源图，因此生成器会确定性复用这些图像。页面和清单会如实标明这一数据限制。

## 生成

```bash
cd /root/autodl-tmp/TZB2026_RS-CalYOLO26x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate torch_env
python demo_showcase/build_demo_mosaic.py
```

生成内容：

```text
demo_showcase/generated/images/demo_scene_grouped/big_demo_scene_grouped_000001.jpg
demo_showcase/generated/labels/demo_scene_grouped/big_demo_scene_grouped_000001.txt
demo_showcase/generated/manifests/demo_scene_grouped_manifest.jsonl
demo_showcase/preview/demo_mosaic_preview.jpg
demo_showcase/demo_manifest.json
```

## 提交检测

保持 `tmux long1` 中的 API 正在运行，然后在普通终端执行：

```bash
python demo_showcase/run_demo_detection.py
```

脚本会提交图片和标签、每2秒查询一次状态，并在成功后打印可直接访问的工作台地址。该任务执行真实 GPU 推理，不使用前端模拟框。新演示图的指标必须以服务器实际运行结果为准，不能根据旧的舰船图结果推断，也不能将代理拼接图指标当作官方隐藏测试集成绩。
