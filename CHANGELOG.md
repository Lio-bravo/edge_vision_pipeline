# 更新日志

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循语义化版本。

## [未发布]

### 变更
- 文档写明工具的两段工作：检测数据的标注训练，以及按模型配套的算子替换和后处理。通用 ONNX 示例在 `examples/onnx_rewrite/`。
- 文档按开源工具整理：运行步骤见 `docs/QUICKSTART.md`，开发说明见 `docs/DEVELOPMENT.md`。仓库内不再写某台开发机的路径。

### 新增
- 图像采集 `evp collect`（`evp/data/collect.py`）：文件夹导入、视频按时间抽帧、本机/RTSP 相机；精确去重，
  可选 dHash 近重复与 Laplacian 模糊过滤，与审核目录和语料库去重；配置节 `collect:`。
- 低代码 Web 工作台 `evp webui`（Gradio，可选依赖 `.[webui]`）：采集、预标注/空白标注、在线画框审核、入库、训练、
  一键部署、验证报告、单图推理与延迟测试；默认只监听 127.0.0.1。
- 精度评估 `evp eval`（`evp/deploy/evaluate.py`）：同一套 COCO 式 mAP 评估 .pt、ncnn fp32、ncnn int8。
- `evp/data/masks.py`：分割掩码 → 检测框（合并碎片、去除噪点）；`scripts/prepare_visa.py`：VisA → evp 数据。
- 工业缺陷 Demo：`examples/visa_pcb`（VisA PCB，CC BY 4.0）。
- `evp/data/ingest.py`：入库逻辑从 CLI 抽出，CLI 与工作台共用。
- 开源合规：分层许可（工具链 AGPL-3.0-only，`runtime/cpp` Apache-2.0）、NOTICE、第三方许可证清单、
  知识产权声明、SPDX 头与 `scripts/add_spdx.py --check`、贡献指南、行为准则、治理与安全策略、Gitee Issue/PR 模板。
- `train.device` / `prelabel.device` 支持 `auto`（有 CUDA 用 GPU 0，否则 CPU）；显式要求 GPU 但没有 CUDA 时报错并给出提示。
- SOP 端到端集成测试 `test_sop_end_to_end`：预标注 → 入库 → 训练 → 部署 → 用部署包预标注 → 增量入库。

### 修复
- 验证空转：没有可比内容的检查（两侧都未检出目标、L2 无信号）不再判为通过，而是记为 UNMEASURED；
  int8 没有任何测量过的门限检查时报告为 UNVERIFIED，并推荐 fp32。
- 训练默认 `device: "0"` 在无 CUDA 环境下会直接失败。

## [0.1.0] - 2026-10-04

### 新增
- 数据阶段：模型预标注（.pt 或 ncnn 部署包）→ LabelMe 复检 → json 转 YOLO、批次划分、合并进语料库（增量，旧 val 不重划）。
- 训练：基于 Ultralytics 的训练封装。
- 部署工具链：.pt 适配器（YOLO v5u/v8/v11/v12/YOLO26 检测、分类）与通用 ONNX 接入；ONNX 算子改写规则引擎；
  pnnx 转换（Windows 下自动使用 WSL）；INT8 校准（kl/aciq/auto）与混合精度；`meta.json`/`model.meta` 前后处理契约。
- 验证阶梯 L1/L2/L3/L3c 与 report.json/report.md。
- C++11 板端运行时：meta 解析、前处理、后处理、demo、单测。
