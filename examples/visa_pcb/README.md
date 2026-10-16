# 工业缺陷检测 Demo：VisA PCB

用公开数据集 VisA 的 PCB 子集（pcb1–pcb3，四类缺陷 **bent / melt / missing / scratch**）走完整条流程：
数据准备 → 审核/入库 → 训练 → 一键部署（ncnn fp32 + INT8）→ 对真值评估 → 延迟测试。

数据许可：VisA © Amazon，**CC BY 4.0**（Zou et al., ECCV 2022，<https://github.com/amazon-science/spot-diff>）。
`scripts/prepare_visa.py` 把像素掩码转成检测框（属于改动），并在生成目录写入 `VISA_ATTRIBUTION.txt`。
数据与模型都不入库（`workspace/` 已在 `.gitignore`）。

## 1. 准备数据

数据放到任意目录，用环境变量 `VISA` 指向它。压缩包约 1.8 GB；单连接很慢时，可分段下载后拼接，再核对文件大小。解压只保留 pcb1–pcb3 和 `split_csv`。

Linux / macOS：

```bash
export VISA=/path/to/VisA
curl -L -o "$VISA/VisA_20220922.tar" https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
python -c "import os, tarfile; from pathlib import Path; v=Path(os.environ['VISA']); t=tarfile.open(v/'VisA_20220922.tar'); t.extractall(v/'extracted', members=[m for m in t if m.name.split('/')[0] in ('pcb1','pcb2','pcb3','split_csv')], filter='data')"
cd examples/visa_pcb
python ../../scripts/prepare_visa.py --root "$VISA/extracted" --categories pcb1 pcb2 pcb3 \
    --labelme-out workspace/labelme --test-out "$VISA/evp_pcb_test"
```

Windows PowerShell：

```powershell
$env:VISA = "D:\data\VisA"
curl.exe -L -o "$env:VISA\VisA_20220922.tar" https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
python -c "import os, tarfile; from pathlib import Path; v=Path(os.environ['VISA']); t=tarfile.open(v/'VisA_20220922.tar'); t.extractall(v/'extracted', members=[m for m in t if m.name.split('/')[0] in ('pcb1','pcb2','pcb3','split_csv')], filter='data')"
cd examples\visa_pcb
python ..\..\scripts\prepare_visa.py --root "$env:VISA\extracted" --categories pcb1 pcb2 pcb3 `
    --labelme-out workspace\labelme --test-out "$env:VISA\evp_pcb_test"
```

- 采用 VisA 官方**监督划分** `split_csv/2cls_highshot.csv`：每类 60 张缺陷图进训练、40 张进测试；另外每类抽 60/20 张
  良品图作负样本。（`1cls.csv` 把缺陷图全部放在测试集，只适合无监督异常检测。）
- 掩码值到缺陷类型的对应关系从单标签样本自动推断（pcb1–3：1 bent、2 melt、3 scratch、4 missing）。
- 掩码碎片在 15 px 内合并，最长边 < 8 px 的噪点丢弃（`--merge-px`、`--min-side`）。
- 结果：训练 360 张（180 张缺陷图，293 个框：melt 148、missing 63、scratch 54、bent 28），
  测试 180 张（120 张缺陷图，185 个框），测试集只用于 `evp eval`，不参与训练、校准和验证。

## 2. 审核、入库、训练、部署

在工作台里做（`evp webui -c project.yaml --open`）：② 页签逐张检查自动生成的框 → ③ 入库 → 训练 → ④ 一键部署。
命令行等价：

```powershell
evp ingest -c project.yaml                # 360 张 -> corpus（train 306 / val 54）
evp train  -c project.yaml                # yolo11n，640，80 epoch（早停 30）
evp deploy -c project.yaml                # 640 输入：fp32 + INT8 + 验证报告
evp deploy -c project.yaml --set deploy.imgsz=[320,320] --set deploy.name=pcb_320
```

## 3. 评估与测速

```powershell
evp eval  -c project.yaml --bundle workspace/deploy/best --data "$env:VISA\evp_pcb_test" --split test `
          --weights workspace/runs/pcb/weights/best.pt
evp bench --bundle workspace/deploy/best --images "$env:VISA\evp_pcb_test\images\test" --threads 1 2 4
```

## 4. 参考结果（2026-10-05，x86 CPU）

训练：yolo11n（COCO 预训练）微调，640 输入，80 epoch，CPU（i7-14700KF）1.2 小时；验证集 mAP50 0.406。数字来自开发机，用来说明工具链，不是板端指标。

**部署验证**（`report.md`，两种输入尺寸都 PASS，推荐 INT8）：

| 检查 | 640 | 320 |
|---|---|---|
| L1 ncnn fp32 vs PyTorch 张量余弦 | 1.0000 | 1.0000 |
| L2 INT8 vs fp32 检测感知相似度 | 0.986 | 0.989 |
| L3 ncnn fp32 + 配套后处理 vs Ultralytics predict，检测 F1 | 1.000 | 1.000 |
| L3c C++ 运行时 vs Python（fp32 输入张量差 / 框差） | 0 / 8.8e-5 px | 0 / 5.7e-5 px |
| 混合精度 | 全 INT8 相似度 0.72 → conv_83 保持 fp32 后 0.97 | 0.95 → 0.98（同一层） |

**对真值的精度**（独立测试集 180 张 / 185 个缺陷，`evp eval`，同一套指标）：

| 输入 | 模型 | mAP50 | mAP50-95 | P@0.25 | R@0.25 |
|---|---|---|---|---|---|
| 640 | PyTorch .pt | 0.539 | 0.301 | 0.784 | 0.492 |
| 640 | ncnn fp32 | 0.537 | 0.302 | 0.776 | 0.486 |
| 640 | ncnn INT8（混合精度） | **0.551** | **0.306** | 0.784 | 0.492 |
| 320 | PyTorch .pt | 0.391 | 0.194 | 0.797 | 0.319 |
| 320 | ncnn fp32 | 0.405 | 0.199 | 0.808 | 0.319 |
| 320 | ncnn INT8（混合精度） | 0.369 | 0.186 | 0.792 | 0.308 |

各类 AP50（640，pt / fp32 / INT8）：bent 0.736 / 0.736 / 0.756，melt 0.461 / 0.450 / 0.452，
missing 0.449 / 0.451 / 0.494，scratch 0.510 / 0.510 / 0.504。

**主机延迟**（`evp bench`，x86 上的 Python 参考运行时，含前后处理；不是板端数据）：

| 输入 | 精度 | 权重 | 1 线程 | 2 线程 | 4 线程 |
|---|---|---|---|---|---|
| 640 | fp32 | 10.33 MB | 78.3 ms | 54.5 ms | 50.2 ms |
| 640 | INT8 | 2.65 MB | 64.4 ms | 42.8 ms | 32.8 ms |
| 320 | fp32 | 10.33 MB | 18.5 ms | 11.7 ms | 9.3 ms |
| 320 | INT8 | 2.65 MB | 14.2 ms | 9.5 ms | 7.5 ms |

结论：

- 转换无损：ncnn fp32 与 PyTorch 的 mAP 差在 ±0.015 内（差异来自 letterbox 填充方式：Ultralytics predict 用最小矩形填充，
  部署用固定方形输入，与板端一致）。
- 量化：640 下 INT8 精度不降（差值在测试集波动范围内），权重缩小 74%，主机提速 18%–35%；320 下 INT8 比 fp32 低 0.036 mAP50。
- 输入尺寸是主要权衡：缺陷小（melt 中位最长边约 17–42 px，原图 1400 px 宽），320 输入 mAP50 降 0.15、主机快约 4 倍。
  RK3506（3×A7）上的取舍需要板端实测后决定。
- 模型本身精度有限（训练缺陷图仅 180 张，missing/scratch 召回偏低）；更多数据或 GPU 长训练可提升，这与部署工具链无关。

复现：本节数字来自 `workspace/deploy/pcb_{640,320}/{report,eval_test,bench}.md`。
