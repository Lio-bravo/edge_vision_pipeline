# edge_vision_pipeline (evp)

开源工具，做两件事，而且两件事接在同一条链上。

1. **标注和训练。** 目标检测从现场图片走到自己的权重：采集 → 预标注 → 人工复检 → 入库 → 训练。
2. **模型转换。** 把权重或通用 ONNX 变成没有 NPU 的 ARM CPU 上能跑的 ncnn（fp32 / INT8）。每种模型从图里拿掉的算子，都有一份配套的后处理；Python 和 C++ 按同一份 `meta` 执行。换了模型族就不能沿用另一族的后处理。

命令行和 Web 工作台读写同一份项目。目标推理框架是 ncnn。睿擎派 RK3506J（3×Cortex-A7，无 NPU）是对照板型之一。

算子替换和后处理怎么配对，见 [docs/DEPLOY_TOOLCHAIN.md](docs/DEPLOY_TOOLCHAIN.md) 里的「配套」一节，里面有可运行的通用 ONNX 示例。

```text
collect(文件夹 / 视频 / 相机，去重 + 模糊过滤)
   │
raw_images ──prelabel──► labelme/*.json ──人工复检──► ingest ──► corpus ──train──► best.pt
                                                                                 │
                     ┌───────────────────── evp deploy ─────────────────────────┘
                     ▼
  export ONNX ─► onnxslim ─► ONNX 规则改写 ─► pnnx ─► ncnn fp32 ─► INT8 ─► meta ─► 验证 ─► 部署包
```

从零跑通见 **[docs/QUICKSTART.md](docs/QUICKSTART.md)**（第一步安装，一直到第九步推理）。

## 目录

| 路径 | 内容 |
|---|---|
| `evp/` | Python 工具链：命令、数据、训练、部署、工作台 |
| `runtime/cpp/` | C++11 运行时，随部署包拷到板端（Apache-2.0） |
| `configs/project.example.yaml` | `evp init` 使用的项目模板 |
| `docs/` | 运行步骤、开发说明、部署、契约、板端 |
| `examples/visa_pcb/` | 公开数据集上的缺陷检测示例 |
| `examples/onnx_rewrite/` | 通用 ONNX 算子替换示例 |
| `scripts/` | 安装原生工具、换行与许可证检查 |
| `tests/` | 单元测试和带 `integration` 标记的真实转换测试 |

训练权重、数据集和工作目录不入库。

## 文档

- [docs/QUICKSTART.md](docs/QUICKSTART.md)：安装和运行，按步骤来
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)：开发环境、测试、设计约束
- [docs/DEPLOY_TOOLCHAIN.md](docs/DEPLOY_TOOLCHAIN.md)：转换、算子替换与后处理配套、INT8
- [docs/META_SPEC.md](docs/META_SPEC.md)：前后处理契约字段
- [docs/BOARD.md](docs/BOARD.md)：板端交叉编译与运行
- [docs/ADAPTERS.md](docs/ADAPTERS.md)：新增一类模型时要改哪些文件
- [examples/visa_pcb/README.md](examples/visa_pcb/README.md)：VisA PCB 缺陷检测示例

## 部署包

`evp deploy` 写到 `workspace/deploy/<name>/`（路径可在 `project.yaml` 里改）：

```text
<name>.fp32.param / .bin          ncnn fp32
<name>.int8.param / .bin / .table ncnn INT8（混合精度）与校准表
meta.json  model.meta  labels.txt 前后处理契约（同一对象生成）
report.json  report.md            验证结论
runtime/                          C++ 运行时源码
```

`report.md` 给出 fp32、int8 是否通过，以及推荐精度。INT8 不达标时推荐 fp32。

## 验证阶梯

| 级别 | 比较 | 门限（默认） |
|---|---|---|
| L1 | 框架 (torch/ORT) vs ncnn fp32，相同输入张量 | cos ≥ 0.999 |
| L2 | ncnn INT8 vs fp32（检测感知相似度） | ≥ 0.93 |
| L3 | ncnn fp32 + 配套后处理 vs Ultralytics `predict()` | 检测 F1 ≥ 0.9；分类：top-1 落在框架 top-5 内，且框架置信 ≥ 0.5 时 top-1 一致，比例 ≥ 0.9 |
| L3c | C++ 运行时 vs Python 参考实现 | fp32：输入张量逐位相同、检测结果一致；INT8 仅参考 |

没有可比内容的检查（例如验证图上什么都没检出）记为 **UNMEASURED**，不算通过。INT8 没有任何测量过的门限检查时报告为 **UNVERIFIED**，并推荐 fp32。换含目标的图片再验：`evp verify --bundle <包> --images <目录>`。

参考结果（320×320，COCO 校准图，开发时测得）：yolo11n / yolo26n / yolo11n-cls 的 fp32 转换余弦为 1.0，与 Ultralytics predict 的检测 F1 为 1.0；C++ fp32 前处理张量差为 0，框差 ≤ 0.0003 px。yolo11n 混合精度后 INT8 相似度约 0.98，检测 F1 约 0.92。YOLO26 端到端头对量化敏感时，报告会建议用 fp32 或 `deploy.e2e: false`。

## 许可证与参与

分层授权，详见 [NOTICE](NOTICE)：

- `runtime/cpp/`：**Apache-2.0**（[runtime/cpp/LICENSE](runtime/cpp/LICENSE)）
- 其余（Python 工具链，依赖 AGPL-3.0 的 Ultralytics）：**AGPL-3.0-only**（[LICENSE](LICENSE)）

第三方依赖：[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。知识产权声明：[docs/IP_STATEMENT.md](docs/IP_STATEMENT.md)。

参与贡献：[CONTRIBUTING.md](CONTRIBUTING.md)、[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)、[GOVERNANCE.md](GOVERNANCE.md)、[SECURITY.md](SECURITY.md)、[CHANGELOG.md](CHANGELOG.md)。
