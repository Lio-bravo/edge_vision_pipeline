# 第三方组件与许可证清单

本项目**不复制、不内嵌**任何第三方源码或二进制。下列组件都是运行时依赖（通过 pip、系统安装或独立进程调用），
版本为开发环境的实测版本（2026-10-05）。许可证来自各包元数据或上游仓库。

## Python 依赖（`requirements.txt` / `pyproject.toml`）

| 组件 | 版本 | 许可证 | 用途 | 关系 |
|---|---|---|---|---|
| numpy | 2.5.2 | BSD-3-Clause（含 0BSD/MIT/Zlib/CC0 部分） | 数组计算 | import |
| PyYAML | 6.0.3 | MIT | 配置文件 | import |
| Pillow | 12.3.0 | MIT-CMU | 读图、EXIF | import |
| opencv-python | 5.0.0.93 | Apache-2.0 | 图像处理 | import |
| onnx | 1.23.1 | Apache-2.0 | ONNX 图改写 | import |
| onnxruntime | 1.30.0 | MIT | 参考推理、等价验证 | import |
| onnxslim | 0.1.97 | MIT | ONNX 简化 | import |
| ncnn（Python 绑定） | 1.0.20260526 | BSD-3-Clause | 推理、验证 | import |
| torch | 2.14.1 | BSD-3-Clause 等（Apache-2.0、BSD-2-Clause、BSL-1.0、MIT 部分） | 导出、训练 | import（可选依赖 `ultralytics`） |
| **ultralytics** | 8.4.173 | **AGPL-3.0** | 训练、导出、预标注 | import（可选依赖 `ultralytics`）→ Python 工具链因此采用 AGPL-3.0 |
| gradio | 6.29.1 | Apache-2.0 | Web 工作台（`evp webui`） | import（可选依赖 `webui`） |
| gradio_image_annotation | 0.5.6 | MIT | 工作台中的画框标注组件 | import（可选依赖 `webui`） |
| gradio 的传递依赖 | 见右 | fastapi/pydantic/typer/rich/orjson 等：MIT；starlette/uvicorn/httpx2/pandas：BSD-3-Clause；huggingface_hub：Apache-2.0；orjson：MPL-2.0 AND (Apache-2.0 OR MIT) | Web 服务 | 间接依赖；MPL-2.0 为文件级弱 copyleft，未修改其源码，与 AGPL-3.0 兼容 |
| pytest | 9.1.1 | MIT | 测试 | 开发依赖 |
| ruff | 0.16.10 | MIT | 代码规范 | 开发依赖 |

## 原生工具（独立进程调用，不链接）

| 组件 | 版本 | 许可证 | 用途 |
|---|---|---|---|
| pnnx | 20260526 | BSD-3-Clause | ONNX → ncnn 转换 |
| ncnn 工具（ncnn2table、ncnn2int8） | master 源码构建 | BSD-3-Clause | INT8 校准与量化 |
| LabelMe（可选） | 用户自行安装 | GPL-3.0 | 人工复检；`evp review` 只启动其可执行文件，不 import |

## C++ 运行时（`runtime/cpp`，Apache-2.0）

| 组件 | 许可证 | 关系 |
|---|---|---|
| ncnn | BSD-3-Clause | 编译链接（与 Apache-2.0 兼容） |
| C++11 标准库 | 随编译器 | — |

运行时不依赖 Ultralytics、OpenCV 或任何 GPL/AGPL 组件。

## 模型与数据（不随仓库分发）

| 内容 | 来源 / 许可证 | 说明 |
|---|---|---|
| `models/pretrained/*.pt`（yolo11n、yolo26n、yolo11n-cls） | Ultralytics，AGPL-3.0 | 仅用于测试，已在 `.gitignore` 中排除，需要时自行下载 |
| COCO val2017 图片 | COCO Consortium；标注 CC BY 4.0，图片版权归 Flickr 原作者 | 仅本地用作校准/验证图片，不入库 |
| VisA 数据集（pcb1–pcb3） | Amazon，**CC BY 4.0**；Zou et al., ECCV 2022，<https://github.com/amazon-science/spot-diff> | 工业缺陷 Demo。`scripts/prepare_visa.py` 把像素掩码转为检测框（属于“改动”，生成目录内附 `VISA_ATTRIBUTION.txt` 署名与改动说明）；数据不入库，按 `examples/visa_pcb/README.md` 自行下载 |
| 用户训练得到的模型 | 由用户的数据与基础权重决定 | 若基于 Ultralytics 预训练权重微调，请自行评估 AGPL-3.0 对模型分发的影响 |

## 维护方式

- 新增依赖时同步更新本文件与 `requirements.txt`，并确认许可证与 AGPL-3.0（工具链）或 Apache-2.0（运行时）兼容。
- 运行时（`runtime/cpp`）**禁止**引入 GPL/AGPL 依赖。
