# 开发说明

给改代码的人。使用步骤见 [QUICKSTART.md](QUICKSTART.md)。

## 1. 范围

一条工具链，两段工作接在一起：检测数据的采集、预标注、复检、入库、训练；以及把得到的权重或通用 ONNX 转成 ncnn。转换时，从图里替换掉的算子必须由该模型族自己的后处理接上，Python 与 C++ 共用 `meta`。细节和示例见 [DEPLOY_TOOLCHAIN.md](DEPLOY_TOOLCHAIN.md)。

目标机器是没有 NPU 的 ARM CPU，推理框架是 ncnn。睿擎派 RK3506J（3×Cortex-A7，256 MB）是当前对照的板型，不是唯一支持的板。

传统视觉算子（OpenCV 一类）替换不在范围内。

数据闭环目前是目标检测。部署侧另外支持 Ultralytics 分类和通用 ONNX（`family: classify` / `raw`）。Segment、Pose、OBB 会明确报错。

## 2. 目录

```text
edge_vision_pipeline/
  evp/                 命令行、配置、数据、训练、部署、工作台
  runtime/cpp/         C++11 运行时（Apache-2.0）：meta、前后处理、demo、单测
  configs/             project.example.yaml，evp init 复制它
  docs/                使用步骤、部署、契约、板端、适配器
  examples/visa_pcb/   工业缺陷示例（数据不入库）
  scripts/             工具安装、换行、SPDX、VisA 数据准备
  tests/               pytest；integration 标记的用例会真的转换模型
```

权重、数据集、`workspace/`、`scratch/`、`models/` 不入库。克隆之后要自己准备预训练权重才能跑集成测试。

## 3. 环境

| 项 | 约定 |
|---|---|
| Python | ≥ 3.10，建议仓库内 `.venv` |
| 安装 | `pip install -r requirements.txt` 然后 `pip install -e ".[dev,webui]"` |
| 原生工具 | Linux：`bash scripts/setup_tools_linux.sh`；Windows：`powershell -File scripts/setup_tools_wsl.ps1` |
| 工具目录 | 默认 `~/evp_deps`，可用 `EVP_TOOLS_HOME` 或 `EVP_TOOLS_HOME_WSL` 覆盖 |
| 集成测试权重 | `models/pretrained/{yolo11n,yolo26n,yolo11n-cls}.pt`（自行下载，不入库） |
| 集成测试图片 | 环境变量 `EVP_TEST_IMAGES` 指向一个图片目录；未设置时用仓库自带的少量样例图 |

可选环境变量：`EVP_PNNX`、`EVP_NCNN_TOOLS_DIR`、`EVP_WSL_DISTRO`、`EVP_DEMO`。

`scripts/setup_tools_linux.sh` 默认使用清华 PyPI 镜像，可用 `PIP_INDEX` 改掉。

## 4. 测试

```bash
python scripts/normalize_eol.py          # 源码换行 LF，.ps1 为 CRLF
python scripts/add_spdx.py --check
python -m ruff check evp tests scripts
python -m pytest -m "not integration" -q
python -m pytest -m integration -q       # 改了转换、量化、验证或 C++ 时
bash scripts/build_runtime.sh            # 改了 runtime/cpp 时；Windows 在 WSL 里跑
```

## 5. 设计决策

1. **切头，不把后处理放进图里。** DFL、sigmoid、NMS 在运行时做。ncnn 图里没有 TopK、Gather 和动态形状，INT8 只作用于卷积。Python 和 C++ 各写一份后处理，由 L3c 保证 fp32 下一致。
2. **meta 是唯一契约。** `meta.json`（Python）和 `model.meta`（C++ 扁平 key=value，板端不需要 JSON 库）由同一个对象生成。
3. **ONNX 规则只做可证明等价的改写。** 每条规则都有 ORT 前后等价测试。逐通道常量的广播方向不确定时不折叠。
4. **校准张量由部署前处理生成**，保证校准分布和板端输入一致。
5. **混合精度靠编辑校准表。** 删掉某层的权重和激活 scale 后，ncnn2int8 会让该层保持 fp32。
6. **INT8 用检测感知相似度。** 原始余弦会被框分布主导，看不出类别分数被量化破坏。
7. **没有测到的检查不算通过。** 两侧都没检出目标时记为 UNMEASURED。INT8 没有测量过的门限检查时记为 UNVERIFIED，并推荐 fp32。
8. **采集默认不做近重复过滤。** 产线上良品和缺陷外观几乎一样，dHash 会把缺陷图删掉。像素完全相同的图片总会被过滤。
9. **工作台不缓存项目。** 每次操作都重新读 `project.yaml`，可以和命令行混用。
10. **verify 和 eval 分开。** verify 证明转换等价；eval 对真值报 mAP。

## 6. 已知限制

1. 还没有在 RK3506J 上交叉编译和跑过。交叉编译的用法写在 [BOARD.md](BOARD.md)，脚本参数是 `TOOLCHAIN` 和 `NCNN_DIR`。
2. Segment / Pose / OBB 未实现。通用 ONNX 检测模型只能走 `family: raw`（输出原始张量，不在板端解码成框）。
3. 分类可以部署和验证，但不能走采集 → LabelMe → 入库这条检测数据流。`evp eval` 只做检测 mAP。
4. C++ demo 读图以 PPM 为主。产品集成时由调用方传入连续 BGR（`Model::detect`）。
5. 工作台的后台任务一次只跑一个。浏览器关掉后，训练或部署仍会在进程里继续。
6. YOLO26 端到端头对 INT8 敏感时，报告会建议 fp32 或 `deploy.e2e: false`。

## 7. 实现时要注意

- Windows 上的 pnnx.exe 会崩溃，工具链应走 WSL。调用形式是 `wsl -d <distro> --cd <dir> --exec ...`，参数不要经过 shell。
- `ncnn2table method=eq` 在上游有段错误，只作可选项。
- ncnn2table 用 `%255s` 读列表：路径不要有空格，列表用相对工作目录的短路径。
- 有系统代理时，Gradio 访问 `127.0.0.1` 可能被转走。`evp webui` 会把 localhost 加进 `NO_PROXY`。自己调用 `build_app()` 时先调 `bypass_proxy_for_localhost()`。
- Gradio 动态下拉框需要 `allow_custom_value=True`。
- 改前后处理时 Python（`evp/deploy/postprocess.py`）和 C++（`runtime/cpp/src/postprocess.cpp`）一起改，并补测试。
- 改 meta 字段：dataclass → to_dict/from_dict → to_flat → C++ `parse_meta` → [META_SPEC.md](META_SPEC.md) → 两侧测试。不兼容变更要提升 `schema_version`。
- 新模型族按 [ADAPTERS.md](ADAPTERS.md)。
