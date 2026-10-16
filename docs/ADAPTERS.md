# 新增模型族适配器（Adapter）指南

适配器 = “在 PyTorch 层把模型切成纯张量网络” + “把被切掉的部分写成配套的运行时后处理”。
当前已实现：`yolo`（Ultralytics anchor-free Detect：v5u/v8/v11/v12 带 DFL，YOLO26 `reg_max=1`，NMS / 端到端两种头）
和 `classify`（Ultralytics Classify）。Segment / Pose / OBB 尚未实现（`detect_family` 会明确报错）。

**不需要写适配器的情况**：通用 `.onnx` 若后处理可以截断（`deploy.onnx.cut_outputs`）且运行时只需输出原始张量，
用 `family: raw` 即可（见 DEPLOY_TOOLCHAIN.md §3）。只有需要“板端直接给出检测框/关键点/掩码”时才写适配器。

## 1. 涉及的位置（以新增 `pose` 为例）

| 层 | 文件 | 要做的事 |
|---|---|---|
| 族识别 | `evp/deploy/export.py::detect_family` | 按头类型**精确**匹配（`type(head) is X`，子类不能误判为父类） |
| 切头 | `evp/deploy/export.py::_patch_<family>_head` | 替换 `head.forward`，返回每个 stride 的原始张量元组；不含 decode/sigmoid/NMS |
| 导出 | `export_ultralytics` | 生成 `OutputSpec`（形状 = ncnn 的 `(c, h, w)`，去 batch）、`PostSpec`、`InputSpec`；导出前用 `torch.zeros` 前向自检形状 |
| 契约 | `evp/deploy/meta.py` | `FAMILIES`/`POST_METHODS` 增加取值；`ModelMeta.task`；`validate()` 增加该族不变量；如需新字段要同时写进 `to_flat`（生成 `model.meta`） |
| C++ 契约 | `runtime/cpp/src/meta.cpp` | 同样的取值与不变量（**必须与 Python validate 一致**）；新字段的解析 |
| Python 后处理 | `evp/deploy/postprocess.py` | 新的 `postprocess_<task>`，坐标回映射用 `to_source` 同款逻辑 |
| C++ 后处理 | `runtime/cpp/src/postprocess.cpp` + `include/evp/evp.h` | 与 Python **逐行等价**（候选遍历顺序、`score > conf` 严格比较、稳定排序、裁剪） |
| 运行时入口 | `evp/deploy/runner.py::NcnnPipeline.run`、`runtime/cpp/src/model.cpp`、`demo/evp_demo.cpp` | 按 family 分发 |
| 验证 | `evp/deploy/verify.py`（L3 与框架 `predict()` 对比）、`evp/deploy/cpp_check.py`（L3c） | 增加该任务的匹配指标；至少一个 **gate=True** 的 fp32 检查 |
| INT8 指标 | `evp/deploy/quant.py::output_similarity` | 若原始余弦会被某些通道主导（如 DFL 框分布），写任务感知相似度 |
| 预标注 | `evp/data/prelabel.py` | 仅检测类任务需要（LabelMe 形状映射） |

## 2. 切头的原则

1. **切在最后一个“规整卷积输出”处**：每个尺度一个 `[C, H, W]`，后面的 reshape/concat/decode 全部交给运行时。
   这样 ncnn 图里没有动态形状、Gather、TopK，INT8 也只作用于卷积。
2. 不在图里做 sigmoid / softmax（分类头例外：Ultralytics 导出模式本身输出概率，`softmax=false`）。
3. `head.forward` 用 `types.MethodType` 替换在 **deepcopy 后的模型**上，原 `yolo` 对象保持可用（L3 参考用）。
4. 端到端/双分支头（如 YOLO26 `one2one_*`）要在 `fuse()` **之前**决定保留哪一支（见 `head.end2end`）。
5. 导出用 `export_onnx`（TorchScript 导出，opset 17，`dynamo=False`）；不要换 dynamo 导出器。

## 3. meta 契约

- 所有前后处理参数只能来自 meta；运行时代码里不允许出现模型相关的常数。
- `OutputSpec.stride` 只对多尺度检测头有意义；`validate()` 会检查 `grid × stride == input`。
- 新增字段步骤：dataclass 字段 → `to_dict/from_dict`（默认值保证旧 meta 可读）→ `to_flat`（`model.meta` 文本）→
  C++ `parse_meta` → `docs/META_SPEC.md` → `tests/test_meta.py` 往返测试 + C++ `test_meta_validation`。
- 不兼容改动必须提升 `SCHEMA_VERSION`（Python 与 C++ 两边）。

## 4. 测试清单（缺一不可）

| 测试 | 位置 | 内容 |
|---|---|---|
| 后处理单测 | `tests/test_postprocess.py` | 手工构造头张量：正常、空输出、阈值边界（`== conf` 不保留）、并列分数顺序、越界裁剪 |
| meta 单测 | `tests/test_meta.py` | 合法/非法形状、取值、往返（json ↔ 对象 ↔ model.meta） |
| C++ 单测 | `runtime/cpp/tests/unit_tests.cpp` | 同一组构造张量，结果与 Python 一致；非法 meta 被拒 |
| 集成 | `tests/test_integration.py`（`@pytest.mark.integration`） | 真实权重 `evp deploy` 全流程：L1 cos ≥ 0.999、L3 与框架一致、L3c C++ = Python（fp32 严格） |
| CLI | `tests/test_cli.py` | `--set deploy.family=<新族>` 等配置校验路径 |

完成后运行：

```bash
python -m ruff check evp tests
python -m pytest -m "not integration"
python -m pytest -m integration
```

C++ 单测的编译方式见 [DEVELOPMENT.md](DEVELOPMENT.md)。

## 5. 常见坑

- **子类误判**：`Segment/Pose/OBB` 都继承 `Detect`，用 `isinstance` 会被当成检测模型，额外输出被静默忽略。
- **遍历顺序**：Python 与 C++ 都按 stride → 行 → 列 收集候选；顺序不同会让并列分数的 NMS 结果不同，L3c 失败。
- **阈值比较**：最终用 sigmoid 后的 `score > conf`（严格大于，与 Ultralytics 一致）；logit 预筛只是加速。
- **INT8 敏感层**：直接产生网络输出的层默认保持 fp32（`deploy.quant.keep_output_layers_fp32: true`），
  其余敏感层可用 `deploy.quant.fp32_layers` 手动指定；
  端到端头对量化更敏感，报告会建议 fp32。
