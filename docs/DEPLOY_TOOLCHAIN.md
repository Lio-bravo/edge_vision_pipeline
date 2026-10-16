# 部署工具链

`evp deploy` 把模型转成 ncnn，并保证**前后处理与模型配套**。核心原则：

1. 网络里只保留 ncnn 能高效执行的“纯张量计算”；前处理（缩放、归一化、排布）和后处理（解码、NMS、top-k）
   移到运行时，由 `meta` 描述，由 Python 参考实现和 C++ 运行时**同一份逻辑**执行。
2. 每一步改写都可验证：ONNX 规则用 onnxruntime 做前后等价测试；转换后 ncnn vs 框架逐输出比较；
   最终在真实图片上和框架 `predict()` 比检测结果。

## 1. 流程

| 步骤 | 实现 | 产物（`deploy_out/_work/<name>/`） |
|---|---|---|
| 导出 | `evp/deploy/export.py`：Ultralytics 模型改成输出**原始头张量**（每个 stride 一个 `[4*reg_max+nc, H, W]`），`torch.onnx.export(opset 17)` | `model.onnx` |
| 简化 | onnxslim（失败则跳过并告警） | |
| 规则改写 | `evp/deploy/onnx_rules.py` | `model.prepared.onnx` |
| 转 ncnn | pnnx（固定输入尺寸、`fp16=0`） | `model.ncnn.param/.bin`, `pnnx.log` |
| 数值检查 | 同一随机输入 ORT vs ncnn，每个输出 cos ≥ 0.999 | |
| INT8 | `evp/deploy/quant.py` | `quant/` |
| meta/验证/打包 | `pipeline.py`、`verify.py`、`cpp_check.py` | 部署包 |

## 2. ONNX 算子改写规则

默认顺序：`strip_input_cast → extract_input_normalization → strip_input_transpose →
extract_input_normalization → einsum_to_matmul → reduce_l2_decompose`（归一化可能在 Transpose 前(NHWC)或后(NCHW)，所以两边各跑一次，结果按顺序合成）。

| 规则 | 作用 | 配套的 meta 变化 |
|---|---|---|
| `strip_input_cast` | 去掉输入处的 `Cast(uint8→float)` | 输入改为 float |
| `strip_input_transpose` | 去掉输入处的 `Transpose(0,3,1,2)`（NHWC→NCHW） | 输入改为 CHW（ncnn 原生） |
| `extract_input_normalization` | 把输入后的 `Sub/Div/Mul/Add` 常量链（标量或逐通道）从图中删除 | 折算进 `input.mean / input.norm` |
| `einsum_to_matmul` | 两操作数、等价于（批量）矩阵乘的 Einsum → `MatMul`（必要时加 Transpose），字母命名任意，如 `bhid,bhjd->bhij` | 无 |
| `reduce_l2_decompose` | `ReduceL2` → `Sqrt(ReduceSumSquare)`，兼容 opset 13/18 写法 | 无 |
| 截断 `cut_outputs` | 在给定张量处截断，删除其后的后处理子图和无用权重 | 后处理由运行时完成 |

逐通道常量必须能确定按通道广播：NCHW 输入要求 `[C,1,1]`/`[1,C,1,1]`；NHWC 要求末维为 C；
形状不明确（例如 NCHW 上的 1-D `[C]`，实际沿 W 广播）时**不折叠**，保证不改变数值。除数为 0 也不折叠。

### 风险算子（ncnn 不支持或 pnnx 无法降级）

`TopK、NonMaxSuppression、NonZero、Unique、Compress`（后处理）、`ScatterND/ScatterElements/GatherND/GatherElements`、
`Loop/If`、`Range/ConstantOfShape/Shape`（动态形状）、`RoiAlign`。

剩余风险算子时转换会报错 `UnsupportedOpError`，并给出**建议截断点**（风险算子之前最近的张量）：

```powershell
evp inspect-onnx model.onnx                          # 算子清单 / 风险算子 / 建议截断点
evp inspect-onnx model.onnx --out p.onnx --cut auto  # 预览改写结果与 meta 变化
```

然后在配置里写 `deploy.onnx.cut_outputs: [张量名]`（或 `auto`），被截掉的部分由运行时后处理实现。

## 配套：算子替换和后处理是成套的

图里删掉什么，运行时就必须用对应的代码补上。`meta.json` / `model.meta` 记录这一套选择（`family`、`post.method`、`reg_max`、每个输出的形状和 stride、输入的 resize / mean / norm）。Python（`evp/deploy/postprocess.py`）和 C++（`runtime/cpp/src/postprocess.cpp`）读的是同一份契约，fp32 下结果要一致。

不能把 A 模型的后处理接到 B 模型的 ncnn 上。检测头的通道布局、要不要 NMS、分类要不要再做 softmax，都写在这份 meta 里。

| 模型 | 图里留下 / 会替换的算子 | 配套后处理 | `post.method` | 代码 |
|---|---|---|---|---|
| YOLO 检测（v5u / v8 / v11 / v12，`reg_max=16`） | 每个 stride 一个原始头 `[4*reg_max+nc, H, W]`。DFL、sigmoid、NMS 不在图里。通用规则（Cast、Transpose、归一化、Einsum、ReduceL2）碰到才会改 | DFL 解码，sigmoid，NMS，坐标映回原图 | `nms` | `postprocess_detect` |
| YOLO26，`deploy.e2e: false` | 保留 one-to-many 头，通常 `reg_max=1`（没有 DFL 分布，框通道就是距离） | 同上，仍做 NMS | `nms` | 同上，按 `reg_max` 分支 |
| YOLO26，`deploy.e2e: true` | 只留 one-to-one 头 | 按分数取 top-k，**不做 NMS** | `e2e_topk` | `postprocess_detect` 里的 `e2e_topk` |
| Ultralytics 分类 | softmax 留在图里（`post.softmax=false`）。输入几何是中心裁剪，不是 letterbox | 取 top-k | `topk` | `postprocess_classify` |
| 通用 ONNX，`family: raw` | 上表那些规则，外加 `cut_outputs` 截掉后处理子图 | 不解码，调用方拿原始张量 | `none` | 无任务后处理 |
| 通用 ONNX，`family: classify` | 同上；单输出。若输出还不是概率，meta 里 `post.softmax=true`，由运行时做 softmax | top-k | `topk` | `postprocess_classify` |

检测和分类的前处理也不混用：检测默认 `letterbox`（灰边 114），分类默认 `center_crop`。这两项同样写在 meta 里，校准和板端推理都走这一套。

两类 320 输入的检测契约示意（`nc=2`，stride 8/16/32）：

```text
# v8/v11：4*16+2 = 66 通道，要做 DFL 和 NMS
family=yolo
reg_max=16
post.method=nms
input.resize=letterbox
output.0.shape=66,40,40
output.0.stride=8

# YOLO26 端到端：4*1+2 = 6 通道，只做 top-k
family=yolo
reg_max=1
post.method=e2e_topk
```

分类契约示意：

```text
family=classify
post.method=topk
post.softmax=0
input.resize=center_crop
output.0.shape=<类别数>
```

Segment / Pose / OBB 还没有配套后处理，导出时会直接报错，避免用检测后处理去解它们的头。

### 可运行示例：通用 ONNX 的算子替换

仓库里的 `examples/onnx_rewrite/` 是一个很小的通用图：uint8 的 NHWC 输入，接着 `Cast`、`Transpose(0,3,1,2)`、`ReduceL2`。它不代表某个 YOLO，用来看规则实际改了什么。

```bash
python examples/onnx_rewrite/build_demo_onnx.py
evp inspect-onnx examples/onnx_rewrite/cast_transpose_l2.onnx \
    --out examples/onnx_rewrite/prepared.onnx
```

改写结果：

```text
before: Cast x1, ReduceL2 x1, Transpose x1
after:  ReduceSumSquare x1, Sqrt x1
strip_input_cast:       输入改为 float32，去掉开头的 Cast
strip_input_transpose:  输入改为 NCHW，去掉开头的 Transpose
reduce_l2_decompose:    ReduceL2 -> Sqrt(ReduceSumSquare)
```

输入形状从 `[1, 8, 8, 3]` 变成 `[1, 3, 8, 8]`。这三处改写和原图等价：板端只喂浮点、通道在前的张量，`ReduceL2` 换成 ncnn 更常见的算子。若这张图走 `family: raw`，`post.method` 为 `none`，运行时不做检测解码。若要在板端直接出类别，应设 `family: classify`，后处理才是 `postprocess_classify`。

YOLO 的 `.pt` 在导出时已经把 DFL 和 NMS 切掉，所以同一次 `evp deploy` 的报告里经常是 `rules applied: none`。那不是没做配套，配套发生在切头：图只出原始头，`post.method=nms` 或 `e2e_topk` 指明用哪一段后处理。通用 ONNX 没有这份切头信息，才靠上面的规则和 `cut_outputs`。

## 3. 接入通用 ONNX（family: classify / raw）

```yaml
deploy:
  weights: models/my_cls.onnx
  family: classify          # classify：单输出 top-k；raw：只做前处理 + 输出原始张量
  imgsz: [224, 224]         # 模型输入的 [h, w]（NHWC 输入会自动对应）
  labels: [ok, ng]
  input: {resize: stretch, mean: null, norm: null}
```

`deploy.input.mean/norm` **描述原始 ONNX 的输入**（像素 → 模型输入）：

- 图内含归一化且未填写：默认原始输入是 0..255 像素，图内归一化被提取到 meta。
- 图内无归一化且未填写：默认 `像素/255`，报告里会提示确认。
- 填写后：与提取出的图内归一化按顺序合成：`mean = m0 + m1/n0, norm = n0·n1`。

分类模型的输出是否已是概率会自动探测（非负且和为 1 时 `softmax=false`）。
YOLO 检测模型必须从 `.pt` 部署（`.pt` 适配器才知道头的切分位置与 reg_max / stride）。

## 4. INT8 量化

- 校准张量由**部署前处理本身**生成（letterbox/归一化与板端完全一致），写成 `.npy` 交给 `ncnn2table type=1`。
- `method: auto`：kl 与 aciq 都跑，取全 INT8 相似度更高的（yolo11n 上 aciq 0.884 vs kl 0.702）。
  `eq` 在 ncnn 上游有段错误，仅保留为可选项。
- 相似度指标（检测模型）：类别分数在任一侧 > 0.02 的位置比 sigmoid 分数余弦；框通道只在 fp32 认为有目标的格子比较；
  两者取最小。原始张量余弦会被 DFL 框分布主导，掩盖类别分数被量化破坏的问题。
- 混合精度：默认输出头卷积保持 fp32；若仍低于 `target_cos`，做逐层留一敏感度分析，按收益贪心把层改回 fp32，
  直到达标或达到 `max_fp32_layers`。实现方式是从校准表删除该层的激活与权重条目（ncnn2int8 跳过没有 scale 的层）。
- 实测：yolo11n 0.884 → 0.983；yolo26n(NMS 头) 0.882 → 0.973；yolo26n 端到端头 0.843 → 0.931（敏感）。

## 5. 工具定位（Windows / WSL）

查找顺序：`EVP_PNNX` / `deploy.tools_dir` / `EVP_NCNN_TOOLS_DIR` → pip 安装的 pnnx → PATH → `~/evp_deps` →
（Windows）WSL 内 `$HOME/evp_deps`。每个候选都会做启动健康检查，崩溃（负返回码、`0xC000xxxx`、134/139）则换下一个。
WSL 调用一律用 `wsl -d <distro> --cd <dir> --exec ...`，参数不经过 shell 展开；路径自动 `E:\x → /mnt/e/x`。

## 6. 常见错误

| 报错 | 原因 / 处理 |
|---|---|
| `UnsupportedOpError: TopK ...` | 图里有后处理算子：按提示设置 `deploy.onnx.cut_outputs` |
| `input dim i: model has fixed ...` | `deploy.imgsz` 与模型固定输入不一致：改成模型的 `[h, w]` |
| `model input has dynamic dims` | 动态输入且未给尺寸：设置 `deploy.imgsz` |
| `ncnn2table failed (method=eq)` | 上游 bug：用 `auto`/`aciq`/`kl` |
| `pnnx not available` | 运行 `scripts/setup_tools_linux.sh`（Windows 用 `setup_tools_wsl.ps1`），或设 `EVP_PNNX` |
| 报告 `int8: FAIL` | 增加现场校准图（64~200 张）、放宽 `max_fp32_layers`，或交付 fp32 |
| 分类严格 top-1 少量不一致（info） | Ultralytics 分类预处理用抗锯齿缩放，ncnn 双线性不抗锯齿；低置信图片 top-1/top-2 可能互换，属预期。门限检查的是“top-1 在框架 top-5 内”和“框架置信 ≥ 0.5 时 top-1 一致” |
