# 通用 ONNX 算子替换示例

一个很小的图：uint8 NHWC → `Cast` → `Transpose` → `ReduceL2`。用来看 evp 怎样把这些算子换成板端能跑的形式。

它不是检测模型，也没有检测后处理。检测、分类各自的配套后处理见 [docs/DEPLOY_TOOLCHAIN.md](../../docs/DEPLOY_TOOLCHAIN.md) 的「配套」一节。

```bash
python examples/onnx_rewrite/build_demo_onnx.py
evp inspect-onnx examples/onnx_rewrite/cast_transpose_l2.onnx \
    --out examples/onnx_rewrite/prepared.onnx
```

`prepared.onnx` 是改写结果，可以删掉，需要时再生成。
