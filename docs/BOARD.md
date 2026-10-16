# 板端部署

部署包里的 `runtime/` 是自包含的 C++11 运行时：`evp_runtime`（静态库：meta 解析、前处理、后处理、模型封装）、
`evp_demo`（命令行 demo）、`evp_unit_tests`。依赖只有 ncnn。

## 1. 主机（Linux / WSL）编译与自测

```bash
bash scripts/build_runtime.sh              # 用 $EVP_TOOLS_HOME/ncnn-install，跑单测，最后一行输出 demo 路径
evp_demo <bundle> a.jpg b.ppm --precision int8 --threads 2 --bench 20
evp_demo <bundle> a.ppm --dump-input /tmp/in   # 导出前处理后的输入张量（一致性比对用）
```

输出为每张图一行 JSON（`detections` 或 `topk`）。

## 2. 交叉编译（ARM Linux，如 RK3506 跑 Linux 时）

1. 用目标工具链交叉编译 ncnn（`-DNCNN_VULKAN=OFF -DNCNN_SIMPLEOCV=ON -DNCNN_OPENMP=ON`，A7 建议 `-DNCNN_ARM82=OFF`）。
2. 编译运行时：

```bash
NCNN_DIR=/path/to/ncnn-arm/install TOOLCHAIN=/path/to/arm-toolchain.cmake BUILD_DIR=build-arm \
  bash scripts/build_runtime.sh
```

3. 把 `evp_demo`、部署包（param/bin/model.meta）拷到板上运行；用 `--bench` 测板端真实时延。

## 3. RT-Thread（睿擎派 RK3506J）注意事项（未在板上验证）

- 官方 AI 示例走 ncnn CPU 推理；把 `runtime/src/*.cpp` 和 `include/` 加入 RT-Thread 工程（SCons），
  与官方 ncnn 包一起编译即可，代码只用 C++11 标准库与 ncnn。
- 文件系统：`model.meta`、`.param`、`.bin` 需放在可读的文件系统（如 SD/eMMC 上的 elm-fat）；
  `evp::load_meta` 用标准 `std::ifstream` 读取。
- 图像来源：demo 读 PPM/JPG 只是参考；实际在相机回调里把**连续存储的 BGR** 帧交给
  `evp::preprocess(bgr, w, h, spec, out, geometry)`（行间有填充或是 YUV 时先转换成连续 BGR）。
- 内存：256MB 板上建议 320×320 以内输入、n 级模型、INT8 优先；线程数按核数（3）设置。
- `model.meta` 的数字按 "C" locale 解析（与进程 locale 无关）；整数越界、`conf/iou` 超出 [0,1]、
  `num_classes` 与标签数不符都会拒绝加载。输入图超过 `kMaxImageBytes`（1 GiB BGR）会被拒绝，避免 32 位平台上 `w*h*3` 溢出。
- `evp::load_ppm` 可直接用于无 JPEG 解码库的板子做离线自测。
- 这部分接口尚未在实际板子上编译运行，首次移植请先跑 `evp_unit_tests` 确认。
