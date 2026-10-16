# 运行步骤

这个工具先把检测数据做成权重（采集、预标注、复检、入库、训练），再把权重或通用 ONNX 转成 ncnn。
转换时，算子替换和后处理是按模型配套的，说明和示例见 [DEPLOY_TOOLCHAIN.md](DEPLOY_TOOLCHAIN.md) 的「配套」一节。

下面从安装走到一份部署包。数据阶段按**目标检测**来做。分类模型和通用 ONNX 可以从「第八步」直接部署，不必走标注。

命令在项目目录里执行，且已经激活虚拟环境。配置文件是 `project.yaml`，相对路径以它所在目录为基准。

## 第一步：安装环境

需要 Python 3.10 及以上。部署还需要本机的 pnnx、ncnn2table、ncnn2int8：Linux 直接装，Windows 装在 WSL 里（Windows 版 pnnx 会崩溃）。

```bash
git clone <仓库地址>
cd edge_vision_pipeline
python -m venv .venv
```

Windows（PowerShell）：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e ".[webui]"
powershell -File scripts\setup_tools_wsl.ps1
```

Linux：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[webui]"
bash scripts/setup_tools_linux.sh
```

只要命令行、不用 Web 工作台时，把 `".[webui]"` 换成 `.`。工具装到 `~/evp_deps`，可用环境变量 `EVP_TOOLS_HOME` 改位置。

检查：

```bash
evp tools
```

每一行应显示实际用到的程序路径（本机或 WSL）。

## 第二步：创建项目

```bash
evp init my_line
cd my_line
```

会生成 `project.yaml` 和 `workspace/`。打开 `project.yaml`，把 `classes` 改成自己的类别，**顺序就是 YOLO 的类别 id，训练开始后不要再改**。

## 第三步：采集图片

任选一种，图片最终都在 `workspace/raw_images`：

```bash
evp collect -c project.yaml --folder ./shots
evp collect -c project.yaml --video line.mp4
evp collect -c project.yaml --camera 0
```

`--camera` 可以是本机摄像头序号，也可以是 RTSP 地址。完全相同的图片会被丢掉。产线上良品和缺陷外观很像，默认**不做**近重复过滤；视频或相机画面变化大时再加 `--dedup 4`。

也可以不跑命令，直接把图片拷进 `workspace/raw_images`。

## 第四步：预标注

`paths.prelabel_model` 指向一个检测权重（`.pt`）。模型里的类别名要和 `classes` 对得上，对不上的框不会写进标注。

```bash
evp prelabel -c project.yaml
```

还没有可用模型时，跳过这一步，在下一步的工作台里做空白标注。已经有部署包时，也可以：

```bash
evp prelabel -c project.yaml --bundle workspace/deploy/best
```

## 第五步：人工复检

```bash
evp webui -c project.yaml
```

浏览器打开后使用「预标注与审核」：改框、改类别、逐张保存。没有模型时用「空白标注」。

只用 LabelMe 时：`python -m pip install labelme`，然后 `evp review -c project.yaml`。

## 第六步：入库

```bash
evp ingest -c project.yaml
```

把复检过的 LabelMe json 转成 YOLO 标签，划出 train/val，合并进 `workspace/corpus`。成功后默认清空中转目录。旧的 val 不会被重新划分。未复检的图片会被拦住。

## 第七步：训练

```bash
evp train -c project.yaml
```

使用 Ultralytics，权重默认 `yolo11n.pt`（首次运行会下载）。`train.device: auto` 在有 CUDA 时用 GPU，否则用 CPU。CPU 训练很慢，正式训练请用 GPU。最佳权重要么在 `workspace/runs/` 下。

## 第八步：部署

```bash
evp deploy -c project.yaml
```

默认拿最新的 `best.pt`，做成 ncnn fp32 和 INT8，并写出前后处理契约和验证报告：

```text
workspace/deploy/<name>/
  report.md          先看这个：fp32 / int8 是否通过，推荐哪一种精度
  meta.json          Python 用的前后处理契约
  model.meta         C++ 用的同一份契约
  *.fp32.param/bin
  *.int8.param/bin
  runtime/           随包附带的 C++ 运行时源码
```

INT8 不达标时报告会推荐 fp32，不会把没验证过的量化模型当成通过。

已有 `.pt` 或 `.onnx`、不想走前面的数据步骤时，在 `project.yaml` 里设置 `deploy.weights`，然后直接运行这一条。通用 ONNX 把 `deploy.family` 设为 `raw` 或 `classify`。

## 第九步：推理、评估、测速

```bash
evp infer --bundle workspace/deploy/best img.jpg --save-vis out/
evp eval  -c project.yaml --bundle workspace/deploy/best --weights workspace/runs/train/weights/best.pt
evp bench --bundle workspace/deploy/best --images workspace/corpus/images/val
```

`eval` 用同一套指标比较 PyTorch、ncnn fp32、ncnn int8。`bench` 是当前这台电脑上 Python 运行时的延迟，不是板子上的延迟。板端编译和运行见 [BOARD.md](BOARD.md)。

退出码：0 成功；2 配置、数据或工具错误（看提示里的 `hint`）；3 部署验证未通过。

## 用工作台一次做完

`evp webui -c project.yaml` 的五个页签对应上面的采集、审核、入库与训练、部署、推理与测速，读写的是同一份 `project.yaml`，可以和命令行混用。

一条命令跑完「已有复检结果 → 训练 → 部署」：

```bash
evp all -c project.yaml
```

公开数据集上的完整例子：[../examples/visa_pcb/README.md](../examples/visa_pcb/README.md)。
