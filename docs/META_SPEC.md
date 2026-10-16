# 前后处理契约（meta）

`meta.json`（Python）与 `model.meta`（C++，扁平 `key=value`，UTF-8，LF）由同一个 `ModelMeta` 对象生成
（`evp/deploy/meta.py: save_meta`），不要手工只改其中一个。`schema_version=1`。

## 输入 `input.*`

| 键 | 含义 |
|---|---|
| `w`, `h` | 网络输入宽高 |
| `blob` | ncnn 输入 blob 名 |
| `color` | 送入网络的通道顺序 `RGB`/`BGR`（图像按 BGR 解码后转换） |
| `resize` | `letterbox`（等比缩放+居中填充，与 Ultralytics LetterBox 一致）/ `stretch` / `center_crop` |
| `pad_value` | letterbox 填充值（0..255，所有通道） |
| `mean`, `norm` | 每通道 `x' = (pixel - mean) * norm` |

letterbox 几何：`r = min(h/H, w/W)`，`new = max(1, round_half_even(W*r)), ...`，
`left = round(dw - 0.1)`, `top = round(dh - 0.1)`；坐标还原：`(x - left) / r` 并裁剪到原图。

## 输出 `output.<k>.*`

`name`（ncnn blob）、`shape`（ncnn 维度 c,h,w；1-D 为 w）、`stride`（检测头）。
检测头每个 stride 一个输出，通道 = `4*reg_max + num_classes`，前 `4*reg_max` 为框（reg_max>1 时为 DFL 分布，
YOLO26 为 1 即直接 ltrb 距离），后面为类别 logits（未过 sigmoid）。

## 后处理 `post.*`

| method | 用途 | 逻辑 |
|---|---|---|
| `nms` | YOLOv8/11、YOLO26(e2e=false) | 解码 → 分数 > conf → 按类偏移 NMS（IoU > iou 抑制）→ 前 max_det |
| `e2e_topk` | YOLO26 一对一头 | 全部锚点按最大类分数取 top-k → conf 过滤 → 前 max_det，无 NMS |
| `topk` | 分类 | （可选 softmax）→ 稳定排序取前 topk |
| `none` | raw | 不做后处理，直接给出输出张量 |

其它键：`conf, iou, max_det, agnostic(0/1), topk, softmax(0/1)`；`reg_max`、`num_classes`、
`labels.count` + `label.<k>`、`file.<precision>.param/bin`。

## 数值约定（保证 Python 与 C++ 逐位一致）

- 前处理用 ncnn `from_pixels_resize` + `substract_mean_normalize`，两侧调用相同。
- 几何计算用 `nearbyint`（四舍六入五成双）。
- 后处理用 double 计算，排序为稳定排序，遍历顺序与 Python 相同；float32 logits 与阈值比较时阈值转为 float。
