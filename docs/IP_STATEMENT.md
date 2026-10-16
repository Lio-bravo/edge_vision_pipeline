# 知识产权与原创性声明

项目名称：edge_vision_pipeline（evp）——面向睿擎工业开发平台的工业视觉数据到部署全流程工具链

1. **原创性**：`evp/`、`runtime/cpp/`、`tests/`、`scripts/`、`configs/`、`docs/` 中的源代码与文档由本项目贡献者
   原创编写。仓库中没有复制或修改第三方项目的源代码。
2. **第三方组件**：所有第三方软件都以依赖形式使用（pip 安装、系统构建或独立进程调用），没有随仓库分发它们的源码或二进制。
   清单、版本和许可证见 [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md)，均遵循各自许可证使用。
3. **开源许可**：按目录分层授权，详见 [NOTICE](../NOTICE)：
   - `runtime/cpp/`：Apache License 2.0；
   - 其余部分：GNU AGPL v3.0 only（因为工具链依赖 AGPL-3.0 的 Ultralytics）。

   每个源文件都带 SPDX 许可证标识，可用 `python scripts/add_spdx.py --check` 校验。
4. **模型与数据**：仓库不包含预训练权重和第三方数据集。测试用的 Ultralytics 权重（AGPL-3.0）和 COCO 图片仅在本地使用。
   演示所用数据集的来源与许可证在 Demo 文档中单独注明。
5. **商标**：RT-Thread、睿擎、Ultralytics、YOLO、ncnn 等名称归各自权利人所有，本项目仅作兼容性说明使用。
6. **贡献约定**：外部贡献按 [CONTRIBUTING.md](../CONTRIBUTING.md) 以相同许可证（入站 = 出站）授权，贡献者需确认对所提交内容拥有相应权利。

声明人：edge_vision_pipeline 贡献者
日期：2026-10-16
