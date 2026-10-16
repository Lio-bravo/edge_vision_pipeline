# 安全策略

## 支持的版本

只有 `master` 分支的最新版本会获得安全修复。

## 报告漏洞

请**不要**为安全问题公开提 Issue。通过 Gitee 私信联系 [GOVERNANCE.md](GOVERNANCE.md) 中列出的维护者，内容包括：

- 受影响的组件，例如 `runtime/cpp` 的 meta/PPM 解析，或 Python 工具链的某条命令；
- 复现步骤或 PoC 文件；
- 可能的影响。

我们会在 3 个工作日内确认收到，修复后在 CHANGELOG 中致谢（除非你希望匿名）。

## 已有的防护（便于评估影响）

- C++ 运行时解析 `model.meta` 时检查所有字段：数值解析不依赖 locale，整数有范围检查，校验规则与 Python 一致。
- PPM 读取使用 64 位尺寸运算，并设单图上限 `kMaxImageBytes`（1 GiB），防止溢出和超大分配。
- 外部工具（pnnx、ncnn2table 等）以参数列表方式启动，不经过 shell 展开；WSL 调用使用 `wsl --exec`。
