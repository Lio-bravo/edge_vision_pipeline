# 贡献指南

欢迎通过 Issue 和 Pull Request（Gitee 上叫 PR）参与。使用见 [docs/QUICKSTART.md](docs/QUICKSTART.md)，改代码前再读 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)。

## 许可证与授权

- `runtime/cpp/` 下的贡献按 **Apache-2.0** 授权，其余部分按 **AGPL-3.0-only** 授权（见 [NOTICE](NOTICE)）。
- 提交 PR 即表示你确认：有权提交这些内容，并同意按上述许可证授权（入站 = 出站）。
- 每个 commit 请加 `Signed-off-by`（`git commit -s`），表示同意 [DCO 1.1](https://developercertificate.org/)。
- `runtime/cpp/` **不得**引入 GPL/AGPL 依赖；新增任何依赖都要同步更新 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)。

## 提问与报告问题

- Bug：用 Gitee Issue 的“缺陷报告”模板，附上 `evp tools` 输出、配置片段、完整报错（包括 `hint:` 行）和复现步骤。
- 新需求或新模型族：先开 Issue 讨论方案，新增模型族按 [docs/ADAPTERS.md](docs/ADAPTERS.md) 的清单实现。
- 安全问题：不要公开提 Issue，见 [SECURITY.md](SECURITY.md)。

## 开发环境

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux:   source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[dev,webui]"
```

Windows 上 pnnx / ncnn 工具装在 WSL：`powershell -File scripts/setup_tools_wsl.ps1`。Linux：`bash scripts/setup_tools_linux.sh`。`ruff` 需要单独安装（`python -m pip install ruff`），它不在运行依赖里。

## 提交前必须通过

```powershell
python scripts/normalize_eol.py               # 换行统一为 LF（.ps1 为 CRLF）
python scripts/add_spdx.py --check            # 每个源文件都有 SPDX 许可证头
python -m ruff check evp tests                # 代码规范
python -m pytest -m "not integration"         # 快速单测
python -m pytest -m integration               # 改了转换/量化/验证/C++ 时必须跑
```

改了 `runtime/cpp` 时，还要重新编译并运行 C++ 单测：`bash scripts/build_runtime.sh`。

## 代码约定

- Python 3.10+，UTF-8 无 BOM，文件开头为 shebang + coding cookie + SPDX 头；ruff 行宽 120。
- 错误统一抛 `evp.errors` 中的子类，并带可操作的 `hint`；不要静默吞掉异常。
- 前后处理改动必须**同时修改 Python 与 C++**，并补充两侧测试（L3c 要求 fp32 下结果完全一致）。
- meta 字段变更按 [docs/ADAPTERS.md](docs/ADAPTERS.md) §3 的步骤执行；不兼容变更必须提升 `SCHEMA_VERSION`。
- 新功能要有正常、边界和失败路径的测试；修 bug 时先写一个能复现它的测试。
- 注释只写代码本身表达不了的约束；公共函数用一段 docstring 说明契约。

## PR 流程

1. Fork 后从 `master` 拉出分支：`feat/<主题>`、`fix/<主题>` 或 `docs/<主题>`。
2. 一个 PR 只做一件事；按 PR 模板填写改动说明和测试结果。
3. 至少一位维护者评审通过、“提交前必须通过”中的检查全部通过（在 PR 描述里贴结果）后合并（见 [GOVERNANCE.md](GOVERNANCE.md)）。
4. 用户可见的改动写入 [CHANGELOG.md](CHANGELOG.md) 的“未发布”一节。
