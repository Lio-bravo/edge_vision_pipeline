## 改动说明

<!-- 做了什么、为什么做；关联的 Issue：#xx -->

## 类型

- [ ] 缺陷修复
- [ ] 新功能
- [ ] 文档
- [ ] 重构 / 其他

## 自检清单

- [ ] `python scripts/normalize_eol.py`
- [ ] `python scripts/add_spdx.py --check`
- [ ] `python -m ruff check evp tests`
- [ ] `python -m pytest -m "not integration"`（贴结果）：
- [ ] 改了转换/量化/验证/C++ → `python -m pytest -m integration`（贴结果）：
- [ ] 改了 `runtime/cpp` → C++ 单测通过，且没有引入 GPL/AGPL 依赖
- [ ] 前后处理改动同时修改了 Python 与 C++，并补充了测试
- [ ] 新增依赖已更新 THIRD_PARTY_LICENSES.md
- [ ] 用户可见的改动已写入 CHANGELOG.md
- [ ] commit 已 `Signed-off-by`（DCO）
