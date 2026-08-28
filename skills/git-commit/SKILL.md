---
name: git-commit
description: 当需要检查 Git 状态、整理提交内容、编写规范 commit message、执行 commit 或 push 时使用。
---

# Git Commit Skill

使用这个技能时，先确认本轮要提交的功能边界，再检查工作区状态。

## 工作规则

- 只提交和当前功能直接相关的正式源码。
- 不提交 `.env`、虚拟环境、缓存、临时测试目录或本地演示产物。
- commit message 使用 `feat:`、`fix:`、`test:`、`docs:` 等清晰前缀。
- commit body 说明这一步新增了什么能力，以及为什么这样拆分。
- push 前先确认测试结果和暂存区内容。

## 输出要求

向用户说明本次提交包含哪些文件、通过了哪些验证，以及最新 commit id。
