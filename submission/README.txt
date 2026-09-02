软件工程专业推免项目：DataBaseAgent（DBAgent）
仓库地址：https://github.com/huaijiao666/DBAgent

DBAgent 是从零实现的本地编程智能体，DB 指 Database。它面向真实代码仓库：模型以原生 function calling 请求工具；本地 Python runtime 自己管理上下文、计划、工具分发、文件修改、命令执行、验证、会话和 trace。模型口头“完成”不等于成功；只有当前代码的 pytest/编译/lint 等确定性证据通过，状态才是 VERIFIED。

特色功能：在用户选择的 workspace 内安全浏览、搜索和读取文件；针对 Python 用 AST 建立 repo map 与 symbol index；维护可见的任务目标、成功标准和步骤状态；以多文件原子 patch 修改代码；运行带 cwd、timeout、输出上限和环境清理的本地命令；检测重复失败并要求重新检查；保存脱敏 JSONL trace、session 和验证结果。支持 CLI、终端 TUI 与仅监听 127.0.0.1 的浏览器工作台。

运行：Python 3.11+。执行 python -m pip install -e ".[dev]"；Windows 再执行 ./scripts/install-dba.ps1 -Install。之后进入任意目标目录运行 DBA，或运行 DBA --ui web；测试命令为 python -m pytest -q。真实 provider 凭据只放在被 Git 忽略的本机配置/环境变量中，不进入仓库、文档、trace 或 session。

完整说明见根目录 README 和 docs/：其中包含架构、配置与端到端任务记录。
