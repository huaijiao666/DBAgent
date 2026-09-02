# DataBaseAgent（DBAgent）

Git 仓库：https://github.com/huaijiao666/DBAgent

DBAgent（DB 即 Database）是从零实现的本地 Coding Agent Harness。模型负责理解与决策；本地 runtime 掌握上下文、工具权限、修改与验证，形成“理解→计划→修改→测试→恢复→交付”闭环。不依赖 Agent 框架或托管 shell/file/patch 服务。

## 特色功能

1. 仓库感知：用 Python AST 提取符号、import 与关系，生成 repo map，按符号读取局部代码。
2. 本地状态：同一模型按完整语义选择问答/编码权限，维护目标、成功标准与步骤；上下文在本地压缩，不依赖服务端会话。
3. 安全编辑：拒绝 workspace 越界与 symlink escape；多文件 patch 先校验再原子写入，失败回滚；命令限制 cwd、超时、输出和环境。
4. 证据完成：pytest、编译和 lint 是验证证据，修改会使旧证据失效；重复失败触发 recovery，超过步数上限则返回 `INCOMPLETE`。
5. 可观察交互：提供 CLI、REPL、TUI 和浏览器工作台；展示计划、代码变更、验证、token 与脱敏 trace，支持 session 恢复和实时指导。

## 如何运行

要求 Python 3.11+：

```powershell
git clone https://github.com/huaijiao666/DBAgent.git
cd DBAgent
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
.\scripts\install-dba.ps1 -Install
```

设置 `OPENAI_API_KEY` 后，在任意工作目录运行 `DBA`，或用 `DBA --ui web` 打开浏览器工作台。测试：`python -m pytest -q`。

## 其它说明

优先支持 Python 静态结构；`run_command` 不是 OS 级沙箱。详情见 `README.md` 和 `docs/`。
