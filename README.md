# Forge

Forge 是一个从零实现的本地、理解代码仓库且能够自验证的 Coding Agent
Harness。模型只负责推理；文件访问、搜索、patch、命令和验证工具都在指定
workspace 内由本地程序执行。

## 开发状态

这是一个用于学习和面试展示、仍在持续演进的原型（版本 `0.1.0`）。当前
runtime 已能探索仓库、维护有预算的本地 context 和 plan、调用本地工具、
原子应用 patch、执行确定性验证、从失败中恢复，并写出 JSONL 执行 trace。

项目明确不使用 LangChain、LlamaIndex、任何 Agent SDK 或其他现成的
coding-agent runtime。

## 环境要求

- Python 3.11 或更高版本
- 配置的 OpenAI-compatible provider 对应的 API key
- Git 是可选依赖，但 `git_diff` 工具和端到端 demo 需要它

runtime 依赖是官方 `openai` Python client；开发依赖放在
`pyproject.toml` 的 `dev` extra 中。

## 安装

在 repository 根目录创建虚拟环境，并以 editable 模式安装 Forge：

### PowerShell（Windows）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### macOS/Linux shell

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

安装后会提供两个命令：

- `forge` — 运行 coding-agent loop。
- `forge-smoke` — 发送一次文本请求，检查模型连接。

## 配置与密钥

Forge 从当前进程环境读取配置，**不会自动加载** `.env` 文件。`.env.example`
仅用于列出变量名和安全默认值；不要把真实凭据写入该文件、源代码、测试、
Git、README 或 trace。

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 必填 | 官方 client 读取的 API key |
| `FORGE_MODEL` | `gpt-5.6-sol` | 发送给 provider 的模型名 |
| `FORGE_REASONING_EFFORT` | `medium` | 可选 `none`、`low`、`medium`、`high`、`xhigh`、`max` |
| `FORGE_API_MODE` | `responses` | `responses` 或 `chat_completions` |
| `FORGE_BASE_URL` | provider 默认值 | 可选的绝对 `http(s)` base URL |

PowerShell 临时会话示例：

```powershell
$env:OPENAI_API_KEY = "<set-this-out-of-band>"
$env:FORGE_MODEL = "gpt-5.6-sol"
$env:FORGE_REASONING_EFFORT = "medium"
$env:FORGE_API_MODE = "responses"
```

POSIX shell 等价配置：

```bash
export OPENAI_API_KEY='<set-this-out-of-band>'
export FORGE_MODEL='gpt-5.6-sol'
export FORGE_REASONING_EFFORT='medium'
export FORGE_API_MODE='responses'
```

如果兼容 provider 只提供 Chat Completions 而没有 Responses，请设置
`FORGE_BASE_URL` 和 `FORGE_API_MODE=chat_completions`。conversation 仍然由
Forge 在本地组装和维护；两个 adapter 都不使用 `previous_response_id` 或
服务端 conversation state。

## 运行 Agent

任务是位置参数。默认 workspace 是当前目录；更推荐显式传入 repository：

```powershell
forge "inspect this repository and explain its architecture" `
  --workspace . `
  --max-steps 16
```

常用选项：

```text
--workspace PATH       workspace 根目录（默认：当前目录）
--max-steps N          模型轮数硬上限（默认：12）
--trace-file PATH      workspace 内的 JSONL trace 路径
                       （默认：.forge/trace.jsonl）
```

终端 UI 会显示 plan 更新、工具调用、精简 observation、改动文件、验证状态、
耗时，以及 provider 返回的模型 token 使用量。成功运行会输出 `VERIFIED`；
触及轮数硬上限时输出 `INCOMPLETE`，并以状态码 `2` 退出。

界面采用无第三方依赖的行式终端 dashboard，适合直接录制演示视频；在真实
TTY 中自动启用颜色，被重定向到文件或 CI 时保持纯文本。设置 `NO_COLOR=1`
可以强制关闭颜色。示意输出：

```text
+-- Forge coding agent --------------------------------------------------+
| Task      修复 calculator.py 中的除零错误                              |
| Workspace C:\demo\repo                                                 |
| Model     gpt-5.6-sol                                                   |
| Budget    12 model turns                                                |
+-------------------------------------------------------------------------+
.    0.4s |  1/12 | MODEL request  context=812~tok  tools=12
->   1.1s |  1/12 | TOOL -> read_file
OK   1.2s |  1/12 | TOOL <- read_file  ok  path=calculator.py
#    3.5s |  4/12 | PATCH  files=['calculator.py']  hunks=1
OK   5.8s |  5/12 | VERIFY  status=passed  kind=test  return_code=0

+-- Run summary ----------------------------------------------------------+
| Status         VERIFIED                                                 |
| Steps          5/12                                                     |
| Verification   passed                                                   |
| Elapsed        5.8s                                                     |
| Files changed  calculator.py                                            |
+-------------------------------------------------------------------------+
```

## DBA 多轮聊天 REPL

安装 editable package 并激活虚拟环境后，在任意工作目录运行：

```powershell
.\.venv\Scripts\Activate.ps1
DBA --workspace .
```

Windows 命令名不区分大小写；在 macOS/Linux 中也可以使用小写入口：

```bash
dba --workspace .
```

启动后会进入 `DBA>` 提示符。普通文本会触发一次完整的本地 Agent loop，
然后返回提示符等待下一轮；当前会话的用户消息和助手回答会由本地
`LocalConversation` 保留并按预算裁剪，不依赖 provider 的服务端历史。每轮
结束时，`SessionContext` 还会结构化保存最新 plan、verification 状态、recovery
hint，以及 patch、测试、命令和错误等关键工具 observation；下一轮会将这些
摘要重新注入 prompt。代码发生修改后，之前的通过验证会自动标为 `stale`，
避免把旧证据误当成当前结果。

内置命令：

```text
/model                 查看当前模型
/model gpt-5.6-sol     将后续请求切换到指定模型
/status                查看模型、会话轮数和最近任务状态
/clear                 清空本地会话历史（不会删除代码）
/help                  显示帮助
/exit                  退出 DBA
```

`DBA` 默认优先读取用户目录下的备份 provider TOML（也可以用
`--config-path PATH` 或环境变量 `FORGE_BACKUP_CONFIG` 指定其他文件），提取当前 provider 的 `base_url`、
`experimental_bearer_token`、`model` 和 `model_reasoning_effort`。这些值只
进入当前 Python 进程中的配置对象，不会修改父进程环境；备份文件必须放在
repository 外，不要提交或复制到 trace。

每一轮仍然使用现有的 `apply_patch`、`run_command`、repository map 和
deterministic verification，因此多轮 REPL 只是交互入口，不是第二个
Planner 或第二套 Agent runtime。

## 连接 smoke test

设置 `OPENAI_API_KEY` 后运行：

```powershell
forge-smoke
```

也可以传入自定义 prompt：

```powershell
forge-smoke "Reply with one short sentence."
```

该命令只执行一次 stateless 请求，并打印模型、response ID、状态和文本；不会
运行 repository 工具。没有 API key 时，会在发出请求前清晰失败。

## 临时 provider 环境（PowerShell）

可选启动脚本
[`scripts/forge-isolated.ps1`](<C:\AAA\DBAgent\DBAgent\scripts\forge-isolated.ps1>)
可以从仓库外部的 TOML 文件读取 provider URL 和 bearer token，仅在 Forge 子
进程期间设置环境变量，并在退出后恢复调用方环境：

```powershell
.\scripts\forge-isolated.ps1 `
  -Task "inspect this repository and explain its architecture" `
  -ConfigPath "C:\path\outside\the\repo\config.toml" `
  -Model "gpt-5.6-luna" `
  -ReasoningEffort max `
  --workspace . --max-steps 16
```

请把 TOML 文件放在 repository 外并且绝不要提交。该脚本只是便捷封装，不能
替代正式的 secret manager。

## 本地工具与安全模型

模型只能请求以下由本地 registry dispatch 的 function tools：

- 只读探索：`list_files`、`read_file`、`search_text`
- Repository intelligence：`get_repo_map`、`search_symbol`、`read_symbol`
- 编辑与检查：`apply_patch`、`create_file`、`write_file`、`git_diff`
- 执行与规划：`run_command`、`update_plan`

所有路径都会 resolve 并检查是否仍位于选定 workspace 内，包括 symlink escape
检查。`apply_patch` 会在写入任何文件前校验全部 hunk；如果后续 replace 失败，
会回滚已经替换的文件。命令具有 timeout、有限长度的 stdout/stderr 返回值和
过滤后的环境变量。

重要限制：`run_command` **不是操作系统级 sandbox**。被启动的程序仍可能利用
自身能力主动访问 workspace 外的路径。不要在没有额外 OS-level
sandbox/container 和受限账户的情况下运行不可信的 repository 或命令。

## Trace 与 context

每次运行默认将 JSONL 事件写入 `.forge/trace.jsonl`。事件包括模型请求/响应、
工具开始/结果、patch、plan 更新、验证、恢复和最终状态。疑似敏感字段会被
脱敏；trace 有意保存摘要，而不是不受限制的原始模型/工具 payload。

`ContextManager` 在明确的字符预算下保存 persistent task context、当前 plan、
repository map、相关代码和 recent observations。旧 observation 会在本地
compact；不会使用托管 conversation memory。

## 开发与测试

在已激活的虚拟环境中运行完整测试：

```powershell
python -m pytest -q
```

其他有用检查：

```powershell
python -m pip check
python -m compileall -q src
```

测试使用 mock model client，不需要 API key。repository 还包含用于
AST/repository-map、patch、验证的 Python fixture，以及
[`docs/e2e-demo.md`](<C:\AAA\DBAgent\DBAgent\docs\e2e-demo.md>) 中描述的三个
scripted end-to-end task。

## 模块结构概览

```text
src/forge/
  cli.py, config.py, trace.py       CLI、配置、可观察性
  llm/                              Responses 与 Chat Completions adapter
  agent/                            state、loop、plan、context、verification
  tools/                            schema、registry、本地工具处理器
  repository/                       scanner、AST 提取、symbol index
  workspace.py, execution.py       路径和命令安全边界
  patching.py                       事务式 patch 应用
tests/                              unit、安全和端到端测试
```

## 已知限制

- 模型行为和 provider 延迟具有不确定性；即使修复本身简单，任务也可能达到
  `INCOMPLETE`。
- `run_command` 没有 OS-level process/filesystem isolation；timeout 不保证整个
  child-process tree 都被终止。
- 返回给模型的输出会截断，但 subprocess 捕获本身不是严格的内存配额。
- `apply_patch` 是受限的 exact-context protocol，不是完整 unified diff，也不
  提供 file locking。
- `create_file`/`write_file` 仍是过渡编辑工具，patching 才是首选接口。
- Python repository intelligence 是浅层 AST 分析，不能解析动态 import 或证明
  runtime semantics。
- Verification 识别常见 pytest/compiler/linter 命令；不熟悉的项目命令即使
  执行成功，也可能不会更新 verification tracker。
- 目前没有 dependency lockfile，因此不能保证完全可复现的依赖解析。

## License

项目许可证请参阅 [`LICENSE`](<C:\AAA\DBAgent\DBAgent\LICENSE>)。
