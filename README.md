# DataBaseAgent（DBAgent）

DBAgent 是一个从零实现的、本地运行的 repository-aware、self-verifying
Coding Agent Harness。模型负责推理和发起原生 function call；代码读取、搜索、
patch、命令执行、验证、会话管理和可观察性全部由本地 Python runtime 自行实现。

DataBaseAgent（DBAgent）中的 DB 指 Database；项目定位是通用的本地编程智能体。

项目内部 package 名为 `dbagent`，面向用户的交互命令是 `DBA`。

> 开发状态：用于软件工程项目展示与学习的可运行原型（`0.1.0`），不是对
> Codex 或 Claude Code 的能力、隔离性或稳定性的等价替代品。

## 项目目标与边界

给定一个用户选择的本地 workspace，DBAgent 可以探索代码仓库、维护显式任务计划、
通过本地工具修改代码、执行确定性检查，并基于测试/编译/lint 结果决定是否验证完成。

```text
用户（CLI / TUI / Browser UI）
            │
            ▼
      DBAgentRepl / BrowserAgentController
            │
            ▼
AgentLoop ─ ContextManager ─ AgentState / TaskPlan
    │              │                  │
    │              │                  └─ verification / recovery / termination
    ▼              ▼
OpenAI Responses 或 Chat Completions 兼容适配层
    │
    ▼
ToolRegistry ─ Workspace / PatchApplier / CommandExecutor / RepositoryIndex
    │
    ▼
TraceRecorder（本地 JSONL）→ CLI、TUI、Browser UI
```

### 题目约束符合性

- 不使用 LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、
  AutoGen、CrewAI 或其他 Agent framework / SDK。
- 不把 Codex CLI、Claude Code 或现成 Coding Agent 当作 runtime。
- 官方 `openai` Python client 只承担模型 API 通信；默认优先使用 Responses API。
- 不启用 Code Interpreter、hosted shell、Files API、web search 或服务端 patch。
- 模型只能请求项目自己定义的 native function tools；每个工具实际在本机执行。
- 不使用 `previous_response_id` 或服务端 conversation state；每轮 prompt 由本地
  `ContextManager` 和会话状态明确构造。

详细模块说明见 [架构说明](docs/architecture.md)；三个可复现的 scripted
端到端演示任务见 [E2E demo](docs/e2e-demo.md)。

### 推免提交材料

本文件是面向使用者与考官的完整项目说明。考核要求的精简 `README.txt` 已放在
[submission/README.txt](submission/README.txt)：提交前只需将其中的仓库地址占位符
替换成你的公开仓库 URL。两分钟 MP4 的任务选择、录屏步骤和逐段讲稿见
[视频演示脚本](docs/video-demo-script.md)。这些材料不含 API key、URL 或本地凭据。

## 当前能力

| 能力 | 本地实现 | 关键事实 |
| --- | --- | --- |
| 仓库探索 | `list_files`、`read_file`、`search_text` | 路径 canonicalize，并限制在 workspace 内 |
| Python 仓库理解 | scanner、ignore rules、AST symbol index | 提供 repo map、symbol 搜索与局部 symbol 阅读；不是完整静态分析 |
| 代码编辑 | `apply_patch` 为主，`create_file` / `write_file` 为过渡接口 | 多文件 exact-context patch 先整体校验；失败不留下半应用状态 |
| 本地执行 | `run_command`、`git_diff` | argv 无 shell、timeout、输出上限、清理后的子进程环境 |
| 任务控制 | `TaskPlan`、`AgentState`、`max_steps` | plan、步骤、观察、验证和终止状态均为显式数据 |
| 自验证 | verification tracker、重复失败检测、recovery hint | 模型说“完成”不等于 `VERIFIED`；需要当前确定性证据 |
| 长任务上下文 | `ContextManager`、`SessionContext` | 本地预算、确定性压缩、保留关键错误/修改/验证事实 |
| 可观察性 | `TraceRecorder`、CLI、TUI、浏览器 dashboard | JSONL trace、计划、工具、变更、验证、耗时与 token 指标 |

## 环境要求

- Python 3.11 或更高版本。
- 可用的 OpenAI 或 OpenAI-compatible provider 凭据。
- Git 可选；`git_diff` 与 E2E demo 需要 Git。
- runtime 唯一第三方依赖是官方 `openai` Python client；开发测试依赖在
  `pyproject.toml` 的 `dev` extra 中。

## 安装

推荐以 editable 模式安装：这样 `DBA` 能稳定定位本 checkout 根目录的本地配置文件。

### PowerShell（Windows）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

安装后提供以下入口：

- `dbagent`：一次性本地 Coding Agent run。
- `dbagent-smoke`：一次 stateless 模型连接检查。
- `DBA` / `dba`：多轮本地 Coding Agent REPL。
- `dbagent-web`：直接启动浏览器 dashboard。

### Windows：在任意工作目录输入 `DBA`

在 repository 根目录执行一次：

```powershell
.\scripts\install-dba.ps1 -Install
```

该脚本创建/复用本项目的 `.venv`、以 editable 模式安装 package，并将该虚拟环境的
`Scripts` 目录加入当前用户 PATH。重新打开终端后即可：

```powershell
cd C:\your\workspace
DBA
```

安装脚本不会写入、打印或上传 API key。

## 配置与密钥

### 推荐的 DBA 本地配置

在**本 repository 根目录**创建两个仅供本机使用的文件：

- `config.toml`：configured OpenAI-compatible provider 的 URL、模型与 bearer token。
- `api_key.txt`：DeepSeek key；文件只能包含一行非空 key。

真实文件都被 `.gitignore` 忽略。clone 后请复制安全模板并自行填写，不要把真实值加入
Git、README、测试、trace 或 session：

```powershell
Copy-Item .\config.toml.example .\config.toml
Copy-Item .\api_key.txt.example .\api_key.txt
# 然后用本机编辑器填写 config.toml 的 token；api_key.txt 只填写一行 DeepSeek key。
```

`config.toml.example` 不含真实 token；`api_key.txt.example` 是空文件。可用结构如下：

```toml
model_provider = "openai_compatible"
model = "gpt-5.6-sol"
model_reasoning_effort = "medium"

[model_providers.openai_compatible]
base_url = "https://provider.example/v1"
experimental_bearer_token = "<fill-your-local-token>"
```

此 TOML 格式用于兼容 configured provider；DBA 将 token 只读入当前进程的
`DBAgentConfig`，并使用 Chat Completions compatibility adapter。它不会写回环境变量、
session 或 trace。

可在提交前确认 Git 不会跟踪本机凭据：

```powershell
git check-ignore -v -- config.toml api_key.txt
```

此外，`config.toml`、`api_key.txt`、`.env*` 都被 Agent 文件工具、repository scanner
和浏览器文件预览主动屏蔽；这不只依赖 `.gitignore`。

### 配置优先级

`DBA`（CLI/TUI/Browser UI）按以下顺序加载 configured provider：

1. `--config-path PATH`
2. `DBAGENT_CONFIG_PATH`（兼容旧名称 `DBAGENT_BACKUP_CONFIG`）
3. 本 repository 根目录、且被忽略的 `config.toml`
4. 当前进程环境变量

DeepSeek 预设按以下顺序读取 key：

1. `DBAGENT_DEEPSEEK_API_KEY`
2. `DBAGENT_DEEPSEEK_KEY_FILE`
3. 本 repository 根目录、且被忽略的 `api_key.txt`

`dbagent` 和 `dbagent-smoke` 刻意只从**当前进程环境**创建 `DBAgentConfig`；它们不自动读取
本地 TOML。这让一次性 CLI 与 smoke test 的凭据来源完全显式。

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 空 | `dbagent` / `dbagent-smoke` 或 DBA 环境回退使用的 API key |
| `DBAGENT_MODEL` | `gpt-5.6-sol` | 模型名 |
| `DBAGENT_REASONING_EFFORT` | `medium` | `none`、`low`、`medium`、`high`、`xhigh`、`max` |
| `DBAGENT_API_MODE` | `responses` | `responses` 或 `chat_completions` |
| `DBAGENT_BASE_URL` | provider 默认 | 可选 OpenAI-compatible base URL |
| `DBAGENT_CONFIG_PATH` | 空 | provider TOML 的显式路径 |
| `DBAGENT_DEEPSEEK_API_KEY` | 空 | DeepSeek key 的显式覆盖 |
| `DBAGENT_DEEPSEEK_KEY_FILE` | 空 | DeepSeek key 文件的显式路径 |

使用纯环境变量的例子：

```powershell
$env:OPENAI_API_KEY = "<set-this-out-of-band>"
$env:DBAGENT_MODEL = "gpt-5.6-sol"
$env:DBAGENT_REASONING_EFFORT = "medium"
$env:DBAGENT_API_MODE = "responses"
```

## 快速开始

### 一次性任务

`dbagent` 的 workspace 默认是**启动命令时的精确当前目录**；不会自动上溯到父目录。

```powershell
cd C:\your\workspace
dbagent "inspect this repository and explain its architecture" --max-steps 16
```

若确实需要从嵌套目录显式寻找项目根：

```powershell
dbagent "run the focused tests and explain the failure" --discover-workspace
```

常用选项：

```text
--workspace PATH       workspace 根目录（默认：精确当前目录）
--discover-workspace   显式向上寻找最近的项目根
--max-steps N          每次 run 的模型轮数硬上限（默认：60）
--mode MODE            auto、ask 或 code（默认：auto）
--trace-file PATH      workspace 内 JSONL trace 路径（默认：.dbagent/trace.jsonl）
```

当达到 `max_steps` 时，Agent 明确输出 `INCOMPLETE` 并以状态码 `2` 退出；不会伪装成
验证成功。

### 多轮 REPL

```powershell
cd C:\your\workspace
DBA --workspace .
```

启动 workspace 默认同样是精确当前目录。`DBA --discover-workspace` 是唯一会向上选择
项目根的方式。

内置命令：

```text
/models                         显示模型预设
/model luna|terra|sol           选择 configured provider 的模型预设
/model deepseek-flash|deepseek-pro
                                选择当前 DeepSeek 兼容预设
/reasoning [LEVEL]              查看或设定后续请求的 reasoning effort
/steps [N]                      查看或设定每次任务的 step budget
/mode auto|ask|code             选择任务模式
/status /context /capabilities  查看状态、context 预算、实际 provider 能力
/plan                           显示保留的结构化计划
/sessions                       列出当前 workspace 的已保存会话
/resume <ID|#|latest>           恢复指定会话
/continue [N]                   继续未完成 plan，可同时调整预算
/new                            创建新会话，保留原会话
/clear                          删除当前会话 checkpoint，不删除用户代码
/steer <要求> 或 /followup <要求>
                                在执行期间把指导放入下一安全边界
/abort                          请求在下一模型/工具边界停止
/help /exit                     查看帮助 / 退出
```

每轮开始前，用户任务会先写入本地 checkpoint；完成后会保存裁剪后的对话、plan、关键
observation 与验证状态到 `.dbagent/sessions/<SESSION_ID>.json`。`/resume` 恢复的是
本地上下文，不会声称恢复一个中断中的 HTTP 请求或子进程。

### 三种展示方式

三种 UI 使用同一个 AgentLoop、ToolRegistry、SessionStore 和 JSONL trace，只改变展示层：

```powershell
DBA --ui cli    # 默认：可滚动的终端 dashboard
DBA --ui tui    # 交互式 TTY 的 ANSI 全屏 dashboard
DBA --ui web    # loopback 浏览器工作台
```

- CLI 适合录屏、重定向输出和 CI。
- TUI 只在交互式 TTY 可用；退出后恢复原终端内容。
- 浏览器模式仅绑定 `127.0.0.1`，使用随机本地访问 token；浏览器从不接收 provider key。

浏览器工作台支持原生目录选择器或手动路径输入；切换工作区会自动创建一个新的空会话，
旧会话保留在原 workspace。仓库树和变更列表可打开多标签文件预览；会话恢复会重建历史
对话、最终回答以及压缩后的本地执行摘要。plan 保留至新的 plan 取代它，变更面板显示文件
数量与 `+/-` 行数。布局分隔条可拖动，并保存在浏览器 `localStorage` 中。

```powershell
# 自动选择本机空闲端口并打开浏览器
DBA --ui web --workspace .

# 指定端口；自行打开终端打印的 localhost URL
DBA --ui web --workspace . --port 8765 --no-browser
```

浏览器模式中的模型和 reasoning effort 应在页面设置控件中选择。`DBA --model` 与
`--reasoning-effort` 是 CLI/TUI 启动覆盖；当前 browser launcher 不转发这两个参数。

## 模型通信与 provider 兼容性

默认模型为 `gpt-5.6-sol`，默认 reasoning effort 为 `medium`。Responses adapter 和
Chat Completions compatibility adapter 都只接受普通文本与项目自定义的 function calls；
它们不启用任何 hosted execution/file/search 能力。

对于只提供 Chat Completions 的 compatible provider，可设置：

```powershell
$env:DBAGENT_BASE_URL = "https://provider.example/v1"
$env:DBAGENT_API_MODE = "chat_completions"
```

DeepSeek `deepseek-flash` / `deepseek-pro` 是实验性兼容路径。某些 provider 在 thinking
与多轮 tool history 组合时要求回传专有字段；DBAgent 的工具轮因此遵守更保守的本地协议，
且只执行原生 function call。若 provider 返回文本形式的伪工具标记，runtime 会安全报错，
不会把文本当命令执行。重要演示优先选择已验证的 Responses provider。

### 手工连接 smoke test

`dbagent-smoke` 会消耗一次模型调用额度，并且只读取环境变量：

```powershell
dbagent-smoke
dbagent-smoke "Reply with one short sentence."
```

它不会运行 repository tools。缺少 `OPENAI_API_KEY` 时会在请求前失败。

## 本地工具、安全与验证

模型可请求的 native function tools：

- 探索：`list_files`、`read_file`、`search_text`
- 仓库理解：`get_repo_map`、`search_symbol`、`read_symbol`
- 编辑：`apply_patch`、`create_file`、`write_file`、`git_diff`
- 执行与计划：`run_command`、`update_plan`

关键安全策略：

- 所有文件路径先 resolve，确认仍处于 workspace；symlink escape 被拒绝。
- `apply_patch` 是受限的 exact-context 多文件协议。它会先校验全部 hunk，随后以可恢复的
  文件替换完成提交；失败返回结构化原因，且不留下半应用修改。
- `run_command` 以 argv 启动，禁止 shell，限定 timeout 和 stdout/stderr 长度，并不向
  子进程无条件继承 API key 等敏感环境变量。
- `git_diff` 使用固定的只读 Git 命令；Agent 从不自动执行 Git commit、rebase、reset 或 push。
- `.env*`、`config.toml`、`api_key.txt` 被本地工具禁止访问。

`run_command` 不是 OS 级 sandbox。即使工具入口做了限制，被启动的可信程序仍可能利用
操作系统权限访问 workspace 外资源。因此不要在不可信仓库、高权限账户或生产机器上运行它；
需要更强隔离时应在受限账户、container 或虚拟机中运行。

验证不依赖另一个 LLM 猜测：运行过的 pytest、compiler 或 linter 等确定性证据保存为
latest verification。若后续修改文件，旧的通过证据会被标记为 `stale`；最终模型文字回答
本身不会自动得到 `VERIFIED`。重复失败和连续无进展会触发 recovery hint；达到预算则是
`INCOMPLETE`。

## Context、session 与 trace

`ContextManager` 本地管理 persistent task context、current plan、repository map、相关代码
和 recent observations。默认总预算为 80,000 字符，使用字符数作为 tokenizer-independent
的保守边界：近期 observation 可保留原文，过旧内容确定性压缩为摘要。每个模型请求会记录
近似 token 使用、近期 observation 数和 compaction 情况。

每次 run 默认向 workspace 写入 `.dbagent/trace.jsonl`。Trace 包含模型请求/响应、工具开始/
结果、patch、plan、verification、recovery 与 final 等结构化事件。疑似敏感字段经过脱敏；
trace 有意保存摘要而非无限原始 payload。`.dbagent/` 被 `.gitignore` 忽略。

## 开发与测试

完整测试不消耗 API 额度，使用 mock/scripted model client：

```powershell
python -m pytest -q
python -m compileall -q src
python -m pip check
git diff --check
```

已覆盖的关键组件包括 tool registry/dispatch、workspace sandbox、symlink、命令 timeout/
截断/环境清理、patch atomicity、context compaction、Python AST repository map、plan transition、
termination、verification recovery、session 与浏览器控制器。Windows 未启用创建 symlink 权限时，
对应安全测试会明确跳过而非伪造通过。

可用 `requirements.lock` 安装项目已验证过的依赖版本；它不是带哈希的跨平台供应链锁。

## 模块结构

```text
src/dbagent/
  cli.py, repl.py, web_ui.py, tui.py  交互入口与展示层
  config.py, provider_config.py       配置与本机凭据加载边界
  llm/                                Responses / Chat Completions 适配层
  agent/                              state、loop、plan、context、verification、control
  tools/                              tool schema、registry、本地 handlers
  repository/                         scanner、Python AST、symbol index、repo map
  workspace.py, execution.py          路径与命令安全边界
  patching.py                          原子 exact-context patch
  trace.py, session_store.py           结构化 trace 与本地会话持久化
tests/                                单元、安全、集成和 scripted E2E 测试
```

## 已知限制

- 模型行为、工具调用质量和 provider 延迟具有不确定性；简单任务也可能达到 `INCOMPLETE`。
- `run_command` 没有 OS-level process/filesystem isolation；timeout 也不保证终止完整
  child-process tree。
- `apply_patch` 是受限 exact-context 协议，而非完整 unified diff；也不提供 file locking。
- `create_file` / `write_file` 是过渡编辑接口，patch 是首选。
- Python repository intelligence 是浅层 AST 分析，不能解析动态 import 或证明 runtime semantics。
- verification tracker 优先识别常见 pytest/compiler/linter 命令；陌生命令可执行但未必更新状态。
- `/resume` 恢复裁剪后的对话和结构化状态，不重放中断前尚未完成的一次模型请求。
- 浏览器 mode 不是远程 Web 服务；关闭本地进程即停止，且不应暴露 loopback URL/token。
- 以非-editable wheel 安装时，请用 `DBAGENT_CONFIG_PATH` / `DBAGENT_DEEPSEEK_KEY_FILE` 指定
  凭据位置，或使用环境变量；默认根目录本地配置面向本 checkout 的 editable 安装。

## License

详见 [LICENSE](LICENSE)。
