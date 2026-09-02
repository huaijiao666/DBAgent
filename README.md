# DBAgent（DBA）

DBAgent 是一个从零实现的本地、理解代码仓库且能够自验证的 Coding Agent
Harness。模型只负责推理；文件访问、搜索、patch、命令和验证工具都在指定
workspace 内由本地程序执行。

项目内部的 Python package 仍使用 `forge` 命名；面向用户的交互命令是 `DBA`。

## 开发状态

这是一个用于学习和面试展示、仍在持续演进的原型（版本 `0.1.0`）。当前
runtime 已能探索仓库、维护有预算的本地 context 和 plan、调用本地工具、
原子应用 patch、执行确定性验证、从失败中恢复，并写出 JSONL 执行 trace。

项目明确不使用 LangChain、LlamaIndex、任何 Agent SDK 或其他现成的
coding-agent runtime。

详细模块边界、状态归属和安全限制见 [架构说明](docs/architecture.md)。

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

安装后会提供三个命令：

- `forge` — 运行 coding-agent loop。
- `forge-smoke` — 发送一次文本请求，检查模型连接。
- `DBA` / `dba` — 启动多轮本地 coding-agent REPL。
- `forge-web` — 直接启动本地浏览器 dashboard（等价于 `DBA --ui web`）。

### Windows 一次性注册 DBA

如果希望在任意工作目录直接输入 `DBA`，在本 repository 根目录执行一次：

```powershell
.\scripts\install-dba.ps1 -Install
```

脚本会确保 `.venv` 和 editable package 存在，并把本项目的 `.venv\Scripts`
加入当前用户 PATH。之后请打开一个新的终端，即可在任意目录运行：

```powershell
cd C:\any\workspace
DBA
```

脚本不会读取、写入或打印 API key。`DBA` 会优先读取 repository 根目录中**被 Git
忽略**的 `config.toml`；也可用 `FORGE_CONFIG_PATH` 或 `--config-path` 指向其他本地
文件。找不到时才回退到当前进程环境变量。

## 配置与密钥

`DBA` 的 provider 配置可放在 repository 根目录中两个**只限本机**的文件：

- `config.toml`：OpenAI-compatible provider 的 URL、模型和 token；可从
  `config.toml.example` 复制后填写。
- `api_key.txt`：DeepSeek API key，仅一行；`api_key.txt.example` 是空白安全模板。

它们都被 `.gitignore` 忽略，绝不会被 package、trace、session 或 README 读取/写入。
用户 clone 后自行复制模板或创建这两个文件并填写值。`.env.example` 只列出环境变量名
和安全默认值；不要把真实凭据写入 `.env`、源代码、测试、Git、README 或 trace。

一个 token 为空的 `config.toml` 模板示例：

```toml
model_provider = "openai_compatible"
model = "gpt-5.6-sol"
model_reasoning_effort = "medium"

[model_providers.openai_compatible]
base_url = "https://provider.example/v1"
experimental_bearer_token = "<fill-your-local-token>"
```

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 必填 | 官方 client 读取的 API key |
| `FORGE_MODEL` | `gpt-5.6-sol` | 发送给 provider 的模型名 |
| `FORGE_REASONING_EFFORT` | `medium` | 可选 `none`、`low`、`medium`、`high`、`xhigh`、`max` |
| `FORGE_API_MODE` | `responses` | `responses` 或 `chat_completions` |
| `FORGE_BASE_URL` | provider 默认值 | 可选的绝对 `http(s)` base URL |
| `FORGE_DEEPSEEK_API_KEY` | 空 | DeepSeek key 的可选环境覆盖 |
| `FORGE_DEEPSEEK_KEY_FILE` | `./api_key.txt` | DeepSeek key 文件的可选显式路径 |
| `FORGE_CONFIG_PATH` | 空 | 可选的 provider TOML 显式路径 |

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

任务是位置参数。未传 `--workspace` 时，`forge` 和 `DBA` 都严格使用启动命令时的
当前目录，避免父目录的 `.git`、`pytest.ini` 或 README 意外扩大 workspace。确实
需要从嵌套 package 向上搜索项目根时，再显式传入 `--discover-workspace`：

```powershell
forge "inspect this repository and explain its architecture" `
  --workspace . `
  --max-steps 16
```

常用选项：

```text
--workspace PATH       workspace 根目录（默认：精确当前目录）
--discover-workspace   显式向父目录搜索最近项目根
--max-steps N          模型轮数硬上限（默认：60）
--mode MODE            auto、ask 或 code（默认：auto）
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
+-- DBAgent coding session ---------------------------------------------+
| Task      修复 calculator.py 中的除零错误                              |
| Workspace C:\demo\repo                                                 |
| Model     gpt-5.6-sol                                                   |
| Budget    12 model turns                                                |
+-------------------------------------------------------------------------+
.    0.4s |  1/12 | 检查  分析中  context=812/7500~tok  recent=2  compact=0
->   1.1s |  1/12 | 检查  读取文件  calculator.py
OK   1.2s |  1/12 | 完成: 读取文件  42 source lines returned  path=calculator.py
#    3.5s |  4/12 | 修改完成  files=['calculator.py']  hunks=1
OK   5.8s |  5/12 | 验证 VERIFY  status=passed  kind=test  return_code=0

+-- Run summary ----------------------------------------------------------+
| Status         VERIFIED                                                 |
| Steps          5/12                                                     |
| Verification   passed                                                   |
| Plan           3/3 steps completed                                      |
| Elapsed        5.8s                                                     |
| Files changed  calculator.py                                            |
+-------------------------------------------------------------------------+
```

## DBA 多轮聊天 REPL

完成上面的 Windows 注册后，在任意工作目录运行：

```powershell
DBA --workspace .
```

DBAgent 提供三种明确的本地展示模式，三者使用**完全相同**的 Agent loop、工具、
session checkpoint 和 JSONL trace；差别只有交互呈现方式：

```powershell
# 默认：可滚动、适合录屏、重定向和 CI 的 CLI dashboard
DBA --ui cli

# 全屏：交互式 TTY 中的 alternate-screen dashboard
DBA --ui tui

# 浏览器 dashboard：本地 loopback 服务 + 浏览器三栏工作台
DBA --ui web
```

TUI 不引入 Agent framework 或 hosted execution：它用标准库 ANSI alternate screen
渲染 task、当前 plan、近期活动、文件改动、verification 和运行中的 steering 输入。
退出时会恢复原来的终端内容。`--ui tui` 只在真实交互式 TTY 中可用；重定向输出、
CI 或不支持 ANSI 的终端请使用 `--ui cli`。

### 浏览器 dashboard

`--ui web` 会在 `127.0.0.1` 启动一个带随机访问 token 的本地服务，并自动打开
浏览器（不希望自动打开时使用 `--no-browser`）。它提供三栏工作台：左侧是
workspace 与仓库树，中间是对话、实时执行 timeline 和任务输入，右侧是 plan、
verification、改动文件与可刷新的 Git diff。运行中会显示当前 step、active tool、
context/model token、耗时；可用 `Send guidance` 把补充约束排入下一安全边界，也可用
`Stop` 请求 Agent 协作退出。

浏览器工作台的对话区和执行时间线各自滚动，新增消息只会在你已经接近底部时自动跟随，
不会把正在阅读的内容强行卷走。左、右两条竖向分隔条和对话/时间线之间的横向分隔条都可
拖动（也支持方向键微调），布局会保存在当前浏览器的 `localStorage` 中；窄窗口会自动切换
为单列布局。界面采用浅色、低装饰的代码工作台风格，重点突出文件、计划、验证和实际变更。

```powershell
# 在当前目录启动，自动选择空闲端口
DBA --ui web --workspace .

# 固定端口并手动复制终端输出的 localhost URL
DBA --ui web --workspace . --port 8765 --no-browser
```

浏览器出于安全策略不能把“文件夹选择器”直接转换成服务器路径，因此 workspace
输入框接受的是**运行 DBA 的这台机器上的绝对路径**；提交后后端会 canonicalize
并验证目录存在。页面不会接收 API key，所有模型请求、文件访问、patch、命令和
trace 仍在本地 Python 进程中执行。服务只绑定 loopback，随机 token 仅用于阻止
同机其他页面随意调用控制 API；关闭终端进程即可停止服务。

直接运行 `DBA` 与 `DBA --workspace .` 等价：默认 workspace 就是启动命令时的
**精确当前目录**，不会因为父目录存在 `.git`、`pytest.ini` 或 `README.md` 而悄悄
扩大权限范围。确实希望从 `src/package` 向上寻找项目根时，才显式使用：

```powershell
DBA --discover-workspace
```

如果环境配置中的模型在当前 provider 上不可用，可以只对本次进程显式覆盖，
而不修改环境：

```powershell
DBA --workspace . --model gpt-5.6-sol --reasoning-effort medium
```

API key 和 URL 仍由当前启动进程的环境提供；覆盖参数不会写入环境或配置文件。

如果没有注册全局命令，也可以在 repository 内激活虚拟环境后运行同一个命令。

Windows 命令名不区分大小写；在 macOS/Linux 中也可以使用小写入口：

```bash
dba --workspace .
```

启动后会进入 `DBA[auto]>` 提示符。普通文本会触发一次完整的本地 Agent loop，
然后返回提示符等待下一轮；当前会话的用户消息和助手回答会由本地
`LocalConversation` 保留并按预算裁剪，不依赖 provider 的服务端历史。每轮
结束时，`SessionContext` 还会结构化保存最新 plan、verification 状态、recovery
hint，以及 patch、测试、命令和错误等关键工具 observation；下一轮会将这些
摘要重新注入 prompt。代码发生修改后，之前的通过验证会自动标为 `stale`，
避免把旧证据误当成当前结果。

每轮请求会原子保存到 workspace 的 `.forge/sessions/<SESSION_ID>.json`，其中只
包含裁剪后的对话、plan、验证摘要和关键 observation，并沿用 trace 的敏感值
脱敏规则。重新打开同一工作目录后，使用 `/sessions` 查看 ID，再用
`/resume <ID>` 精确恢复；`/resume #` 可按列表序号恢复；`/resume latest` 恢复最近更新的会话。旧版本的
`.forge/session.json` 会作为 `legacy` 会话继续可读。所有文件均受 `.gitignore`
保护，不使用 provider 的服务端 conversation state。

每次启动都会创建一个标记为 `[new]` 的空 session ID，**不会自动恢复历史**。
成功 `/resume <ID>` 后终端会显示 `Resumed context` 面板，明确列出恢复的标题、
聊天轮数、verification、关键 observation 数量以及是否恢复了 plan；若存在 plan
和验证摘要，也会紧接着可视化展示。

长任务会在每个完成的 Agent step 后写入本地 checkpoint；即使 provider 失败或用户
中断，已完成的 plan 更新、验证结果和关键工具 observation 也会保存为
`Checkpoint: interrupted`，之后可用 `/resume <ID>` 恢复。尚未返回的那一次模型
请求本身无法从中间继续，但其用户任务文本在请求前已保存。

当上一个 code 任务留下未完成 plan，用户明确输入“继续/接着”或 `continue/resume`
时，REPL 会把该 plan 恢复为下一轮的真实 runtime state，而不只是复制一段文本；
上轮若为 `failed/stale` 验证，本轮在取得新的确定性证据前不能声称完成。普通的
新请求不会自动继承旧 plan，避免把无关任务绑在一起。

内置命令：

```text
/models                显示可直接选择的模型别名与 provider
/model                 显示模型别名；也可继续输入 provider 的原始模型名
/model luna            使用启动时环境配置的 gpt-5.6-luna
/model terra           使用启动时环境配置的 gpt-5.6-terra
/model sol             使用启动时环境配置的 gpt-5.6-sol
/model deepseek-flash  使用 DeepSeek V4 Flash
/model deepseek-pro    使用 DeepSeek V4 Pro
/reasoning             查看当前 reasoning effort 和可用等级
/reasoning high        对后续请求设置 reasoning effort
/steps [N]             查看或设置本 REPL 后续任务的 model turn 上限
/mode                  查看当前任务模式
/mode auto             自动区分问答与编码任务
/mode ask              只做仓库调查和回答，不开放编辑工具
/mode code             开放规划、patch、命令和验证能力
/status                查看模型、会话轮数和最近任务状态
/context               查看最近一次请求的本地 context 预算和压缩统计
/capabilities          查看当前 provider 实际生效的工具/推理能力
/plan                  查看当前会话保留的最新结构化 plan
/sessions              列出当前 workspace 的会话 ID、时间、轮数和验证状态
/resume <ID|#>         按完整 ID、唯一前缀或会话列表序号恢复
/resume latest         恢复最近更新的会话
/continue [N]          继续当前未完成 plan；可选地设定新的 step budget
/new                   开始新会话并保留已有会话
/clear                 清空并删除当前会话，保留其他会话（不会删除代码）
/help                  显示帮助
/exit                  退出 DBA
```

在正常交互式终端的一次任务执行期间，也可直接输入 `/steer <指令>`（或直接输入
一行普通文本）补充下一安全边界的用户指令；输入 `/abort` 会阻止下一次模型请求或
本地工具执行。它不会伪称能取消正在飞行中的 HTTP 请求或已启动的子进程。Responses
API 路径还会显示经过小批量整理的服务端文本增量；工具参数不会在完成前被当作可执行
数据展示或执行。

为复现本项目已测试的开发依赖，可在新虚拟环境中执行：

```powershell
python -m pip install -r requirements.lock
python -m pip install -e .
python -m pytest
```

`max_steps` 不是模型能力的固定上限，而是 DBAgent 每个本地 Agent run 的安全预算：
它避免模型、provider 或工具异常时无限消耗时间和额度。普通命令默认 60 步；可用
`DBA --max-steps 64`、REPL 中的 `/steps 64`，或在已中断任务后直接
`/continue 64`。续跑会恢复本地 plan、关键 observation 和验证状态，但开始一个新的、
明确受限的 Agent run，而不是悄悄无限循环。

`DBA` 不再扫描用户目录或任何聊天软件备份；它只从 repository 根目录的忽略文件
`config.toml` 读取默认 provider 配置，或从 `FORGE_CONFIG_PATH`、兼容的
`FORGE_BACKUP_CONFIG`、`--config-path` 和当前进程环境接收显式配置。凭据只进入当前
进程内存，不会写入 session、trace 或 Git。

选择 `/model deepseek-flash` 或 `/model deepseek-pro` 时，DBA 会优先读取
`FORGE_DEEPSEEK_API_KEY`；没有时读取 repository 根目录中被忽略的 `api_key.txt`，并使用
`https://api.deepseek.com` 的 Chat Completions 兼容接口。该 key 不写入 session、trace、终端输出或本 repository；切回 `luna`、
`terra` 或 `sol` 会恢复本次启动时的 configured provider 凭证和 URL。DeepSeek 的两个预设使用显式的
`deepseek` provider policy。由于该兼容接口在 thinking 与多轮工具历史组合下有
额外协议要求，DBAgent 的本地工具任务会明确显示并强制使用 `thinking disabled`；
`/reasoning` 只作为偏好保存，不代表 DeepSeek 工具轮实际启用了该能力。这样不会
把不完整的 `reasoning_content` 当成可恢复的本地上下文，也不会把文本 DSML 误当作
工具调用执行。DeepSeek 预设应视为 experimental compatibility，重要演示优先使用
稳定的 Responses provider。

Windows 是默认运行路径。WSL 可以单独安装 Python、editable package 和 provider
配置，但它不会自动复用 Windows 的 `.venv` 或 Windows 路径；当前项目没有必要
为了命令兼容性强制切换到 WSL。

每一轮仍然使用现有的 `apply_patch`、`run_command`、repository map 和
deterministic verification，因此多轮 REPL 只是交互入口，不是第二个
Planner 或第二套 Agent runtime。

`auto` 模式会将“怎么运行、有什么功能、解释架构”等请求路由到 `ask`：不创建
装饰性计划、不暴露编辑工具，收集足够证据后强制生成答案。带有“修复、实现、
添加、修改代码”等明确意图的请求进入 `code`：仅对真正的多步骤任务维护 plan，
修改后必须用本地测试、编译器或 linter 形成当前验证证据。自动分类是启发式的，
重要任务可以用 `/mode` 明确覆盖。

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
[`scripts/forge-isolated.ps1`](scripts/forge-isolated.ps1)
适合只使用当前 PowerShell 进程中已有的 provider 环境变量；它不修改调用方环境：

```powershell
.\scripts\forge-isolated.ps1 `
  -Task "inspect this repository and explain its architecture" `
  --workspace . --max-steps 16
```

若需要自动读取个人 TOML，请直接使用 `DBA`；该脚本仍适合 CI 或显式环境配置。

## 本地工具与安全模型

模型只能请求以下由本地 registry dispatch 的 function tools：

- 只读探索：`list_files`、`read_file`、`search_text`
- Repository intelligence：`get_repo_map`、`search_symbol`、`read_symbol`
- 编辑与检查：`apply_patch`、`create_file`、`write_file`、`git_diff`
- 执行与规划：`run_command`、`update_plan`

所有文件路径都会 resolve 并检查是否仍位于选定 workspace 内，包括 symlink
escape 检查。`apply_patch` 会在写入任何文件前校验全部 hunk；如果后续 replace
失败，会回滚已经替换的文件。命令具有 timeout、有限长度的 stdout/stderr 返回值和
过滤后的环境变量；它使用 argv 而非 shell，但目前**不是 OS 级沙箱**，不应在不可信
repository 或高权限机器上当作隔离边界。

patch 被拒绝时终端会展示具体原因，例如 `context did not match`、`context is
ambiguous` 或 `replacement makes no change`。失败是原子的；runtime 会要求模型
不要重复同一补丁，而是缩小唯一上下文，必要时只对小文件使用 `write_file`
作为显式 fallback。

重要限制：`run_command` **不是操作系统级 sandbox**。被启动的程序仍可能利用
自身能力主动访问 workspace 外的路径。不要在没有额外 OS-level
sandbox/container 和受限账户的情况下运行不可信的 repository 或命令。

## Trace 与 context

每次运行默认将 JSONL 事件写入 `.forge/trace.jsonl`。事件包括模型请求/响应、
工具开始/结果、patch、plan 更新、验证、恢复和最终状态。疑似敏感字段会被
脱敏；trace 有意保存摘要，而不是不受限制的原始模型/工具 payload。重新启动
DBA 时会追加新事件，不会截断之前用于诊断失败的历史。

`ContextManager` 在明确的字符预算下保存 persistent task context、当前 plan、
repository map、相关代码和 recent observations。旧 observation 会在本地
compact；不会使用托管 conversation memory。默认总上限为 80,000 字符（用字符
数做 tokenizer-independent 的保守边界）：最近 8 条 observation 可保留原文，单条
最多 6,000 字符；更早的结果变成本地摘要。每次 `model_request` 会显示近似 token
占用、保留的 recent observation 和已摘要数量；出现“上下文摘要”并不等于丢失全部
历史，`truncated` 才表示某条特别长的原始输出被截断。长时间等待 provider 时，终端会输出降频
心跳，但心跳不会写入 JSONL，也不表示模型已经取得实际进展。

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

- CLI dashboard 保持可录制和可重定向；TUI 是交互式 ANSI dashboard，浏览器模式是
  loopback 三栏工作台。两种界面都不会把完整源码、stdout 或模型隐藏 reasoning 直接
  刷屏；交互终端支持安全边界上的 `/steer`、`/followup` 和 `/abort`，浏览器提供
  `Stop`，但都不支持在单次 HTTP 请求或运行中的子进程中强制抢占。
- 模型行为和 provider 延迟具有不确定性；即使修复本身简单，任务也可能达到
  `INCOMPLETE`。
- 第三方 OpenAI-compatible provider 的连接质量、响应延迟、模型语言和 function
  calling 质量不由 Harness 控制；瞬时连接/限流/超时默认最多重试 5 次，仍失败
  才明确报告，并且重试不消耗 Agent step。
- `auto` 模式依赖可解释的关键词启发式，存在误分类可能；可用 `/mode ask|code`
  明确覆盖。
- `run_command` 没有 OS-level process/filesystem isolation；timeout 不保证整个
  child-process tree 都被终止。
- 返回给模型的输出会截断，但 subprocess 捕获本身不是严格的内存配额。
- `apply_patch` 是受限的 exact-context protocol，不是完整 unified diff，也不
  提供 file locking。
- `/resume <ID>` 恢复的是裁剪后的对话与结构化任务状态，不会重放进程崩溃前已经完成
  但尚未写入 session checkpoint 的单个模型推理；本轮用户请求会在 API 调用前
  先保存，因此至少不会丢失任务文本。
- `create_file`/`write_file` 仍是过渡编辑工具，patching 才是首选接口。
- Python repository intelligence 是浅层 AST 分析，不能解析动态 import 或证明
  runtime semantics。
- Verification 识别常见 pytest/compiler/linter 命令；不熟悉的项目命令即使
  执行成功，也可能不会更新 verification tracker。
- 重复证据检测按工具参数和未变化文件范围做保守判断；它会避免把相同或被完整
  覆盖的读取算作进展，但不会替代模型对证据是否充分的判断。
- DeepSeek Chat Completions 兼容路径的 native tool calling 和 thinking 协议仍受
  provider 行为影响，默认稳定演示应优先选择已验证的 Responses provider。
- `requirements.lock` 固定了本项目在 Windows 上测试过的 Python 包版本；它不是
  带哈希的跨平台供应链锁，也不替代受控环境中的镜像、Python 解释器版本和 OS 依赖。

## License

项目许可证请参阅 [`LICENSE`](<C:\AAA\DBAgent\DBAgent\LICENSE>)。
