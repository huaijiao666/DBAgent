# DBAgent 使用与配置

本页保留运行细节；项目定位、架构和约束请看根目录 [README](../README.md)。任何真实
key、token、provider URL 都只能存在于 Git 忽略的本机文件或当前进程环境中。

## 安装与入口

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
.\scripts\install-dba.ps1 -Install
```

安装脚本只为当前用户 PATH 加入本项目 `.venv\Scripts`，不写入或打印凭据。重新打开
终端后可在任意 workspace：

```powershell
DBA
DBA --ui web
dbagent "inspect this repository and explain its architecture"
dbagent-smoke
```

`DBA` / `dba` 是多轮 REPL；`dbagent` 是一次性 run；`dbagent-smoke` 只发普通文本模型
请求，不执行 repository tools；`dbagent-web` 直接启动 loopback browser UI。

## Workspace 规则

workspace 默认是启动命令时的**精确当前目录**，不会自动扩展到父级仓库。只有显式使用
`--discover-workspace` 才会向上寻找项目根；这避免在嵌套目录中无意扩大文件访问权限。

```text
--workspace PATH       显式 workspace
--discover-workspace   明确允许向上寻找项目根
--max-steps N          每个用户任务的硬模型轮数上限（默认 60）
--mode auto|ask|code   语义自动路由、只读问答、允许修改
--trace-file PATH      workspace 内 JSONL trace（默认 .dbagent/trace.jsonl）
```

## 本机 provider 配置

交互式 `DBA` 按以下顺序读取 configured provider：

1. `--config-path PATH`
2. `DBAGENT_CONFIG_PATH`（兼容 `DBAGENT_BACKUP_CONFIG`）
3. 本项目根目录、被 Git 忽略的 `config.toml`
4. 当前进程环境变量

DeepSeek key 的优先级是：

1. `DBAGENT_DEEPSEEK_API_KEY`
2. `DBAGENT_DEEPSEEK_KEY_FILE`
3. 本项目根目录、被 Git 忽略的 `api_key.txt`

可从模板创建本机文件：

```powershell
Copy-Item .\config.toml.example .\config.toml
Copy-Item .\api_key.txt.example .\api_key.txt
```

模板只包含安全占位符。configured provider 的 TOML 需要顶层 `model_provider`、`model`、
`model_reasoning_effort`，及 `model_providers.<name>` 内的 `base_url` 和
`experimental_bearer_token`。运行时只将这些值读入内存；工具、repo scanner、浏览器文件预览
都主动屏蔽 `config.toml`、`api_key.txt`、`.env`。

`dbagent` 和 `dbagent-smoke` 刻意只读**当前进程环境**，便于一次性调用与 CI 中明确管理凭据。

## 模型与推理强度

默认值：

```text
DBAGENT_MODEL=gpt-5.6-sol
DBAGENT_REASONING_EFFORT=medium
DBAGENT_API_MODE=responses
```

常见变量：

| 变量 | 用途 |
| --- | --- |
| `OPENAI_API_KEY` | 一次性 CLI 或环境回退使用的 key |
| `DBAGENT_MODEL` | 默认模型名 |
| `DBAGENT_REASONING_EFFORT` | `none/low/medium/high/xhigh/max` |
| `DBAGENT_API_MODE` | `responses` 或 `chat_completions` |
| `DBAGENT_BASE_URL` | OpenAI-compatible provider base URL |

REPL 中可使用：

```text
/models
/model luna|terra|sol
/model deepseek-flash|deepseek-pro
/reasoning [LEVEL]
/steps [N]
```

Chat Completions compatible provider 只允许普通文本和项目自定义 function calling。DeepSeek
路径属于兼容模式：工具轮使用保守的本地协议，文本 DSML/XML 伪工具调用会被拒绝，不会执行。

## 多轮会话

```text
/mode auto|ask|code       切换任务权限
/status /context          查看当前状态与 context 预算
/plan                     查看保留的结构化计划
/sessions                 列出当前 workspace 本地会话
/resume <ID|#|latest>     恢复指定本地会话
/continue [N]             延续未完成任务，可提高轮数预算
/new                      新会话，不删除旧 checkpoint
/steer <要求>             将补充要求放入下一个安全边界
/abort                    请求停止
/clear                    删除当前 session checkpoint，不删除用户代码
```

会话数据保存在 workspace 的 `.dbagent/sessions/`。resume 恢复裁剪过的对话、plan、关键
observation 与 verification，不重放一个中断的模型请求或子进程。

## 三种展示方式

```powershell
DBA --ui cli
DBA --ui tui
DBA --ui web
```

- CLI：适合录屏和普通终端。
- TUI：交互式 TTY 全屏 dashboard，退出后恢复终端。
- Browser：只监听 `127.0.0.1`，以随机本地 token 保护；前端不接收 provider key。支持选择
  workspace、repo/changes 预览、多标签文件、会话恢复、计划、验证和 trace 摘要。

## 安全提醒

不要在不可信仓库、高权限账户或生产机器上执行 Agent。`run_command` 使用 argv、timeout、
输出截断、白名单环境与危险命令 policy，但不是完整 OS sandbox。更强隔离应交给低权限账户、
container 或虚拟机。
