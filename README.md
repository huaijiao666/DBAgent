# DataBaseAgent（DBAgent）

> 从零实现的本地、repository-aware、self-verifying Coding Agent Harness。

**Git 仓库：** <https://github.com/huaijiao666/DBAgent>

DBAgent 面向真实本地代码仓库完成编程任务。模型负责分析和提出原生 function calling
请求；本地 Python runtime 负责保存任务状态与上下文、执行工具、保护文件边界、应用修改、
运行测试并记录可审计 trace。因此，模型说“已完成”不会直接结束任务：只有当前代码上的
pytest、编译或 lint 等确定性证据满足要求，任务才会标记为 `VERIFIED`。

## 可以做什么

```text
用户任务
  -> 理解仓库与相关符号
  -> 制定可见计划
  -> 读取/搜索/修改本地代码
  -> 执行测试、编译或 lint
  -> 根据失败继续修复
  -> 输出变更、验证证据与最终总结
```

### 1. 面向仓库的代码理解

- 扫描用户选择的 workspace，遵守 ignore 规则，提供文件浏览、文本搜索和安全读取。
- 针对 Python 项目，使用 AST 提取 class、function、method、import 与基础关系，生成紧凑的
  repository map；模型可以按 symbol 查找和读取局部代码，而不必反复塞入整个文件。
- `ContextManager` 把任务目标、成功标准、当前计划、repo map、相关代码和近期 observation
  分层管理；旧的长输出会被确定性压缩，避免多轮任务的 prompt 无限制增长。

### 2. 显式计划与多轮执行

- `AgentState` 保存任务、步骤号、计划、工具调用、变更代次、验证状态和终止原因；CLI、TUI
  与浏览器工作台只是这份状态的不同展示。
- `auto` 模式先由同一模型通过原生 `select_task_mode` 按完整语义选择只读或编码权限，
  不用本地关键词规则猜测任务类型。进入编码路径后，同一模型生成
  `goal / success criteria / steps / step status`，随后通过 `update_plan` 更新，不额外引入 Planner 或 Reviewer Agent。
- 支持连续 tool calling；正常无工具回答可结束，`max_steps`、取消、重复失败与无进展均有明确
  termination state，避免把未完成任务包装成成功。

### 3. 本地编码工具与安全控制

- 所有文件工具都在指定 workspace 内运行。路径会 canonicalize，越界路径和 symlink escape
  会被拒绝。
- 提供 `list_files`、`read_file`、`search_text`、`get_repo_map`、`search_symbol`、
  `read_symbol`、`run_command`、`create_file`、`write_file`、`apply_patch` 与 `git_diff`。
- 命令执行限制 cwd、timeout 与 stdout/stderr 长度，并对危险命令执行 policy；子进程不会无条件
  继承 API key 等敏感环境变量。
- 编辑优先使用多文件 exact-context patch。所有 hunk 会先校验；写入过程遇到失败会 rollback，
  不留下半应用状态。`git_diff` 可让模型和用户看到真实的改动。

### 4. 自验证与失败恢复

- 运行过的 pytest、compiler、lint 等命令会登记为 verification evidence。代码修改后，旧证据会
  因 mutation generation 改变而失效。
- Agent 会保存已发现的测试命令，修改后优先建议 targeted test，并在最终完成前执行合适的最终
  验证。
- 测试或工具失败不会让 loop 崩溃，而是作为 observation 反馈给模型；连续相同失败或多轮无进展
  会产生 recovery hint，要求重新检查假设、文件或测试范围。

### 5. 可观察、可恢复的人机交互

- 每一轮会产生脱敏 JSONL trace：`model_request`、`model_response`、`mode_selected`、`tool_start`、
  `tool_result`、`patch_applied`、`plan_updated`、`verification`、`recovery`、`final`。
- 提供一次性 CLI、多轮 REPL、全屏终端 TUI 和仅监听 `127.0.0.1` 的浏览器工作台。界面展示
  当前计划、有效执行摘要、文件变更、验证状态、耗时与可用的 token usage。
- session 与压缩后的本地状态可恢复；`/resume`、`/sessions`、`/steer` 支持长任务继续进行和
  接受用户指导。

## 如何运行

### 环境与安装

要求 Python 3.11+；建议安装 Git，以便 Agent 使用 `git_diff`。运行时模型通信只依赖官方
`openai` Python client。

```powershell
git clone https://github.com/huaijiao666/DBAgent.git
cd DBAgent
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
.\scripts\install-dba.ps1 -Install
```

安装一次后，可以在任意目标目录启动：

```powershell
cd C:\your\workspace
DBA
```

常用入口：

| 命令 | 用途 |
| --- | --- |
| `DBA` | 默认多轮终端交互；workspace 就是启动时当前目录 |
| `DBA --ui tui` | 全屏终端界面 |
| `DBA --ui web` | 本机浏览器工作台 |
| `dbagent "<task>"` | 一次性执行一个任务 |
| `dbagent-smoke` | 只验证模型连通性，会消耗一次模型请求 |

默认不会自动向上切换到项目根。只有显式指定 `--discover-workspace` 时，才会从启动目录向上
寻找项目根；这避免误把相邻目录纳入工具权限范围。

### 配置模型

默认模型为 `gpt-5.6-sol`，可用 `DBAGENT_MODEL` 与 `DBAGENT_REASONING_EFFORT` 覆盖。真实
provider 凭据只能放在本机、被 Git 忽略的 `config.toml`、`api_key.txt` 或进程环境变量中；
绝不要写入源码、README、trace、session 或测试。

配置优先级、OpenAI/Chat Completions compatibility、DeepSeek 模型切换、`/model`、`/resume`、
`/steer`、TUI 与浏览器工作台的完整使用方式见 [使用与配置](docs/usage.md)。

### 运行测试

```powershell
python -m pytest -q
```

当前测试覆盖 tool registry、workspace sandbox、symlink、命令 timeout/截断、patch atomicity、
context compaction、AST repo map、plan transition、termination、verification recovery、session、
trace 与浏览器控制器。

## 演示与可验证性

推荐视频任务是 Snake Arena 的两段独立真实运行：先在空工作区创建 React + FastAPI + SQLite
的网页贪吃蛇，再在同一项目上增加“疯狂模式”（障碍、限时金色食物、加速、减速道具和按模式
区分的排行榜）。第一段展示从零创建前端、后端、数据库与联调；第二段展示 Agent 读取既有
实现、复用经典模式、跨文件修改、验证兼容性。完整 prompt、镜头表和剪辑规则保存在
`docs/video-demo-script.md`；公开文档导航保持只列项目资料，不把视频脚本混入其中。

仓库中的 `tests/fixtures/taskboard_repo` 仍保留为快速、可重复的端到端测试 fixture，用于验证
repository/service/CLI/tests 的跨层 feature 和 self-verifying loop。已观察到的任务记录见
[端到端任务记录](docs/e2e-demo.md)。

## 其它说明

- 项目的模型 API 通信、Agent loop、上下文、tool schema/dispatch、执行、安全、patch、验证与
  trace 均由本项目代码实现；本地工具不是服务端托管的 shell、文件或 patch 工具。
- `run_command` 提供 policy、timeout、输出上限与环境清理，但它不是 OS 级隔离环境；不要把
  不可信或高风险 workspace 交给它执行。
- Python repo map 是稳定、轻量的 AST 索引，不试图证明动态 import、反射或所有运行时语义。
- 模型能力、网络延迟和第三方 provider 的 function-call 兼容性仍会影响任务成功率。遇到预算
  耗尽时，Agent 应诚实报告 `INCOMPLETE` 并保留 trace，而不是宣称已经完成。

## 文档导航

- [架构与运行时状态](docs/architecture.md)：模块边界、状态所有权、context、验证与 trace。
- [使用与配置](docs/usage.md)：安装、provider、本地凭据、命令、会话和 UI。
- [端到端任务记录](docs/e2e-demo.md)：三个独立 fixture 任务及已观察到的结果。
- [提交版 README.txt](submission/README.txt)：适合提交材料的 1000 字内摘要。
