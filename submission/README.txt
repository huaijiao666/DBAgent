软件工程专业推免项目：DataBaseAgent（DBAgent）

仓库地址：<提交前替换为你的公开 Git 仓库 URL>

DataBaseAgent（DBAgent）是从零实现的本地编程智能体，DB 指 Database；目标是完成通用代码仓库中的真实编程任务。

运行环境：Python 3.11+，建议安装 Git。执行 `python -m pip install -e ".[dev]"` 安装依赖；Windows 可执行 `./scripts/install-dba.ps1 -Install`，然后在任意工作目录运行 `DBA`。首次使用时，在项目根目录复制安全模板为被 Git 忽略的本地配置文件，并自行填写 provider 凭据；密钥不进入代码、README、trace 或会话记录。

主要功能：模型通过原生 function calling 请求本地工具；智能体可在选定 workspace 内浏览文件、搜索文本、提取 Python AST 符号、维护结构化计划、以原子 patch 修改多文件、运行带超时和输出上限的本地命令、查看 Git diff，并以 pytest/编译/lint 等确定性结果验证任务。达到步数上限或缺少验证证据时明确报告未完成，不把模型口头声明当作成功。

工程实现：项目未使用 LangChain、LlamaIndex、OpenAI Agents SDK、AutoGen、CrewAI 等 Agent 框架，也未使用 Code Interpreter、hosted shell、Files API 等托管执行能力。上下文、工具注册与分发、sandbox、patch 原子性、会话恢复、验证、重试和 JSONL trace 均由本地 Python 代码实现。提供 CLI、终端 TUI 与仅绑定本机回环地址的浏览器工作台。
