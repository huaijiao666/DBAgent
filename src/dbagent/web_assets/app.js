const token = new URLSearchParams(location.search).get("token") || "";
const $ = (id) => document.getElementById(id);
let latestEvent = 0;
let status = null;
let changes = { files: [], summary: { files: 0, added: 0, deleted: 0 } };
let liveText = "";
let repositoryEntries = [];
let sessionProcess = null;
let workspaceRefreshTimer = null;
const previews = { repo: { tabs: [], active: null }, change: { tabs: [], active: null } };

const api = async (path, options = {}) => {
  const response = await fetch(path, { ...options, headers: { "X-DBAgent-Token": token, ...(options.headers || {}) } });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "本地请求失败");
  return data;
};
const toast = (message) => { const node = $("toast"); node.textContent = message; node.classList.add("show"); setTimeout(() => node.classList.remove("show"), 3500); };
const escapeHtml = (text) => String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#039;" }[c]));
const labels = { idle: "空闲", running: "执行中", completed: "已完成", verified: "已验证", incomplete: "未完成", error: "错误", aborted: "已停止", blocked: "已阻塞", not_run: "尚未运行", passed: "通过", failed: "失败", in_progress: "进行中", pending: "待执行" };
const zh = (value) => labels[String(value ?? "").toLowerCase()] || String(value ?? "—");
const toolLabel = (value) => ({ list_files: "查看项目结构", read_file: "阅读相关代码", search_text: "搜索相关引用", get_repo_map: "分析仓库结构", search_symbol: "定位代码符号", read_symbol: "阅读目标实现", select_task_mode: "语义判断任务类型", apply_patch: "应用代码修改", create_file: "创建项目文件", write_file: "写入项目文件", run_command: "运行本地检查", git_diff: "复核实际改动", update_plan: "更新任务计划" }[String(value ?? "")] || "本地操作");
const apiModeLabel = (value) => value === "chat_completions" ? "兼容对话接口" : value === "responses" ? "Responses 接口" : String(value ?? "本地接口");
const providerLabel = (value) => value === "configured" ? "已配置服务" : String(value ?? "本地模型服务");
const pathName = (path) => String(path).split("/").at(-1);
const taskExcerpt = (value, limit = 88) => { const compact = String(value ?? "").replace(/\s+/g, " ").trim(); return compact.length <= limit ? compact : `${compact.slice(0, Math.max(1, limit - 1)).trimEnd()}…`; };
function setSessionTitle(title, prefix = "会话") { const node = $("session-title"); const compact = taskExcerpt(title, 58); node.textContent = `${prefix}：${compact || "未命名会话"}`; node.title = String(title || ""); }

function renderMarkdown(text) {
  const blocks = [];
  const escaped = escapeHtml(String(text ?? "").replace(/\r\n?/g, "\n")).replace(/```([^\n]*)\n([\s\S]*?)```/g, (_m, language, code) => `\u0000BLOCK${blocks.push(`<pre><code data-language="${escapeHtml(language.trim())}">${code.replace(/\n$/, "")}</code></pre>`) - 1}\u0000`);
  const inline = (value) => value.replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>").replace(/\*([^*]+)\*/g, "<em>$1</em>");
  let html = ""; let list = null;
  const closeList = () => { if (list) html += `</${list}>`; list = null; };
  for (const line of escaped.split("\n")) {
    const heading = line.match(/^(#{1,3})\s+(.+)$/); const bullet = line.match(/^[-*+]\s+(.+)$/); const ordered = line.match(/^\d+\.\s+(.+)$/);
    if (/^\u0000BLOCK\d+\u0000$/.test(line)) { closeList(); html += line; }
    else if (heading) { closeList(); html += `<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`; }
    else if (bullet || ordered) { const type = ordered ? "ol" : "ul"; if (list !== type) { closeList(); list = type; html += `<${type}>`; } html += `<li>${inline((bullet || ordered)[1])}</li>`; }
    else { closeList(); html += line.trim() ? `<p>${inline(line)}</p>` : "<div class=markdown-gap></div>"; }
  }
  closeList(); return html.replace(/\u0000BLOCK(\d+)\u0000/g, (_m, index) => blocks[Number(index)] || "");
}
function eventTitle(event) { return ({ run_started: "任务开始", tool_start: "调用工具", tool_result: "工具结果", model_request: "请求模型", model_response: "模型响应", mode_selected: "语义路由", model_error: "模型错误", context_compacted: "压缩上下文", verification: "验证", final: "最终状态", plan_updated: "更新计划", patch_applied: "应用补丁", recovery: "恢复建议", browser_run_error: "运行错误", browser_run_finished: "任务结束", assistant_update: "执行说明", user_steering_queued: "已收到指导", user_steering_applied: "已采纳指导", user_follow_up_started: "继续会话", session_created: "新建会话" }[event] || "执行事件"); }
function eventStyle(event, payload) { if (event === "verification") return "verify"; if (event === "model_error" || event === "browser_run_error" || payload?.success === false) return "error"; if (["patch_applied", "plan_updated"].includes(event)) return "implement"; return event === "tool_start" && payload?.phase === "inspect" ? "inspect" : ""; }
function eventDetail(event, payload) {
  if (event === "run_started") return `${({ code: "编码", ask: "问答", auto: "自动" }[payload.mode] || "自动")}模式 · ${payload.task || ""}`;
  if (event === "tool_start") return `${payload.intent || "执行本地操作"}${payload.plan_step ? ` · 计划 ${payload.plan_step}` : ""}`;
  if (event === "tool_result") return payload.success ? (payload.return_code !== undefined ? `本地检查结束 · 返回码=${payload.return_code}` : "本地操作已完成。") : `本地操作未成功；错误证据已反馈给 Agent 继续处理。`;
  if (event === "model_request") return `上下文 ${payload.context_usage?.approximate_tokens ?? "?"}~Token · 可用工具 ${payload.tools?.length ?? 0} 个`;
  if (event === "model_response") return `模型已响应${payload.function_call_count ? "，将继续执行本地操作" : "。"}${payload.usage?.total_tokens ? ` · ${payload.usage.total_tokens} Token` : ""}`;
  if (event === "mode_selected") return `已根据完整需求语义选择${payload.mode === "code" ? "编码" : "只读问答"}权限${payload.source === "fallback" ? "（供应方协议异常后的安全降级）" : ""}。`;
  if (event === "model_error") return `模型请求暂时失败 · 第 ${payload.attempt ?? "?"}/${payload.max_attempts ?? "?"} 次${payload.will_retry ? "，正在重试" : "，任务将安全结束"}`;
  if (event === "context_compacted") return `保留 ${payload.recent_observations ?? "?"} 条近期观察 · 压缩 ${payload.compacted_observations ?? "?"} 条旧观察`;
  if (event === "verification") return `${payload.kind || "检查"} · ${zh(payload.status)} · 返回码=${payload.return_code ?? "?"}`;
  if (event === "final") return `${zh(payload.status)} · 验证 ${zh(payload.verification_status)}`;
  if (event === "plan_updated") return `计划进度：${payload.completed_steps ?? 0}/${payload.total_steps ?? "?"}`;
  if (event === "run_abort_requested") return "已请求停止；当前本地命令或 API 请求结束后将安全退出。";
  if (event === "browser_run_error") return "模型或本地任务异常结束；请查看最终结果与本地 trace。";
  if (event === "user_steering_applied") return "追加的指导已进入下一轮模型上下文。";
  if (event === "user_follow_up_started") return "已在当前本地会话中启动后续任务。";
  if (event === "assistant_update") return payload.text || "模型正在说明下一步。";
  return JSON.stringify(payload || {});
}

function renderStatus(next) {
  status = next; const run = next.run || {}; const step = Number(next.current_step ?? run.step ?? 0); const maxSteps = Number(next.max_steps ?? run.max_steps ?? 0);
  $("connection").textContent = `本地服务 · ${next.active ? "Agent 正在执行" : "就绪"}`; $("connection-dot").classList.add("connected"); $("workspace-path").value = next.workspace; $("max-steps").value = next.max_steps;
  $("live-label").textContent = next.active ? "● 运行中" : "空闲"; $("live-label").classList.toggle("running", next.active); $("run-clock").textContent = `${Number(next.elapsed_seconds || 0).toFixed(1)}s`;
  $("metric-step").textContent = `${step} / ${maxSteps || "—"}`; $("metric-tool").textContent = next.current_tool ? toolLabel(next.current_tool) : "—"; $("metric-tokens").textContent = next.token_usage?.total_tokens ?? "—"; $("metric-context").textContent = next.context_usage?.approximate_tokens ?? "—";
  $("current-action").textContent = next.active ? (next.current_tool ? `正在${toolLabel(next.current_tool)}…` : eventTitle(next.last_event || "model_request")) : (run.status ? `本次任务：${zh(run.status)}` : "等待任务");
  $("activity-summary").textContent = next.active ? `第 ${step}/${maxSteps} 步 · ${next.current_tool ? toolLabel(next.current_tool) : "模型分析"}` : (run.status ? `任务${zh(run.status)}，点击展开执行摘要` : "等待任务");
  const session = next.session || {}; setSessionTitle(session.title || "未命名会话"); $("provider-context").textContent = `${providerLabel(next.provider)} · ${apiModeLabel(next.api_mode)} · 本地工作区边界`; $("session-summary").innerHTML = `<strong title="${escapeHtml(session.title || "未命名会话")}">${escapeHtml(taskExcerpt(session.title || "未命名会话", 62))}</strong><br><code>${escapeHtml(session.id || "")}</code><br>${next.active ? `第 ${step}/${maxSteps} 步 · ${escapeHtml(next.current_tool ? toolLabel(next.current_tool) : "等待模型")}` : `${escapeHtml(zh(run.status))} · 验证 ${escapeHtml(zh(session.verification || run.verification))}`}`;
  populateSettings(next); renderPlan(next.plan); renderRun(run, next); if (next.last_error) toast(next.last_error);
}
function populateSettings(next) {
  const model = $("model"); if (!model.dataset.ready) { model.innerHTML = next.model_options.map((item) => `<option value="${escapeHtml(item.alias)}">${escapeHtml(item.alias)} · ${escapeHtml(item.model)}</option>`).join(""); model.dataset.ready = "1"; }
  const matching = [...model.options].find((item) => item.textContent.includes(next.model)); if (matching) model.value = matching.value;
  const reasoning = $("reasoning"); if (!reasoning.dataset.ready) { reasoning.innerHTML = next.reasoning_options.map((item) => `<option value="${item}">${item}</option>`).join(""); reasoning.dataset.ready = "1"; } reasoning.value = next.reasoning_effort;
  $("run").disabled = next.active; $("steer").disabled = false; $("steer").textContent = next.active ? "发送指导" : "继续执行"; $("abort").disabled = !next.active;
}
function renderPlan(plan) {
  const list = $("plan-list"); if (!plan) { list.innerHTML = '<div class="muted">DBAgent 创建计划后会在此保留，直到新的计划生成。</div>'; $("plan-count").textContent = "—"; return; }
  const steps = Array.isArray(plan.steps) ? plan.steps : []; const completed = steps.filter((item) => item.status === "completed").length; $("plan-count").textContent = `${completed} / ${steps.length} 已完成`;
  list.innerHTML = `<div class="plan-goal">${escapeHtml(plan.goal || "未说明目标")}</div>` + steps.map((item) => `<div class="plan-item"><span class="plan-marker ${escapeHtml(item.status)}">${({ completed: "✓", in_progress: "›", blocked: "!" }[item.status] || "·")}</span><div>${escapeHtml(item.description || item.step_id || "未命名步骤")}<small>${escapeHtml(item.step_id || item.id || "step")} · ${escapeHtml(zh(item.status))}</small></div></div>`).join("");
}
function renderRun(run, live) {
  const verification = live.latest_verification?.status || run.verification || "not_run"; $("verification-badge").textContent = zh(verification); $("verification-badge").className = `status-badge ${verification}`;
  const evidence = live.latest_verification || {}; const command = Array.isArray(evidence.command) ? evidence.command.join(" ") : "";
  $("verification-detail").innerHTML = `<div class="metric"><span>任务状态</span><strong>${escapeHtml(zh(run.status))}</strong></div><div class="metric"><span>确定性证据</span><strong>${escapeHtml(zh(verification))}</strong></div>${evidence.kind ? `<div class="metric"><span>检查类型</span><strong>${escapeHtml(String(evidence.kind))}${evidence.return_code !== null && evidence.return_code !== undefined ? ` · 返回码 ${escapeHtml(String(evidence.return_code))}` : ""}</strong></div>` : ""}${command ? `<div class="verification-command"><span>最近命令</span><code>${escapeHtml(command)}</code></div>` : ""}`; renderChanges();
  if (run.final_answer) { appendFinalAnswer(run.final_answer, run, live); collapseSessionProcess(run, live); if (!live.active) $("activity-details").open = false; }
}
function renderTree(tree) {
  repositoryEntries = (tree.entries || []).filter((entry) => entry.kind === "file");
  const node = $("file-tree"); node.innerHTML = tree.entries.length ? tree.entries.map((entry) => { const depth = entry.path.split("/").length - 1; const icon = entry.kind === "directory" ? "▸" : "·"; return entry.kind === "file" ? `<button class="file-row file" data-path="${escapeHtml(entry.path)}" style="padding-left:${4 + depth * 15}px" title="预览 ${escapeHtml(entry.path)}">${icon} ${escapeHtml(pathName(entry.path))}</button>` : `<div class="file-row folder" style="padding-left:${4 + depth * 15}px">${icon} ${escapeHtml(pathName(entry.path))}</div>`; }).join("") + (tree.truncated ? '<div class="muted">文件树最多显示 350 项。</div>' : "") : '<div class="muted">工作区为空。</div>';
  node.querySelectorAll(".file-row.file").forEach((button) => button.addEventListener("click", () => openPreview("repo", button.dataset.path)));
}
function renderChanges() {
  const summary = changes.summary || {}; const files = changes.files || []; $("changes-count").textContent = `${summary.files ?? files.length} 个文件 · +${summary.added || 0} / −${summary.deleted || 0}`;
  $("changes-list").innerHTML = files.length ? files.map((file) => `<button class="change-row" data-path="${escapeHtml(file.path)}"><span>${escapeHtml(pathName(file.path))}</span><small>${escapeHtml(file.status)} · +${file.added} / −${file.deleted}</small></button>`).join("") : '<div class="muted">尚未报告本地变更。</div>';
  $("changes-list").querySelectorAll(".change-row").forEach((button) => button.addEventListener("click", () => openPreview("change", button.dataset.path)));
}
async function refreshChanges() { try { changes = await api("/api/changes"); renderChanges(); } catch (_) {} }

function ensurePreview(kind) { $("app").classList.add(kind === "repo" ? "show-repo-preview" : "show-change-preview"); applyLayout(); }
function closePreview(kind) { previews[kind] = { tabs: [], active: null }; $("app").classList.remove(kind === "repo" ? "show-repo-preview" : "show-change-preview"); renderPreview(kind); applyLayout(); }
function renderPreview(kind) {
  const preview = previews[kind]; const tab = preview.tabs.find((item) => item.path === preview.active); const tabs = $(`${kind}-tabs`); tabs.innerHTML = preview.tabs.map((item) => `<button class="file-tab ${item.path === preview.active ? "active" : ""}" data-path="${escapeHtml(item.path)}" title="${escapeHtml(item.path)}"><span>${escapeHtml(pathName(item.path))}</span><b data-close="${escapeHtml(item.path)}" aria-label="关闭">×</b></button>`).join("");
  $(`${kind}-code-meta`).textContent = tab ? `${tab.path} · ${tab.line_count || 0} 行${tab.truncated ? " · 预览已截断" : ""}` : ""; const code = $(`${kind}-code`); code.innerHTML = tab?.error ? `<span class="preview-error">${escapeHtml(tab.error)}</span>` : escapeHtml(tab?.content || ""); code.dataset.language = tab?.language || "text";
  tabs.querySelectorAll(".file-tab").forEach((button) => button.addEventListener("click", (event) => { const closing = event.target.closest("b")?.dataset.close; if (closing) { event.stopPropagation(); preview.tabs = preview.tabs.filter((item) => item.path !== closing); preview.active = preview.tabs.at(-1)?.path || null; if (!preview.tabs.length) return closePreview(kind); } else { preview.active = button.dataset.path; } renderPreview(kind); }));
}
async function openPreview(kind, path) {
  if (!path) return; ensurePreview(kind); const preview = previews[kind]; if (preview.tabs.some((item) => item.path === path)) { preview.active = path; renderPreview(kind); return; }
  try { preview.tabs.push(await api(`/api/file?path=${encodeURIComponent(path)}`)); } catch (error) { preview.tabs.push({ path, error: `无法读取当前文件：${error.message}` }); }
  preview.active = path; renderPreview(kind);
}
function beginSessionProcess() {
  const log = $("conversation-log");
  sessionProcess = document.createElement("article");
  sessionProcess.className = "message agent agent-turn";
  sessionProcess.innerHTML = '<div class="agent-avatar">DB</div><div class="agent-turn-content"><div class="agent-turn-header"><strong>DBAgent</strong><span class="turn-state">正在理解任务</span></div><div class="turn-progress"><p class="turn-current" data-key="initial">我先确认任务目标与当前工作区。</p></div><details class="turn-execution" hidden><summary><strong>查看执行摘要</strong><span>0 项</span></summary><div class="session-process-body"></div></details><div class="turn-final" hidden></div></div>';
  log.append(sessionProcess); log.scrollTop = log.scrollHeight;
}
function appendSessionProcess(title, detail, kind = "") {
  if (!sessionProcess) beginSessionProcess();
  const state = sessionProcess.querySelector(".turn-state");
  // The card opens with the run-start wording. Update that line when the
  // matching trace arrives instead of rendering a duplicate progress note.
  const progressKey = title === "正在理解任务" ? "initial" : `event:${title}`;
  appendProgressNote(detail, progressKey); state.textContent = title; delete sessionProcess.dataset.streamStep;
  const body = sessionProcess.querySelector(".session-process-body"); const key = `${title}:${detail}`;
  if (![...body.children].some((line) => line.dataset.key === key)) {
    const line = document.createElement("div"); line.className = `session-process-line ${kind}`; line.dataset.key = key;
    line.innerHTML = `<span>${escapeHtml(title)}</span><small>${escapeHtml(detail)}</small>`; body.append(line);
    // Retain a meaningful sequence of decisions, edits, and verification
    // evidence for the post-run execution summary. Raw tool mechanics remain
    // available in the lower trace rather than consuming this space.
    while (body.children.length > 12) body.firstElementChild.remove();
  }
  sessionProcess.querySelector(".turn-execution summary span").textContent = `${body.children.length} 项`;
  $("conversation-log").scrollTop = $("conversation-log").scrollHeight;
}
function appendProgressNote(text, key, append = false) {
  const progress = sessionProcess.querySelector(".turn-progress"); let note = progress.querySelector(`[data-key="${CSS.escape(key)}"]`);
  if (!note) { note = document.createElement("p"); note.className = "turn-current"; note.dataset.key = key; progress.append(note); }
  note.textContent = append ? `${note.textContent}${text}`.slice(-1600) : text;
  while (progress.children.length > 6) progress.firstElementChild.remove();
  return note;
}
function appendConversationText(text, step, complete = false) {
  if (!text) return; if (!sessionProcess) beginSessionProcess();
  const state = sessionProcess.querySelector(".turn-state");
  const streamStep = String(step ?? "");
  const incremental = !complete && sessionProcess.dataset.streamStep === streamStep;
  sessionProcess.dataset.streamStep = streamStep;
  // A streamed delta is followed by one complete assistant_update for the
  // same model turn.  Replace that note rather than appending it a second
  // time; non-streaming providers still display the complete explanation.
  appendProgressNote(text, `stream:${streamStep}`, incremental);
  state.textContent = "正在分析";
  $("conversation-log").scrollTop = $("conversation-log").scrollHeight;
}
function changedFileText(payload) {
  const files = Array.isArray(payload.changed_files) ? payload.changed_files : (payload.path ? [payload.path] : []);
  const names = files.slice(0, 3).map((path) => `「${pathName(path)}」`);
  const extra = files.length > names.length ? ` 等 ${files.length} 个文件` : "";
  const deltas = Array.isArray(payload.line_changes) ? payload.line_changes : [];
  const added = deltas.reduce((total, item) => total + (Number(item.added) || 0), 0);
  const deleted = deltas.reduce((total, item) => total + (Number(item.deleted) || 0), 0);
  const delta = deltas.length ? `（+${added} / −${deleted} 行）` : "";
  return `${names.join("、") || "相关文件"}${extra}${delta}`;
}
function verificationText(payload) {
  const command = Array.isArray(payload.command) ? payload.command.join(" ") : "本地检查";
  const compact = command.length > 100 ? `${command.slice(0, 97)}…` : command;
  if (payload.status === "passed") return `已通过「${compact}」，当前修改获得了本地验证。`;
  if (payload.timed_out) return `「${compact}」超时；我会依据已有证据调整检查或实现。`;
  return `「${compact}」未通过；我会根据可复现错误继续定位并修复。`;
}
function conversationMilestone(name, payload) {
  if (name === "run_started") return ["正在梳理任务", "我会先确认现有实现和可用的本地验证方式，再推进可交付的结果。", "inspect"];
  if (name === "plan_updated" && payload.current_step_description) return ["执行计划已更新", `当前重点：${payload.current_step_description}`, "plan"];
  if (name === "tool_result" && payload.success && ["apply_patch", "create_file", "write_file"].includes(payload.tool_name)) return ["项目文件已变更", `已处理 ${changedFileText(payload)}；下一步将用本地检查确认这次变更。`, "implement"];
  if (name === "tool_result" && payload.success === false && ["apply_patch", "create_file", "write_file", "run_command"].includes(payload.tool_name)) return ["发现需要处理的问题", `本地${toolLabel(payload.tool_name)}没有成功；错误证据已保留，正在据此调整后续操作。`, "error"];
  if (name === "verification") return [payload.status === "passed" ? "本地验证已通过" : "发现可复现的问题", verificationText(payload), payload.status === "passed" ? "verify" : "error"];
  if (name === "user_steering_applied") return ["已纳入追加目标", "新的要求已成为当前任务的必做范围；不会只在最终总结中说明未完成。", "thinking"];
  if (name === "model_error") return ["正在重试请求", payload.will_retry ? "模型请求暂时失败，正在自动重试。" : "模型请求失败，当前任务将安全结束。", "error"];
  if (name === "browser_run_error") return ["任务已停止", payload.error || "本地任务因错误停止。", "error"];
  if (name === "run_abort_requested") return ["正在停止", "我会在下一个安全边界停止执行。", "error"];
  return null;
}
function collapseSessionProcess(run, live) {
  if (!sessionProcess || sessionProcess.dataset.collapsed) return;
  const details = sessionProcess.querySelector(".turn-execution"); const verification = live.latest_verification?.status || run.verification;
  details.hidden = false; details.open = false; details.querySelector("summary strong").textContent = "查看执行摘要";
  details.querySelector("summary span").textContent = `${zh(run.status)} · 验证 ${zh(verification)}`;
  sessionProcess.dataset.collapsed = "true";
}
function appendFinalAnswer(text, run = {}, live = {}) {
  if (!sessionProcess) beginSessionProcess();
  if (sessionProcess.dataset.finalKey === text) return;
  const verification = live.latest_verification?.status || run.verification || "not_run"; const changed = live.changed_files || run.changed_files || [];
  const final = sessionProcess.querySelector(".turn-final"); final.hidden = false;
  final.innerHTML = `<div class="run-summary"><span>${escapeHtml(zh(run.status))}</span><span>验证：${escapeHtml(zh(verification))}</span><span>变更：${changed.length} 个文件</span></div><div class="markdown-body">${renderMarkdown(text)}</div>`;
  sessionProcess.dataset.finalKey = text; sessionProcess.querySelector(".turn-progress").hidden = true;
  sessionProcess.querySelector(".turn-state").textContent = "最终结果"; collapseSessionProcess(run, live);
  $("conversation-log").scrollTop = $("conversation-log").scrollHeight;
}
function appendUserMessage(text, guidance = false) { const log = $("conversation-log"); const node = document.createElement("article"); node.className = "message user"; node.innerHTML = `<div class="agent-avatar">Y</div><div><strong>你${guidance ? " · 追加指导" : ""}</strong><p>${escapeHtml(text)}</p></div>`; log.append(node); log.scrollTop = log.scrollHeight; }
function appendHistoricalAnswer(text) { const log = $("conversation-log"); const node = document.createElement("article"); node.className = "message agent agent-turn restored-turn"; node.innerHTML = `<div class="agent-avatar">DB</div><div class="agent-turn-content"><div class="agent-turn-header"><strong>DBAgent</strong><span class="turn-state">历史结果</span></div><div class="turn-final"><div class="markdown-body">${renderMarkdown(text)}</div></div></div>`; log.append(node); }
function appendHistoricalExecution(entries) {
  if (!entries.length) return;
  const log = $("conversation-log"); const node = document.createElement("article"); node.className = "message agent agent-turn restored-turn";
  const lines = entries.map((entry) => `<div class="session-process-line ${escapeHtml(entry.kind || "")}"><span>${escapeHtml(entry.title || "已恢复本地记录")}</span><small>${escapeHtml(entry.detail || "")}</small></div>`).join("");
  node.innerHTML = `<div class="agent-avatar">DB</div><div class="agent-turn-content"><div class="agent-turn-header"><strong>DBAgent</strong><span class="turn-state">历史执行记录</span></div><details class="turn-execution"><summary><strong>查看已恢复的执行摘要</strong><span>${entries.length} 项</span></summary><div class="session-process-body">${lines}</div></details></div>`;
  log.append(node);
}
function resetConversation(message = "这是一个新的本地会话。描述任务后，我会先建立计划，再用本地工具完成与验证。") { $("conversation-log").innerHTML = `<article class="welcome"><div class="agent-avatar">DB</div><div><strong>DBAgent</strong><p>${escapeHtml(message)}</p></div></article>`; clearActivity(); }
function restoreSessionHistory(history) {
  const conversation = Array.isArray(history?.conversation) ? history.conversation : []; const execution = Array.isArray(history?.execution) ? history.execution : [];
  if (!conversation.length) { resetConversation("已恢复一个尚未开始任务的本地会话。"); return; }
  $("conversation-log").innerHTML = ""; sessionProcess = null;
  conversation.forEach((message) => { if (message?.role === "user") appendUserMessage(message.content || ""); else if (message?.role === "assistant") appendHistoricalAnswer(message.content || ""); });
  appendHistoricalExecution(execution); $("conversation-log").scrollTop = $("conversation-log").scrollHeight; clearActivity();
}
function appendLiveText(text) { if (!text) return; liveText = `${liveText}${text}`.slice(-12000); const box = $("live-process-output"); box.hidden = false; box.textContent = liveText; box.scrollTop = box.scrollHeight; }
function clearActivity() { liveText = ""; sessionProcess = null; $("live-process-output").textContent = ""; $("live-process-output").hidden = true; $("timeline").innerHTML = '<div class="empty-state">正在准备执行证据…</div>'; $("activity-details").open = true; }
function isUsefulExecutionEvent(name, payload) { return ["run_started", "mode_selected", "plan_updated", "patch_applied", "verification", "model_error", "final", "browser_run_error", "browser_run_finished", "user_steering_queued", "user_steering_applied", "run_abort_requested"].includes(name) || (name === "tool_result" && payload.success === false); }
function isWorkspaceMutation(name, payload) { return name === "patch_applied" || (name === "tool_result" && payload.success === true && ["apply_patch", "create_file", "write_file"].includes(payload.tool_name)); }
function scheduleWorkspaceRefresh() {
  if (workspaceRefreshTimer !== null) return;
  workspaceRefreshTimer = window.setTimeout(async () => {
    workspaceRefreshTimer = null;
    try {
      const [tree, changed] = await Promise.all([api("/api/tree"), api("/api/changes")]);
      changes = changed; renderTree(tree); renderChanges();
    } catch (_) {}
  }, 120);
}
function addEvent(event) {
  const trace = event.event === "trace" ? event.payload.trace : null; const name = trace?.event || event.event; const payload = trace?.payload || event.payload || {};
  if (event.event === "workspace_changed") { resetConversation("已切换工作区，并自动创建了一个新的本地会话。旧工作区的会话仍保存在原目录。 "); $("session-title").textContent = "新建会话"; return; }
  if (event.event === "session_created") { resetConversation(); $("session-title").textContent = "新建会话"; return; }
  if (event.event === "run_started") { appendUserMessage(payload.task); setSessionTitle(payload.task); clearActivity(); }
  const milestone = conversationMilestone(name, payload); if (milestone) appendSessionProcess(...milestone);
  // The runtime has already filtered assistant_update down to a concrete
  // finding or a reasoned decision. Put it in the one DBAgent turn card;
  // ordinary token deltas and tool mechanics stay out of the conversation.
  if (name === "assistant_update" && payload.text) { appendSessionProcess("分析判断", payload.text, "thinking"); return; }
  if (name === "model_stream") return;
  if (isWorkspaceMutation(name, payload)) scheduleWorkspaceRefresh();
  if (!isUsefulExecutionEvent(name, payload)) { if (name === "tool_result") refreshChanges(); return; }
  const timeline = $("timeline"); const line = document.createElement("article"); line.className = `event ${eventStyle(name, payload)}`; line.innerHTML = `<div class="event-title"><span>${escapeHtml(eventTitle(name))}</span><span>${new Date(event.timestamp * 1000).toLocaleTimeString()}</span></div><div class="event-detail">${escapeHtml(eventDetail(name, payload))}</div>`; timeline.querySelector(".empty-state")?.remove(); timeline.append(line); timeline.scrollTop = timeline.scrollHeight;
  if (name === "tool_result" || name === "patch_applied" || name === "browser_run_finished") refreshChanges();
}
async function refresh() { const [next, tree, changed] = await Promise.all([api("/api/status"), api("/api/tree"), api("/api/changes")]); changes = changed; renderStatus(next); renderTree(tree); }
async function configure() { renderStatus(await api("/api/settings", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: $("model").value, reasoning_effort: $("reasoning").value, max_steps: Number($("max-steps").value) }) })); }
function events() { const source = new EventSource(`/api/events?token=${encodeURIComponent(token)}&after=${latestEvent}`); source.addEventListener("update", (message) => { const event = JSON.parse(message.data); latestEvent = event.id; addEvent(event); api("/api/status").then(renderStatus).catch(() => {}); }); source.onerror = () => { $("connection").textContent = "正在重连本地服务…"; }; }
function applyFreshWorkspace(next) { renderStatus(next); resetConversation("已切换工作区，并自动创建了一个新的本地会话。旧工作区的会话仍保存在原目录。"); $("session-title").textContent = "新建会话"; $("task").placeholder = "描述你想完成的编程任务…"; }
async function chooseWorkspace() { try { const result = await api("/api/workspace-picker", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) }); if (!result.selected) return; applyFreshWorkspace(result.status); closePreview("repo"); closePreview("change"); await refresh(); toast("已切换工作区，并新建了空白会话。"); } catch (error) { toast(error.message); } }
async function openSessions() { const dialog = $("sessions-dialog"); $("sessions-list").innerHTML = '<div class="muted">正在读取会话…</div>'; dialog.showModal(); try { const result = await api("/api/sessions"); const sessions = result.sessions || []; $("sessions-list").innerHTML = sessions.length ? sessions.map((item) => `<button class="session-row ${item.session_id === result.active_session_id ? "active" : ""}" data-session-id="${escapeHtml(item.session_id)}"><div><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.session_id)} · ${escapeHtml(item.updated_at)} · ${item.turns} 轮</small></div><span class="session-state">${escapeHtml(zh(item.status))}</span></button>`).join("") : '<div class="muted">这个工作区还没有可恢复会话。先运行一个任务，DBAgent 会自动保存检查点。</div>'; $("sessions-list").querySelectorAll(".session-row").forEach((button) => button.addEventListener("click", () => resumeSession(button.dataset.sessionId))); } catch (error) { $("sessions-list").innerHTML = `<div class="muted">无法读取会话：${escapeHtml(error.message)}</div>`; } }
async function resumeSession(sessionId) { try { const next = await api("/api/sessions/resume", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ session_id: sessionId }) }); $("sessions-dialog").close(); renderStatus(next); restoreSessionHistory(next.history); $("session-title").textContent = `已恢复：${next.session?.title || sessionId}`; $("task").placeholder = "已恢复本地上下文。输入“继续”或说明下一步…"; toast("已恢复历史对话、执行摘要、计划和验证状态。"); } catch (error) { toast(error.message); } }
async function newSession() { try { const next = await api("/api/sessions/new", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }); renderStatus(next); $("session-title").textContent = "新建会话"; $("task").placeholder = "描述你想完成的编程任务…"; resetConversation(); toast("已新建空白会话；旧会话仍可在“会话”中恢复。"); } catch (error) { toast(error.message); } }
$("apply-workspace").addEventListener("click", async () => { try { applyFreshWorkspace(await api("/api/workspace", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: $("workspace-path").value }) })); closePreview("repo"); closePreview("change"); await refresh(); toast("已切换工作区，并新建了空白会话。"); } catch (error) { toast(error.message); } });
$("choose-workspace").addEventListener("click", chooseWorkspace); $("new-session").addEventListener("click", newSession); $("sidebar-new-session").addEventListener("click", newSession); $("open-sessions").addEventListener("click", openSessions); $("sidebar-sessions").addEventListener("click", openSessions); $("close-sessions").addEventListener("click", () => $("sessions-dialog").close());
$("reload-tree").addEventListener("click", () => refresh().catch((error) => toast(error.message))); $("refresh").addEventListener("click", () => refresh().catch((error) => toast(error.message)));
$("run").addEventListener("click", async () => { const task = $("task").value.trim(); if (!task) return toast("请先描述一个编程任务。"); try { await configure(); $("task").value = ""; renderStatus(await api("/api/tasks", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ task, mode: $("mode").value }) })); } catch (error) { toast(error.message); } });
$("steer").addEventListener("click", async () => { const message = $("task").value.trim(); if (!message) return toast("请先输入要追加的指导。"); try { const next = await api("/api/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "steer", message }) }); renderStatus(next); if (next.steering_mode !== "follow_up") appendUserMessage(message, true); $("task").value = ""; toast(next.steering_mode === "follow_up" ? "已在当前会话中开始执行后续要求。" : "已提交；DBAgent 会在下一个安全边界执行这项要求。"); } catch (error) { toast(error.message); } });
$("abort").addEventListener("click", async () => { try { renderStatus(await api("/api/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "abort" }) })); toast("已请求停止，Agent 将在下一个安全边界退出。"); } catch (error) { toast(error.message); } });
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => { document.querySelectorAll(".tab,.inspector-panel").forEach((item) => item.classList.remove("active")); tab.classList.add("active"); $(`${tab.dataset.tab}-panel`).classList.add("active"); if (tab.dataset.tab === "changes") refreshChanges(); })); document.querySelectorAll(".close-preview").forEach((button) => button.addEventListener("click", () => closePreview(button.dataset.preview)));
function updateFileReferenceMenu() { const input = $("task"); const match = input.value.match(/(?:^|\s)@([^\s@]*)$/); const menu = $("file-reference-menu"); if (!match) { menu.hidden = true; return; } const query = match[1].toLowerCase(); const matches = repositoryEntries.filter((entry) => entry.path.toLowerCase().includes(query)).slice(0, 8); if (!matches.length) { menu.hidden = true; return; } menu.innerHTML = matches.map((entry) => `<button data-path="${escapeHtml(entry.path)}"><span>@${escapeHtml(entry.path)}</span><small>${escapeHtml(pathName(entry.path))}</small></button>`).join(""); menu.hidden = false; menu.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => { input.value = input.value.replace(/@[^\s@]*$/, `@${button.dataset.path} `); input.focus(); menu.hidden = true; })); }
$("task").addEventListener("input", updateFileReferenceMenu); $("task").addEventListener("blur", () => setTimeout(() => { $("file-reference-menu").hidden = true; }, 120)); $("task").addEventListener("keydown", (event) => { if (event.ctrlKey && event.key === "Enter") $("run").click(); if (event.key === "Escape") $("file-reference-menu").hidden = true; });
const layoutKey = "dbagent-layout-v3"; const layout = { sidebar: 300, inspector: 420, conversation: 250, repoPreview: 270, changePreview: 290 };
function limits(kind) { return kind === "repoPreview" || kind === "changePreview" ? [180, 520] : kind === "conversation" ? [200, 600] : kind === "inspector" ? [260, 560] : [220, 480]; }
function clamp(value, kind) { const [minimum, maximum] = limits(kind); return Math.min(maximum, Math.max(minimum, value)); }
function normalizeLayout() { if (window.innerWidth <= 920) return; const extra = ($("app").classList.contains("show-repo-preview") ? layout.repoPreview + 7 : 0) + ($("app").classList.contains("show-change-preview") ? layout.changePreview + 7 : 0) + 14; layout.sidebar = Math.min(480, Math.max(220, Math.min(layout.sidebar, window.innerWidth - layout.inspector - 320 - extra))); layout.inspector = Math.min(560, Math.max(260, Math.min(layout.inspector, window.innerWidth - layout.sidebar - 320 - extra))); }
function applyLayout() { normalizeLayout(); const root = document.documentElement; root.style.setProperty("--sidebar-width", `${layout.sidebar}px`); root.style.setProperty("--inspector-width", `${layout.inspector}px`); root.style.setProperty("--conversation-height", `${layout.conversation}px`); root.style.setProperty("--repo-preview-width", `${layout.repoPreview}px`); root.style.setProperty("--change-preview-width", `${layout.changePreview}px`); }
function saveLayout() { try { localStorage.setItem(layoutKey, JSON.stringify(layout)); } catch (_) {} }
function loadLayout() { try { const saved = JSON.parse(localStorage.getItem(layoutKey) || "{}"); Object.keys(layout).forEach((key) => { if (Number.isFinite(saved[key])) layout[key] = clamp(saved[key], key); }); } catch (_) {} applyLayout(); }
function startResize(kind, event) { if (event.button !== 0) return; event.preventDefault(); document.body.dataset.resizing = kind; const startX = event.clientX; const startY = event.clientY; const initial = layout[kind]; const move = (point) => { if (kind === "sidebar") layout.sidebar = clamp(initial + point.clientX - startX, kind); if (kind === "inspector") layout.inspector = clamp(initial - point.clientX + startX, kind); if (kind === "conversation") layout.conversation = clamp(initial + point.clientY - startY, kind); if (kind === "repoPreview") layout.repoPreview = clamp(initial + point.clientX - startX, kind); if (kind === "changePreview") layout.changePreview = clamp(initial - point.clientX + startX, kind); applyLayout(); }; const finish = () => { document.removeEventListener("pointermove", move); document.removeEventListener("pointerup", finish); document.removeEventListener("pointercancel", finish); delete document.body.dataset.resizing; saveLayout(); }; document.addEventListener("pointermove", move); document.addEventListener("pointerup", finish, { once: true }); document.addEventListener("pointercancel", finish, { once: true }); }
$("left-resizer").addEventListener("pointerdown", (event) => startResize("sidebar", event)); $("right-resizer").addEventListener("pointerdown", (event) => startResize("inspector", event)); $("center-resizer").addEventListener("pointerdown", (event) => startResize("conversation", event)); $("repo-preview-resizer").addEventListener("pointerdown", (event) => startResize("repoPreview", event)); $("change-preview-resizer").addEventListener("pointerdown", (event) => startResize("changePreview", event));
function nudgeResize(kind, delta) { layout[kind] = clamp(layout[kind] + delta, kind); applyLayout(); saveLayout(); }
function resizeKeys(kind, event) { const horizontal = kind !== "conversation"; const positive = horizontal ? event.key === "ArrowRight" : event.key === "ArrowDown"; const negative = horizontal ? event.key === "ArrowLeft" : event.key === "ArrowUp"; if (positive) nudgeResize(kind, 12); if (negative) nudgeResize(kind, -12); }
$("left-resizer").addEventListener("keydown", (event) => resizeKeys("sidebar", event)); $("right-resizer").addEventListener("keydown", (event) => resizeKeys("inspector", event)); $("center-resizer").addEventListener("keydown", (event) => resizeKeys("conversation", event)); $("repo-preview-resizer").addEventListener("keydown", (event) => resizeKeys("repoPreview", event)); $("change-preview-resizer").addEventListener("keydown", (event) => resizeKeys("changePreview", event));
loadLayout(); window.addEventListener("resize", () => { normalizeLayout(); applyLayout(); saveLayout(); }); refresh().then(events).catch((error) => toast(error.message));
