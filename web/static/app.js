const form = document.getElementById("chat-form");
const queryInput = document.getElementById("query");
const sendBtn = document.getElementById("send-btn");
const messages = document.getElementById("messages");
const thread = document.getElementById("thread");
const activityText = document.getElementById("activity");
const demoPanel = document.getElementById("demo-panel");
const drawerMask = document.getElementById("drawer-mask");
const toggleDemoBtn = document.getElementById("toggle-demo");
const demoCloseBtn = document.getElementById("demo-close");
const toggleMetaBtn = document.getElementById("toggle-meta");
const toggleOnlineBtn = document.getElementById("toggle-online");
const suggestionsSection = document.getElementById("suggestions");
const suggestionList = document.getElementById("suggestion-list");
const tcmPanel = document.getElementById("tcm-panel");
const tcmForm = document.getElementById("tcm-form");
const tcmSubmitBtn = document.getElementById("tcm-submit");
const tcmSubmitStatus = document.getElementById("tcm-submit-status");
const tcmProgressCard = document.getElementById("tcm-progress-card");
const tcmProgressList = document.getElementById("tcm-progress-list");
const tcmProgressNote = document.getElementById("tcm-progress-note");
const tcmAnalysisStack = document.getElementById("tcm-analysis-stack");
const tcmAnalysisList = document.getElementById("tcm-analysis-list");
const scrollBottomBtn = document.getElementById("scroll-bottom-btn");
const modeNormalBtn = document.getElementById("mode-normal");
const modeTcmBtn = document.getElementById("mode-tcm");
const toggleTitleColorBtn = document.getElementById("toggle-title-color");
const toggleModeInlineBtn = document.getElementById("toggle-mode-inline");
const llmModelSelect = document.getElementById("llm-model-select");
const toggleThinkingBtn = document.getElementById("toggle-thinking");
const tcmFlow = document.getElementById("tcm-flow");
const flowCollect = document.getElementById("flow-collect");
const flowQuestionnaire = document.getElementById("flow-questionnaire");
const flowResult = document.getElementById("flow-result");

const INTENT_LABEL_MAP = {
  daily_chat: "日常对话",
  symptom_consult: "健康问题与症状咨询",
  medical_knowledge: "医学科普与疾病知识",
  medication_question: "用药与健康品安全科普",
  lifestyle_guidance: "健康生活方式指导",
  appointment_process: "就医咨询",
  report_interpretation: "报告解读",
  after_sales: "产品/服务售后",
  human_service: "人工服务转接",
  non_medical: "非医疗信息",
  emergency: "高危分诊",
  other: "其他",
};

let showMeta = false;
let enableOnlineSearch = false;
let selectedLlmProvider = "volcengine";
let selectedLlmModel = "deepseek-v3-2-251201";
let enableThinking = false;
let backgroundColored = false;
let tcmSubmitting = false;
let shouldAutoScroll = true;
let generalSessionId = "";
const SCROLL_BOTTOM_THRESHOLD = 120;
let currentMode = "normal";
const DEFAULT_NORMAL_THREAD_HTML = thread ? thread.innerHTML : "";
const modeViewState = {
  normal: {
    threadHtml: DEFAULT_NORMAL_THREAD_HTML,
    suggestions: [],
  },
  tcm: {
    threadHtml: "",
    suggestions: [],
  },
};
const modeBridgeState = {
  normal: { lastUser: "", lastAssistant: "" },
  tcm: { lastUser: "", lastAssistant: "" },
};
const tcmState = {
  active: false,
  sessionId: "",
  questionnaire: [],
  round: 0,
  confidence: 0,
  done: false,
};

const TCM_PROGRESS_STEPS = [
  { key: "extract", label: "提取症状" },
  { key: "precheck", label: "风险筛查" },
  { key: "retrieve", label: "检索医案" },
  { key: "infer", label: "辨证分析" },
  { key: "questionnaire", label: "生成问卷" },
];

function messageDistanceToBottom() {
  if (!messages) return 0;
  return Math.max(0, messages.scrollHeight - messages.scrollTop - messages.clientHeight);
}

function isNearMessageBottom(threshold = SCROLL_BOTTOM_THRESHOLD) {
  return messageDistanceToBottom() <= threshold;
}

function updateScrollBottomButton() {
  if (!scrollBottomBtn) return;
  const show = !shouldAutoScroll && messageDistanceToBottom() > SCROLL_BOTTOM_THRESHOLD;
  scrollBottomBtn.hidden = !show;
}

function syncWindowScroll(force = false) {
  const root = document.scrollingElement || document.documentElement;
  if (!root) return;
  const remain = Math.max(0, root.scrollHeight - root.scrollTop - window.innerHeight);
  if (!force && remain > SCROLL_BOTTOM_THRESHOLD) return;
  window.scrollTo({
    top: root.scrollHeight,
    behavior: force ? "auto" : "smooth",
  });
}

function scrollToBottom(force = false) {
  if (!messages) return;
  if (!force && !shouldAutoScroll) {
    updateScrollBottomButton();
    return;
  }
  messages.scrollTop = messages.scrollHeight;
  requestAnimationFrame(() => {
    if (!messages) return;
    messages.scrollTop = messages.scrollHeight;
    syncWindowScroll(force);
    updateScrollBottomButton();
  });
}

function followToBottom() {
  shouldAutoScroll = true;
  scrollToBottom(true);
}

function setDemoCollapsed(collapsed) {
  if (!demoPanel || !toggleDemoBtn) return;
  const open = !collapsed;
  demoPanel.classList.toggle("is-open", open);
  if (drawerMask) {
    drawerMask.hidden = !open;
  }
  document.body.classList.toggle("demo-open", open);
  toggleDemoBtn.setAttribute("aria-expanded", String(open));
  toggleDemoBtn.setAttribute("aria-label", open ? "收起快速演示问题" : "展开快速演示问题");
}

function setMetaVisibility(visible) {
  showMeta = visible;
  if (toggleMetaBtn) {
    toggleMetaBtn.textContent = visible ? "隐藏状态信息" : "显示状态信息";
    toggleMetaBtn.setAttribute("aria-pressed", String(visible));
  }

  const metaNodes = document.querySelectorAll(".meta");
  metaNodes.forEach((node) => node.classList.toggle("hidden", !visible));
}

function setOnlineSearch(enabled) {
  enableOnlineSearch = Boolean(enabled);
  if (!toggleOnlineBtn) return;
  toggleOnlineBtn.textContent = enableOnlineSearch ? "联网: 开" : "联网: 关";
  toggleOnlineBtn.setAttribute("aria-pressed", String(enableOnlineSearch));
}

function setThinking(enabled) {
  enableThinking = Boolean(enabled);
  if (!toggleThinkingBtn) return;
  toggleThinkingBtn.textContent = enableThinking ? "思考: 开" : "思考: 关";
  toggleThinkingBtn.setAttribute("aria-pressed", String(enableThinking));
}

function hasVisibleChatContent() {
  const hasThread = Boolean(thread && thread.children.length > 0);
  const hasSuggestions = Boolean(suggestionsSection && !suggestionsSection.hidden);
  const hasFlow = Boolean(tcmFlow && !tcmFlow.hidden);
  const hasTcm = Boolean(tcmPanel && !tcmPanel.hidden);
  const hasActivity = Boolean(activityText && !activityText.hidden && String(activityText.textContent || "").trim());
  return hasThread || hasSuggestions || hasFlow || hasTcm || hasActivity;
}

function updateMessagesVisibility() {
  if (!messages) return;
  const hasContent = hasVisibleChatContent();
  messages.classList.add("has-content");
  document.body.classList.toggle("conversation-started", hasContent);
}

function setBackgroundColored(enabled) {
  backgroundColored = Boolean(enabled);
  document.body.classList.toggle("bg-colored", backgroundColored);
  if (!toggleTitleColorBtn) return;
  toggleTitleColorBtn.setAttribute("aria-pressed", String(backgroundColored));
  toggleTitleColorBtn.textContent = backgroundColored ? "背景: 流光" : "背景: 白色";
}

function shouldHideTcmInitMessage(text) {
  const normalized = String(text || "").replace(/\s+/g, "");
  return normalized.includes("已进入中医辨证问诊模式。为了提供更准确的诊断，请您尽量补充更详细的症状信息：");
}

function getCurrentSuggestionQueries() {
  if (!suggestionList) return [];
  return Array.from(suggestionList.querySelectorAll("button[data-query]"))
    .map((btn) => String(btn.dataset.query || "").trim())
    .filter((x) => x);
}

function captureCurrentModeView() {
  const state = modeViewState[currentMode];
  if (!state) return;
  state.threadHtml = thread ? thread.innerHTML : "";
  state.suggestions = getCurrentSuggestionQueries();
}

function restoreModeView(mode) {
  const state = modeViewState[mode];
  if (!state) return;
  if (thread) {
    if (state.threadHtml) {
      thread.innerHTML = state.threadHtml;
    } else {
      thread.innerHTML = mode === "normal" ? DEFAULT_NORMAL_THREAD_HTML : "";
      state.threadHtml = thread.innerHTML;
    }
  }
  renderSuggestions(state.suggestions || []);
  setMetaVisibility(showMeta);
  updateMessagesVisibility();
  followToBottom();
}

function updateBridgeState(role, text) {
  const key = role === "user" ? "lastUser" : "lastAssistant";
  if (!modeBridgeState[currentMode]) return;
  modeBridgeState[currentMode][key] = String(text || "").replace(/\s+/g, " ").trim().slice(0, 280);
}

function buildTcmSeedSummary() {
  const from = modeBridgeState.normal || { lastUser: "", lastAssistant: "" };
  const parts = [];
  if (from.lastUser) {
    parts.push(`普通咨询用户问题: ${from.lastUser}`);
  }
  if (from.lastAssistant) {
    parts.push(`普通咨询回复摘要: ${from.lastAssistant}`);
  }
  return parts.join("；").slice(0, 380);
}

function setTcmFlowStep(step) {
  if (tcmFlow) {
    tcmFlow.hidden = currentMode !== "tcm";
  }

  const steps = [flowCollect, flowQuestionnaire, flowResult];
  steps.forEach((node) => {
    if (!node) return;
    node.classList.remove("is-active", "is-done");
  });

  if (step === "result") {
    if (flowCollect) flowCollect.classList.add("is-done");
    if (flowQuestionnaire) flowQuestionnaire.classList.add("is-done");
    if (flowResult) flowResult.classList.add("is-active");
    updateMessagesVisibility();
    return;
  }

  if (step === "questionnaire") {
    if (flowCollect) flowCollect.classList.add("is-done");
    if (flowQuestionnaire) flowQuestionnaire.classList.add("is-active");
    updateMessagesVisibility();
    return;
  }

  if (flowCollect) flowCollect.classList.add("is-active");
  updateMessagesVisibility();
}

function refreshTcmPanelVisibility() {
  if (!tcmPanel) return;
  if (currentMode !== "tcm") {
    tcmPanel.hidden = true;
    if (tcmAnalysisStack) tcmAnalysisStack.hidden = true;
    if (tcmSubmitBtn) tcmSubmitBtn.hidden = true;
    if (tcmSubmitStatus) tcmSubmitStatus.hidden = true;
    updateMessagesVisibility();
    return;
  }
  const hasQuestionnaire = Array.isArray(tcmState.questionnaire) && tcmState.questionnaire.length > 0;
  const hasAnalysis = Boolean(tcmAnalysisList && tcmAnalysisList.children.length > 0);
  if (tcmAnalysisStack) tcmAnalysisStack.hidden = !hasAnalysis;
  if (tcmSubmitBtn) tcmSubmitBtn.hidden = !hasQuestionnaire;
  if (tcmSubmitStatus) tcmSubmitStatus.hidden = !hasQuestionnaire;
  tcmPanel.hidden = !(hasQuestionnaire || hasAnalysis);
  updateMessagesVisibility();
}

function syncModeUi() {
  const inTcm = currentMode === "tcm";

  if (modeNormalBtn) {
    modeNormalBtn.classList.toggle("is-active", !inTcm);
    modeNormalBtn.setAttribute("aria-selected", String(!inTcm));
  }
  if (modeTcmBtn) {
    modeTcmBtn.classList.toggle("is-active", inTcm);
    modeTcmBtn.setAttribute("aria-selected", String(inTcm));
  }
  if (toggleModeInlineBtn) {
    toggleModeInlineBtn.classList.toggle("is-active", inTcm);
    toggleModeInlineBtn.setAttribute("aria-pressed", String(inTcm));
    toggleModeInlineBtn.setAttribute("aria-label", inTcm ? "中医模式已开启，点击关闭" : "中医模式已关闭，点击开启");
    toggleModeInlineBtn.title = inTcm ? "中医模式：开" : "中医模式：关";
  }
  if (queryInput) {
    queryInput.placeholder = inTcm ? "请详细描述您的症状" : "有医疗相关的问题都可以问我哦~";
  }
  if (tcmFlow) {
    tcmFlow.hidden = !inTcm;
  }
  if (!inTcm && tcmProgressCard) {
    tcmProgressCard.hidden = true;
  }
  refreshTcmPanelVisibility();
}

async function switchMode(mode, options = {}) {
  const targetMode = mode === "tcm" ? "tcm" : "normal";
  const notify = options.notify !== false;
  const hardExit = Boolean(options.hardExit);
  const forceInit = Boolean(options.forceInit);

  if (targetMode === currentMode && !hardExit && !forceInit) {
    syncModeUi();
    return;
  }

  const fromMode = currentMode;
  if (fromMode !== targetMode) {
    captureCurrentModeView();
  }

  if (targetMode === "tcm") {
    currentMode = "tcm";
    modeViewState.tcm.threadHtml = "";
    syncModeUi();
    restoreModeView("tcm");
    if (tcmState.done) {
      setTcmFlowStep("result");
    } else if (tcmState.questionnaire.length) {
      setTcmFlowStep("questionnaire");
    } else {
      setTcmFlowStep("collect");
    }

    if (forceInit || !tcmState.active || !tcmState.sessionId) {
      setLoading(true, "切换到中医辨证模式...");
      try {
        await startTcmMode(buildTcmSeedSummary());
      } catch (err) {
        currentMode = "normal";
        syncModeUi();
        restoreModeView("normal");
        createAssistantMessage(`中医模式启动失败: ${err}`);
      } finally {
        setLoading(false);
      }
    } else {
      if (notify) {
        setActivity("已切换到中医辨证模式。请继续补充症状或填写问卷。");
      }
      refreshTcmPanelVisibility();
    }
    return;
  }

  currentMode = "normal";
  restoreModeView("normal");
  if (hardExit) {
    stopTcmMode(false);
  }
  syncModeUi();
  if (notify) {
    createAssistantMessage("已切换到普通咨询模式。");
  }
}

function riskLabel(riskLevel) {
  return riskLevel === "high" ? "高危" : "常规";
}

function sourceLabel(source) {
  return source === "llm" ? "模型" : "规则";
}

function intentLabel(intent) {
  return INTENT_LABEL_MAP[intent] || "待人工判断";
}

function setActivity(text = "") {
  if (!activityText) return;
  const message = String(text || "").trim();
  if (!message) {
    activityText.textContent = "";
    activityText.hidden = true;
    activityText.classList.add("hidden");
    updateMessagesVisibility();
    return;
  }
  activityText.textContent = message;
  activityText.hidden = false;
  activityText.classList.remove("hidden");
  updateMessagesVisibility();
}
function setLoading(loading, text = "") {
  if (sendBtn) sendBtn.disabled = loading;
  if (loading) {
    setActivity(text || "正在思考中...");
  } else {
    setActivity("");
  }
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatConfidencePercent(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return "--";
  return `${Math.round(Math.max(0, Math.min(1, num)) * 100)}%`;
}

function ensureTcmProgressTemplate() {
  if (!tcmProgressList) return;
  if (tcmProgressList.children.length) return;

  TCM_PROGRESS_STEPS.forEach((step) => {
    const item = document.createElement("li");
    item.className = "tcm-progress-item";
    item.dataset.stage = step.key;

    const label = document.createElement("span");
    label.className = "tcm-progress-label";
    label.textContent = step.label;

    const indicator = document.createElement("span");
    indicator.className = "tcm-progress-indicator";
    indicator.setAttribute("aria-hidden", "true");

    item.appendChild(label);
    item.appendChild(indicator);
    tcmProgressList.appendChild(item);
  });
}

function setTcmProgressVisible(visible) {
  if (!tcmProgressCard) return;
  tcmProgressCard.hidden = !visible;
  refreshTcmPanelVisibility();
}

function setTcmProgressNote(message = "", type = "info") {
  if (!tcmProgressNote) return;
  const text = String(message || "").trim();
  tcmProgressNote.classList.remove("is-success", "is-error");

  if (!text) {
    tcmProgressNote.textContent = "";
    tcmProgressNote.hidden = true;
    return;
  }

  tcmProgressNote.textContent = text;
  tcmProgressNote.hidden = false;
  if (type === "success") {
    tcmProgressNote.classList.add("is-success");
  } else if (type === "error") {
    tcmProgressNote.classList.add("is-error");
  }
}

function setTcmProgressByIndex(index, status = "active") {
  if (!tcmProgressList) return;
  const items = Array.from(tcmProgressList.querySelectorAll(".tcm-progress-item"));
  if (!items.length) return;

  items.forEach((item, i) => {
    item.classList.remove("is-pending", "is-active", "is-done", "is-error");
    if (i < index) {
      item.classList.add("is-done");
      return;
    }
    if (i === index) {
      item.classList.add(status === "error" ? "is-error" : status === "done" ? "is-done" : "is-active");
      return;
    }
    item.classList.add("is-pending");
  });
}

function setTcmProgressDoneTo(stageKey) {
  const index = TCM_PROGRESS_STEPS.findIndex((step) => step.key === stageKey);
  if (index < 0) return;
  setTcmProgressByIndex(index, "done");
}

function setTcmProgressActive(stageKey) {
  const index = TCM_PROGRESS_STEPS.findIndex((step) => step.key === stageKey);
  if (index < 0) return;
  setTcmProgressByIndex(index, "active");
}

function setTcmProgressError(stageKey) {
  const index = TCM_PROGRESS_STEPS.findIndex((step) => step.key === stageKey);
  if (index < 0) return;
  setTcmProgressByIndex(index, "error");
}

function resetTcmProgress() {
  ensureTcmProgressTemplate();
  if (!tcmProgressList) return;
  const items = Array.from(tcmProgressList.querySelectorAll(".tcm-progress-item"));
  items.forEach((item) => {
    item.classList.remove("is-active", "is-done", "is-error");
    item.classList.add("is-pending");
  });
  setTcmProgressNote("");
}

function clearTcmProgress() {
  resetTcmProgress();
  setTcmProgressVisible(false);
}

function handleTcmStageProgress(stage, stageText) {
  const code = String(stage || "").trim();
  if (!code) return;

  if (code === "merge") setTcmProgressActive("extract");
  if (code === "extract") setTcmProgressActive("extract");
  if (code === "precheck") setTcmProgressActive("precheck");
  if (code === "retrieve") setTcmProgressActive("retrieve");
  if (code === "infer" || code === "score") setTcmProgressActive("infer");
  if (code === "questionnaire") setTcmProgressActive("questionnaire");
  if (code === "ready") setTcmProgressDoneTo("questionnaire");
  if (code === "extract_fail") setTcmProgressError("extract");
  if (code === "red_flag") setTcmProgressError("precheck");
  if (code === "need_more") setTcmProgressDoneTo("precheck");

  if (!stageText) return;
  if (code === "extract_fail" || code === "red_flag") {
    setTcmProgressNote(stageText, "error");
    return;
  }
  if (code === "ready") {
    setTcmProgressNote(stageText, "success");
    return;
  }
  setTcmProgressNote(stageText, "info");
}

function resolveTcmAnalysisCardKey(payload, options = {}) {
  const explicit = String(options.key || "").trim();
  if (explicit) return explicit;
  const roundRaw = Number(payload.round);
  if (Number.isFinite(roundRaw) && roundRaw > 0) {
    return `round-${Math.floor(roundRaw)}`;
  }
  return "round-0";
}

function findTcmAnalysisCardByKey(cardKey) {
  if (!tcmAnalysisList) return null;
  const key = String(cardKey || "").trim();
  if (!key) return null;
  return Array.from(tcmAnalysisList.querySelectorAll(".tcm-analysis-card")).find(
    (node) => String(node.dataset.analysisKey || "").trim() === key,
  ) || null;
}

function renderTcmAnalysisCard(data, options = {}) {
  if (!tcmAnalysisStack || !tcmAnalysisList) return false;
  const payload = data && typeof data === "object" ? data : {};
  const cardKey = resolveTcmAnalysisCardKey(payload, options);
  const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
  const result = payload.result && typeof payload.result === "object" ? payload.result : {};
  const medicineSuggestions = Array.isArray(result.patent_medicine_suggestions) ? result.patent_medicine_suggestions : [];
  const roundRaw = Number(payload.round);
  const roundNo = Number.isFinite(roundRaw) && roundRaw > 0 ? Math.floor(roundRaw) : 0;
  const messageText = String(payload.message || "").trim();

  const hasResultSummary = Boolean(
    String(result.final_syndrome || "").trim() ||
      String(result.analysis || "").trim() ||
      String(result.advice || "").trim(),
  );
  const hasMeaningfulContent = Boolean(candidates.length || hasResultSummary || medicineSuggestions.length || messageText);
  if (!hasMeaningfulContent) {
    const stale = findTcmAnalysisCardByKey(cardKey);
    if (stale) stale.remove();
    refreshTcmPanelVisibility();
    return false;
  }

  const refsText = Array.isArray(payload.case_refs)
    ? payload.case_refs
        .slice(0, 5)
        .map((item) => {
          const lineNo = Number(item && item.line_no);
          const src = String((item && item.source) || "").trim();
          const file = String((item && item.file) || "").trim();
          const head = [src, file].filter((x) => x).join(":");
          if (head) {
            return `${head}${Number.isFinite(lineNo) && lineNo > 0 ? `#${lineNo}` : ""}`;
          }
          return Number.isFinite(lineNo) && lineNo > 0 ? `line#${lineNo}` : "";
        })
        .filter((x) => x)
        .join(" / ")
    : "";

  const leadSyndrome = String(
    result.final_syndrome || (candidates[0] && candidates[0].name) || "待进一步辨证",
  ).trim();
  const secondChoices = Array.isArray(result.second_choices)
    ? result.second_choices.map((x) => String(x || "").trim()).filter((x) => x).slice(0, 3)
    : [];
  const cardTitle = String(options.title || "").trim()
    || (roundNo > 0 ? `第${roundNo}轮辨证结果` : (options.preview ? "首轮辨证分析（生成问卷中）" : "首轮辨证结果"));

  const candidateRows = candidates
    .slice(0, 3)
    .map((item) => {
      const name = escapeHtml(String(item && item.name ? item.name : "待进一步辨证"));
      const scoreRaw = Number(item && item.score);
      const scoreText = Number.isFinite(scoreRaw)
        ? `${Math.round(Math.max(0, Math.min(1, scoreRaw)) * 100)}%`
        : "--";
      const reason = escapeHtml(String(item && item.reason ? item.reason : "暂无补充说明"));
      return `
        <li class="tcm-analysis-candidate">
          <div class="tcm-analysis-candidate-head">
            <span class="tcm-analysis-candidate-name">${name}</span>
            <span class="tcm-analysis-candidate-score">${scoreText}</span>
          </div>
          <div class="tcm-analysis-candidate-reason">${reason}</div>
        </li>
      `;
    })
    .join("");

  const noteRows = [];
  if (messageText) {
    noteRows.push(`<div class="tcm-analysis-note tcm-analysis-message">${escapeHtml(messageText).replace(/\n/g, "<br>")}</div>`);
  }
  const analysis = String(result.analysis || "").trim();
  const advice = String(result.advice || "").trim();
  if (analysis) {
    noteRows.push(`<div class="tcm-analysis-note"><strong>分析：</strong>${escapeHtml(analysis)}</div>`);
  }
  if (advice) {
    noteRows.push(`<div class="tcm-analysis-note"><strong>建议：</strong>${escapeHtml(advice)}</div>`);
  }

  const medicineRows = medicineSuggestions
    .slice(0, 3)
    .map((item) => {
      const name = escapeHtml(String(item && item.name ? item.name : "中成药建议"));
      const fitFor = escapeHtml(String(item && item.fit_for ? item.fit_for : "请结合辨证结果使用"));
      const cautions = escapeHtml(String(item && item.cautions ? item.cautions : "请遵医嘱并关注个体差异"));
      const why = escapeHtml(String(item && item.why ? item.why : "与当前辨证结果匹配"));
      const evidence = item && typeof item.evidence === "object" ? item.evidence : {};
      const sourceTitle = escapeHtml(String(evidence.title || ""));
      const sourceFile = escapeHtml(String(evidence.file || ""));
      const sourceLine = Number(evidence.line_no);
      const sourceText = sourceTitle || sourceFile
        ? `${sourceTitle}${sourceFile ? ` (${sourceFile}${Number.isFinite(sourceLine) && sourceLine > 0 ? `:${sourceLine}` : ""})` : ""}`
        : "";
      const excerpt = escapeHtml(String(evidence.excerpt || "").trim());
      const sourceDocId = escapeHtml(String(evidence.doc_id || "").trim());
      const hasEvidence = Boolean(sourceText || excerpt || sourceDocId);
      return `
        <li class="tcm-medicine-item">
          <div class="tcm-medicine-name">${name}</div>
          <div class="tcm-medicine-line"><strong>适用证型：</strong>${fitFor}</div>
          <div class="tcm-medicine-line"><strong>推荐理由：</strong>${why}</div>
          <div class="tcm-medicine-line"><strong>禁忌提示：</strong>${cautions}</div>
          ${
            hasEvidence
              ? `
                <details class="tcm-medicine-evidence">
                  <summary>查看原文索引</summary>
                  ${sourceText ? `<div class="tcm-medicine-evidence-source">${sourceText}</div>` : ""}
                  ${sourceDocId ? `<div class="tcm-medicine-evidence-source">ID: ${sourceDocId}</div>` : ""}
                  ${excerpt ? `<div class="tcm-medicine-evidence-excerpt">${excerpt}</div>` : ""}
                </details>
              `
              : ""
          }
        </li>
      `;
    })
    .join("");

  let cardNode = findTcmAnalysisCardByKey(cardKey);
  if (!cardNode) {
    cardNode = document.createElement("section");
    cardNode.className = "tcm-analysis-card";
    cardNode.dataset.analysisKey = cardKey;
    tcmAnalysisList.appendChild(cardNode);
  }

  cardNode.innerHTML = `
    <div class="tcm-analysis-title-row">
      <div class="tcm-analysis-title">${escapeHtml(cardTitle)}</div>
      <span class="tcm-analysis-round">${roundNo > 0 ? `第${roundNo}轮` : "首轮"}</span>
    </div>
    <div class="tcm-analysis-body">
      <div class="tcm-analysis-grid">
        <div class="tcm-analysis-item">
          <span class="tcm-analysis-item-label">当前主证候</span>
          <div class="tcm-analysis-item-value">${escapeHtml(leadSyndrome)}</div>
        </div>
        <div class="tcm-analysis-item">
          <span class="tcm-analysis-item-label">当前置信度</span>
          <div class="tcm-analysis-item-value">${formatConfidencePercent(payload.confidence)}</div>
        </div>
        <div class="tcm-analysis-item">
          <span class="tcm-analysis-item-label">参考医案</span>
          <div class="tcm-analysis-item-value">${escapeHtml(refsText || "暂无")}</div>
        </div>
      </div>
      ${
        secondChoices.length
          ? `<div class="tcm-analysis-note"><strong>备选证候：</strong>${escapeHtml(secondChoices.join("、"))}</div>`
          : ""
      }
      ${candidateRows ? `<ul class="tcm-analysis-candidates">${candidateRows}</ul>` : ""}
      ${noteRows.join("")}
      ${
        medicineRows
          ? `
            <section class="tcm-medicine-section">
              <div class="tcm-medicine-title">中成药建议</div>
              <ul class="tcm-medicine-list">${medicineRows}</ul>
            </section>
          `
          : ""
      }
    </div>
  `;

  if (options.moveToEnd !== false) {
    tcmAnalysisList.appendChild(cardNode);
  }

  tcmAnalysisStack.hidden = false;
  refreshTcmPanelVisibility();
  return true;
}

function clearTcmAnalysisCard() {
  if (!tcmAnalysisStack || !tcmAnalysisList) return;
  tcmAnalysisList.innerHTML = "";
  tcmAnalysisStack.hidden = true;
  refreshTcmPanelVisibility();
}

function markdownToSafeHtml(text) {
  const source = String(text || "");
  const markedApi = window.marked;
  let html = "";

  if (markedApi && typeof markedApi.parse === "function") {
    html = markedApi.parse(source, { gfm: true, breaks: true });
  } else {
    html = escapeHtml(source).replace(/\n/g, "<br>");
  }

  const purifier = window.DOMPurify;
  if (purifier && typeof purifier.sanitize === "function") {
    return purifier.sanitize(html, { USE_PROFILES: { html: true } });
  }
  return html;
}

function setAssistantBubbleMarkdown(bubble, text) {
  if (!bubble) return;
  bubble.innerHTML = markdownToSafeHtml(text);
}

function hasMarkdownSyntax(text) {
  const src = String(text || "");
  if (!src.trim()) return false;
  return /(^|\n)\s{0,3}(#{1,6}\s|[-*+]\s|\d+\.\s|>\s)|`{1,3}|\*\*[^*]+\*\*|__[^_]+__|\[[^\]]+\]\([^)]+\)|(^|\n)\|.+\|/m.test(src);
}

function finalizeAssistantStreamBubble(bubble, text) {
  if (!bubble) return;
  const src = String(text || "");
  if (!hasMarkdownSyntax(src)) {
    bubble.textContent = src;
    return;
  }
  bubble.classList.add("markdown-transition");
  setAssistantBubbleMarkdown(bubble, src);
  requestAnimationFrame(() => {
    bubble.classList.remove("markdown-transition");
  });
}

function createMessageShell(role) {
  const article = document.createElement("article");
  article.className = `msg ${role}`;

  const body = document.createElement("div");
  body.className = "msg-body";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const meta = document.createElement("div");
  meta.className = "meta hidden";

  body.appendChild(bubble);
  body.appendChild(meta);
  article.appendChild(body);

  if (thread) {
    thread.appendChild(article);
    updateMessagesVisibility();
    scrollToBottom();
  } else if (messages) {
    messages.appendChild(article);
    updateMessagesVisibility();
    scrollToBottom();
  }

  return { bubble, meta };
}

function applyMeta(metaNode, meta) {
  if (!metaNode || !meta) return;

  const riskClass = meta.risk_level === "high" ? "risk-high" : "risk-low";
  const handoffClass = meta.handoff ? "handoff-on" : "";
  const confidencePct = `${Math.round((meta.confidence || 0) * 100)}%`;
  const refs = (meta.citations || []).join(" | ") || "无";
  const refsClass = refs === "无" ? "refs-empty" : "";

  metaNode.innerHTML = `
    <div class="meta-grid">
      <span class="meta-chip ${riskClass}"><strong>风险</strong>${riskLabel(meta.risk_level)}</span>
      <span class="meta-chip"><strong>意图</strong>${intentLabel(meta.intent || "other")}</span>
      <span class="meta-chip"><strong>来源</strong>${sourceLabel(meta.intent_source)}</span>
      <span class="meta-chip"><strong>置信度</strong>${confidencePct}</span>
      <span class="meta-chip ${handoffClass}"><strong>人工</strong>${meta.handoff ? "已触发" : "不需要"}</span>
      <span class="meta-chip ${refsClass}"><strong>引用</strong>${refs}</span>
    </div>
  `;
  metaNode.classList.toggle("hidden", !showMeta);
}

function createUserMessage(text) {
  const { bubble } = createMessageShell("user");
  bubble.textContent = text;
  updateBridgeState("user", text);
  followToBottom();
}

function createAssistantMessage(text, meta = null) {
  const { bubble, meta: metaNode } = createMessageShell("assistant");
  setAssistantBubbleMarkdown(bubble, text);
  updateBridgeState("assistant", text);
  if (meta) {
    applyMeta(metaNode, meta);
  }
  scrollToBottom();
}

function createAssistantStreamMessage() {
  const shell = createMessageShell("assistant");
  shell.bubble.classList.add("streaming");
  shell.bubble.classList.remove("has-text");
  return shell;
}

function renderSuggestions(items) {
  if (!suggestionsSection || !suggestionList) return;

  suggestionList.innerHTML = "";
  const questions = Array.isArray(items) ? items.filter((x) => String(x || "").trim()) : [];

  if (!questions.length) {
    suggestionsSection.hidden = true;
    updateMessagesVisibility();
    return;
  }

  questions.slice(0, 4).forEach((question) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "suggest-chip";
    btn.dataset.query = String(question).trim();
    btn.textContent = String(question).trim();
    suggestionList.appendChild(btn);
  });

  suggestionsSection.hidden = false;
  updateMessagesVisibility();
}

function setTcmSubmitStatus(message = "", type = "hint") {
  if (!tcmSubmitStatus) return;
  const text = String(message || "").trim();
  tcmSubmitStatus.textContent = text;
  tcmSubmitStatus.classList.remove("hint", "loading", "success", "error");
  if (text) {
    tcmSubmitStatus.classList.add(type);
  }
}

function setTcmSubmitting(loading, message = "") {
  tcmSubmitting = Boolean(loading);
  if (tcmSubmitBtn) {
    tcmSubmitBtn.disabled = tcmSubmitting;
    tcmSubmitBtn.classList.toggle("is-loading", tcmSubmitting);
    tcmSubmitBtn.textContent = tcmSubmitting ? "提交中..." : "提交问卷";
  }
  if (tcmForm) {
    tcmForm.classList.toggle("is-submitting", tcmSubmitting);
    const controls = tcmForm.querySelectorAll("input, button, textarea");
    controls.forEach((control) => {
      control.disabled = tcmSubmitting;
    });
  }
  if (message) {
    setTcmSubmitStatus(message, tcmSubmitting ? "loading" : "hint");
  }
}

function derivePartialKeywords(question, rawKeywords) {
  const direct = Array.isArray(rawKeywords)
    ? Array.from(
        new Set(
          rawKeywords
            .map((x) => String(x || "").trim())
            .filter((x) => x.length >= 2)
            .slice(0, 8),
        ),
      )
    : [];
  if (direct.length) return direct;

  const text = String(question || "");
  const chunks = text
    .replace(/[（(].*?[)）]/g, "")
    .split(/[，、,；;。？！?]/)
    .flatMap((part) => part.split(/或|和|及|并且|并|且|与|\//))
    .map((part) =>
      String(part || "")
        .replace(/^(是否|是不是|有无|有没有|会不会|是否有|是否出现|是否伴有|是否明显|是否持续)+/g, "")
        .replace(/(吗|呢|情况|表现|症状)+$/g, "")
        .trim(),
    )
    .filter((part) => part.length >= 2);

  return Array.from(new Set(chunks)).slice(0, 6);
}

function escapeCssName(value) {
  const text = String(value || "");
  if (window.CSS && typeof window.CSS.escape === "function") {
    return window.CSS.escape(text);
  }
  return text.replaceAll("\\", "\\\\").replaceAll('"', '\\"');
}

function renderTcmQuestionnaire(questions) {
  if (!tcmPanel || !tcmForm || !tcmSubmitBtn) return;

  tcmForm.innerHTML = "";
  setTcmSubmitting(false);
  tcmState.questionnaire = Array.isArray(questions) ? questions : [];

  if (!tcmState.questionnaire.length) {
    refreshTcmPanelVisibility();
    setTcmSubmitStatus("");
    return;
  }

  tcmState.questionnaire.forEach((q, idx) => {
    const qid = q.id || `q${idx + 1}`;
    const block = document.createElement("div");
    block.className = "tcm-question";

    const title = document.createElement("p");
    title.textContent = `${idx + 1}. ${q.question || ""}`;
    block.appendChild(title);

    if (q.purpose) {
      const purpose = document.createElement("small");
      purpose.textContent = `目的：${q.purpose}`;
      block.appendChild(purpose);
    }

    const opts = document.createElement("div");
    opts.className = "tcm-options";
    const choices = Array.isArray(q.options) && q.options.length ? q.options : ["是", "部分是", "否", "不确定"];
    const partialKeywords = derivePartialKeywords(q.question, q.partial_keywords);

    choices.forEach((opt) => {
      const label = document.createElement("label");
      label.className = "tcm-option-chip";
      label.dataset.value = String(opt);
      const input = document.createElement("input");
      input.type = "radio";
      input.name = qid;
      input.value = opt;
      input.className = "tcm-option-input";
      const text = document.createElement("span");
      text.className = "tcm-option-text";
      text.textContent = String(opt);
      label.appendChild(input);
      label.appendChild(text);
      opts.appendChild(label);
    });

    const partialBox = document.createElement("div");
    partialBox.className = "tcm-partial-box";
    partialBox.dataset.qid = qid;
    partialBox.hidden = true;

    const partialTitle = document.createElement("p");
    partialTitle.className = "tcm-partial-title";
    partialTitle.textContent = "若选择“部分是”，请勾选符合的症状：";
    partialBox.appendChild(partialTitle);

    const partialChips = document.createElement("div");
    partialChips.className = "tcm-partial-chips";

    partialKeywords.forEach((keyword) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "tcm-partial-chip";
      chip.dataset.role = "keyword";
      chip.dataset.keyword = keyword;
      chip.setAttribute("aria-pressed", "false");
      chip.textContent = keyword;
      chip.addEventListener("click", () => {
        chip.classList.toggle("is-active");
        chip.setAttribute("aria-pressed", chip.classList.contains("is-active") ? "true" : "false");
      });
      partialChips.appendChild(chip);
    });

    const otherChip = document.createElement("button");
    otherChip.type = "button";
    otherChip.className = "tcm-partial-chip tcm-partial-chip-other";
    otherChip.dataset.role = "other";
    otherChip.setAttribute("aria-pressed", "false");
    otherChip.textContent = "其他";
    partialChips.appendChild(otherChip);

    const otherWrap = document.createElement("div");
    otherWrap.className = "tcm-partial-other";
    otherWrap.hidden = true;

    const otherInput = document.createElement("input");
    otherInput.type = "text";
    otherInput.className = "tcm-partial-other-input";
    otherInput.maxLength = 40;
    otherInput.placeholder = "请输入其他症状（可选）";
    otherWrap.appendChild(otherInput);

    otherChip.addEventListener("click", () => {
      const active = !otherChip.classList.contains("is-active");
      otherChip.classList.toggle("is-active", active);
      otherChip.setAttribute("aria-pressed", active ? "true" : "false");
      otherWrap.hidden = !active;
      if (active) {
        otherInput.focus();
      } else {
        otherInput.value = "";
      }
    });

    partialBox.appendChild(partialChips);
    partialBox.appendChild(otherWrap);

    const updatePartialVisibility = () => {
      const checked = tcmForm.querySelector(`input[name="${escapeCssName(qid)}"]:checked`);
      const isPartial = Boolean(checked && checked.value === "部分是");
      partialBox.hidden = !isPartial;
      block.classList.toggle("partial-open", isPartial);
    };

    const radios = opts.querySelectorAll(`input[name="${escapeCssName(qid)}"]`);
    radios.forEach((radio) => {
      radio.addEventListener("change", updatePartialVisibility);
    });

    block.appendChild(opts);
    block.appendChild(partialBox);
    tcmForm.appendChild(block);
    updatePartialVisibility();
  });

  refreshTcmPanelVisibility();
  setTcmSubmitStatus("请选择每一题答案；若选“部分是”，请补充具体症状后提交。", "hint");
}

function clearTcmQuestionnaire() {
  if (!tcmPanel || !tcmForm) return;
  tcmForm.innerHTML = "";
  setTcmSubmitting(false);
  setTcmSubmitStatus("");
  tcmState.questionnaire = [];
  refreshTcmPanelVisibility();
  updateMessagesVisibility();
}

function parseSSEBlock(block) {
  const lines = block.split("\n");
  let event = "message";
  const dataLines = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  const raw = dataLines.join("\n");
  if (!raw) return { event, data: {} };

  try {
    return { event, data: JSON.parse(raw) };
  } catch {
    return { event, data: { text: raw } };
  }
}

async function streamAssistantReply(query) {
  const streamMsg = createAssistantStreamMessage();
  let finalMeta = null;
  const streamParts = [];
  let hasToken = false;
  let pendingText = "";
  let rafFlushId = 0;
  let shouldScroll = false;

  const flushPendingText = () => {
    if (rafFlushId) {
      rafFlushId = 0;
    }
    if (!pendingText) return;
    streamParts.push(pendingText);
    streamMsg.bubble.textContent += pendingText;
    pendingText = "";
    if (shouldScroll) {
      scrollToBottom(true);
      shouldScroll = false;
    }
  };

  const scheduleFlushPendingText = () => {
    if (rafFlushId) return;
    rafFlushId = requestAnimationFrame(() => {
      flushPendingText();
    });
  };

  setActivity("正在准备回复...");

  const resp = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      enable_online_search: enableOnlineSearch,
      session_id: generalSessionId,
      llm_provider: selectedLlmProvider,
      llm_model: selectedLlmModel,
      llm_thinking: enableThinking,
    }),
  });

  if (!resp.ok || !resp.body) {
    throw new Error(`http_${resp.status}`);
  }
  setActivity("正在生成回复...");

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";

    for (const block of blocks) {
      const trimmed = block.trim();
      if (!trimmed) continue;

      const parsed = parseSSEBlock(trimmed);
      if (parsed.event === "token") {
        const chunkText = String(parsed.data.text || "");
        if (chunkText && !hasToken) {
          hasToken = true;
          setActivity("正在输出回复...");
        }
        if (chunkText && !streamMsg.bubble.classList.contains("has-text")) {
          streamMsg.bubble.classList.add("has-text");
        }
        pendingText += chunkText;
        shouldScroll = true;
        scheduleFlushPendingText();
      } else if (parsed.event === "meta") {
        finalMeta = parsed.data;
        const sid = String((finalMeta && finalMeta.session_id) || "").trim();
        if (sid) {
          generalSessionId = sid;
        }
      } else if (parsed.event === "error") {
        throw new Error(parsed.data.error || "stream_error");
      }
    }
  }

  if (rafFlushId) {
    cancelAnimationFrame(rafFlushId);
    rafFlushId = 0;
  }
  flushPendingText();

  const finalText = streamParts.join("");
  if (!finalText.trim()) {
    streamMsg.bubble.textContent = "(empty)";
    updateBridgeState("assistant", "(empty)");
  } else {
    finalizeAssistantStreamBubble(streamMsg.bubble, finalText);
    updateBridgeState("assistant", finalText);
  }
  streamMsg.bubble.classList.remove("streaming");
  streamMsg.bubble.classList.remove("has-text");
  applyMeta(streamMsg.meta, finalMeta);
  scrollToBottom(true);
  const sid = String((finalMeta && finalMeta.session_id) || "").trim();
  if (sid) {
    generalSessionId = sid;
  }
  return finalMeta || {};
}

async function startTcmMode(seedQuery = "") {
  const resp = await fetch("/api/tcm/init", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ seed_query: seedQuery }),
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data.error || `http_${resp.status}`);
  }

  tcmState.active = true;
  tcmState.sessionId = data.session_id;
  tcmState.questionnaire = [];
  tcmState.round = 0;
  tcmState.confidence = 0;
  tcmState.done = false;

  clearTcmProgress();
  clearTcmAnalysisCard();
  clearTcmQuestionnaire();
  const initMessage = String(data.message || "").trim();
  if (initMessage && !shouldHideTcmInitMessage(initMessage)) {
    setActivity(initMessage);
  }
  setTcmFlowStep("collect");
  syncModeUi();
  renderSuggestions([
    "我最近一周怕冷，经常感觉乏力、食欲差",
    "我常常口干口渴，失眠多梦、一晚上醒好几次",
    "退出中医辨证模式，回到普通咨询",
  ]);
}

function stopTcmMode(notify = true) {
  tcmState.active = false;
  tcmState.sessionId = "";
  tcmState.questionnaire = [];
  tcmState.round = 0;
  tcmState.confidence = 0;
  tcmState.done = false;
  clearTcmProgress();
  clearTcmAnalysisCard();
  clearTcmQuestionnaire();
  setTcmFlowStep("collect");
  syncModeUi();
  if (notify) {
    createAssistantMessage("已退出中医辨证模式，回到普通咨询。你可以继续描述问题。");
  }
}

function buildTcmMeta(data) {
  const riskHigh = Array.isArray(data.red_flags) && data.red_flags.length > 0;
  return {
    intent: "symptom_consult",
    intent_source: "llm",
    risk_level: riskHigh ? "high" : "low",
    confidence: Number(data.confidence || 0),
    handoff: riskHigh,
    citations: (data.case_refs || []).map((x) => {
      const lineNo = Number(x && x.line_no);
      const src = String((x && x.source) || "").trim();
      const file = String((x && x.file) || "").trim();
      const head = [src, file].filter((v) => v).join(":");
      if (head) {
        return `${head}${Number.isFinite(lineNo) && lineNo > 0 ? `#${lineNo}` : ""}`;
      }
      return Number.isFinite(lineNo) && lineNo > 0 ? `line#${lineNo}` : "";
    }),
  };
}

async function sendTcmCollect(query) {
  ensureTcmProgressTemplate();
  setTcmProgressVisible(true);
  resetTcmProgress();
  setTcmProgressActive("extract");
  setTcmProgressNote("正在整理症状描述...", "info");
  setTcmSubmitStatus("");
  setActivity("正在整理症状描述...");

  const resp = await fetch("/api/tcm/collect/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: tcmState.sessionId, user_input: query }),
  });
  if (!resp.ok || !resp.body) {
    setTcmProgressError("extract");
    setTcmProgressNote("请求失败，请检查网络后重试。", "error");
    throw new Error(`http_${resp.status}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finalData = null;
  let hasAnalysisPreview = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";

    for (const block of blocks) {
      const trimmed = block.trim();
      if (!trimmed) continue;
      const parsed = parseSSEBlock(trimmed);

      if (parsed.event === "stage") {
        const stageCode = String(parsed.data.stage || "").trim();
        const stageText = String(parsed.data.text || "").trim();
        handleTcmStageProgress(stageCode, stageText);
        if (stageText) {
          setActivity(stageText);
          scrollToBottom(true);
        }
        continue;
      }

      if (parsed.event === "analysis") {
        const analysisData = parsed.data && typeof parsed.data === "object" ? parsed.data : {};
        const analysisText = String(analysisData.message || "").trim();
        if (analysisText) {
          hasAnalysisPreview = true;
          setTcmProgressDoneTo("infer");
          setTcmProgressActive("questionnaire");
          setTcmProgressNote("辨证分析已完成，正在生成问卷...", "info");
          setActivity("辨证分析已完成，正在生成问卷...");
          updateBridgeState("assistant", analysisText);
          renderTcmAnalysisCard(analysisData, { key: "round-0", preview: true });
          scrollToBottom(true);
        }
        continue;
      }

      if (parsed.event === "result") {
        finalData = parsed.data || {};
        continue;
      }

      if (parsed.event === "error") {
        setTcmProgressNote(String(parsed.data.error || "辨证流程失败"), "error");
        throw new Error(parsed.data.error || "tcm_stream_error");
      }
    }
  }

  const tail = buffer.trim();
  if (tail) {
    const parsed = parseSSEBlock(tail);
    if (parsed.event === "result") {
      finalData = parsed.data || finalData;
    } else if (parsed.event === "error") {
      setTcmProgressNote(String(parsed.data.error || "辨证流程失败"), "error");
      throw new Error(parsed.data.error || "tcm_stream_error");
    }
  }

  const data = finalData || {};
  if (!Object.keys(data).length) {
    setTcmProgressError("questionnaire");
    setTcmProgressNote("未收到有效结果，请重试。", "error");
    throw new Error("tcm_stream_empty_result");
  }

  tcmState.done = Boolean(data.done);
  tcmState.round = Number(data.round || 0);
  tcmState.confidence = Number(data.confidence || 0);

  const finalText = String(data.message || "已记录。请继续补充。").trim();
  if (finalText && !hasAnalysisPreview) {
    updateBridgeState("assistant", finalText);
  }
  scrollToBottom(true);

  if (tcmState.done) {
    const hasAnalysisCard = renderTcmAnalysisCard(data, { key: "round-0" });
    if (Array.isArray(data.red_flags) && data.red_flags.length) {
      setTcmProgressError("precheck");
      setTcmProgressNote("检测到高风险症状，已暂停问卷并建议线下就医。", "error");
      if (finalText) {
        setActivity(finalText);
      }
    } else {
      setTcmProgressDoneTo("questionnaire");
      setTcmProgressNote("已完成本轮辨证。", "success");
      if (finalText) {
        setActivity(finalText);
      }
    }
    setTcmFlowStep("result");
    clearTcmQuestionnaire();
    if (!hasAnalysisCard) {
      clearTcmAnalysisCard();
    }
    renderSuggestions(["退出中医辨证模式，回到普通咨询"]);
    return;
  }

  if (data.extraction_ok === false) {
    setTcmProgressError("extract");
    setTcmProgressNote("症状提取失败，请稍后重试。", "error");
    if (finalText) {
      setActivity(finalText);
    }
    setTcmFlowStep("collect");
    clearTcmAnalysisCard();
    clearTcmQuestionnaire();
    renderSuggestions(["退出中医辨证模式，回到普通咨询"]);
    return;
  }
  if (data.need_more) {
    setTcmProgressDoneTo("precheck");
    setTcmProgressNote("信息不足，请继续补充症状后再生成问卷。", "info");
    if (finalText) {
      setActivity(finalText);
    }
    setTcmFlowStep("collect");
    clearTcmAnalysisCard();
    clearTcmQuestionnaire();
    renderSuggestions([
      "我还会口渴、出汗多",
      "我睡眠差、大便偏稀",
      "退出中医辨证模式，回到普通咨询",
    ]);
  } else {
    setTcmProgressDoneTo("questionnaire");
    setTcmProgressNote("问卷已生成，可在下方继续填写。", "success");
    if (finalText) {
      setActivity("问卷已生成，请在下方继续填写。");
    }
    setTcmFlowStep("questionnaire");
    renderTcmAnalysisCard(data, { key: "round-0" });
    renderTcmQuestionnaire(data.questionnaire || []);
    renderSuggestions(["先填写下方问卷再提交", "退出中医辨证模式，回到普通咨询"]);
  }
}

async function submitTcmQuestionnaire() {
  if (tcmSubmitting) {
    return;
  }
  if (!tcmState.active || !tcmState.sessionId || !tcmState.questionnaire.length) {
    setTcmSubmitStatus("当前没有可提交的问卷。", "error");
    return;
  }

  const answers = {};
  for (const q of tcmState.questionnaire) {
    const qid = q.id;
    if (!qid) continue;
    const checked = document.querySelector(`input[name="${escapeCssName(qid)}"]:checked`);
    if (!checked) {
      setTcmSubmitStatus("问卷还有未选择项，请先完成所有题目。", "error");
      return;
    }
    const value = String(checked.value || "").trim();
    if (value === "部分是") {
      const partialBox = tcmForm ? tcmForm.querySelector(`.tcm-partial-box[data-qid="${qid}"]`) : null;
      const keywordButtons = partialBox
        ? Array.from(partialBox.querySelectorAll(".tcm-partial-chip[data-role='keyword'].is-active"))
        : [];
      const selectedKeywords = keywordButtons
        .map((btn) => String(btn.dataset.keyword || "").trim())
        .filter((x) => x);

      const otherToggle = partialBox ? partialBox.querySelector(".tcm-partial-chip[data-role='other']") : null;
      const otherInput = partialBox ? partialBox.querySelector(".tcm-partial-other-input") : null;
      const otherEnabled = Boolean(otherToggle && otherToggle.classList.contains("is-active"));
      const otherText = otherEnabled && otherInput ? String(otherInput.value || "").trim() : "";

      if (!selectedKeywords.length && !otherText) {
        setTcmSubmitStatus("该题选择了“部分是”，请至少勾选一个症状或填写“其他”。", "error");
        return;
      }

      answers[qid] = {
        value,
        selected_keywords: selectedKeywords,
        other_text: otherText,
      };
    } else {
      answers[qid] = { value };
    }
  }

  setTcmSubmitting(true, "已提交，正在复判并生成结果...");
  setLoading(true, "中医辨证复判中...");
  try {
    const resp = await fetch("/api/tcm/questionnaire", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: tcmState.sessionId, answers }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.error || `http_${resp.status}`);
    }

    tcmState.done = Boolean(data.done);
    tcmState.round = Number(data.round || tcmState.round);
    tcmState.confidence = Number(data.confidence || tcmState.confidence);

    renderTcmAnalysisCard(data, {
      key: `round-${Math.max(1, Math.floor(Number(data.round || tcmState.round || 1)))}`,
    });
    const doneMessage = String(data.message || "").trim();
    if (doneMessage) {
      updateBridgeState("assistant", doneMessage);
    }

    if (tcmState.done) {
      setTcmFlowStep("result");
      clearTcmQuestionnaire();
      setTcmSubmitStatus("问卷已提交，本轮辨证已完成。", "success");
      setActivity("本轮辨证已完成，可查看上方结果卡。");
    } else {
      setTcmFlowStep("questionnaire");
      renderTcmQuestionnaire(data.questionnaire || []);
      setTcmSubmitStatus("问卷已提交，已生成新一轮问题。", "success");
      setActivity("新一轮问卷已生成，请继续作答。");
    }

    renderSuggestions(data.follow_ups || ["退出中医辨证模式，回到普通咨询"]);
  } catch (err) {
    setTcmSubmitStatus(`提交失败：${err}`, "error");
    setActivity(`问卷提交失败：${err}`);
  } finally {
    setTcmSubmitting(false);
    if (sendBtn) {
      sendBtn.disabled = false;
    }
  }
}

function isTcmStartQuery(query) {
  return query.includes("中医辨证") || query.includes("中医诊疗") || query.includes("中医问诊");
}

function isTcmExitQuery(query) {
  return query.includes("退出中医辨证模式");
}

function isTcmRestartQuery(query) {
  return query.includes("重新开始辨证");
}

async function sendQuery(rawQuery) {
  const query = String(rawQuery || "").trim();
  if (!query) return;
  const inTcmMode = currentMode === "tcm";

  if (isTcmStartQuery(query)) {
    if (!inTcmMode) {
      createUserMessage(query);
    } else {
      updateBridgeState("user", query);
    }
    if (queryInput) queryInput.value = "";
    await switchMode("tcm", { notify: false });
    return;
  }

  if (isTcmExitQuery(query)) {
    if (!inTcmMode) {
      createUserMessage(query);
    } else {
      updateBridgeState("user", query);
    }
    await switchMode("normal", { notify: false, hardExit: true });
    createAssistantMessage("已退出中医辨证模式，回到普通咨询。你可以继续描述问题。");
    renderSuggestions([]);
    return;
  }

  if (currentMode === "tcm" && isTcmRestartQuery(query)) {
    updateBridgeState("user", query);
    setLoading(true, "重置中医辨证会话...");
    try {
      await startTcmMode("");
    } catch (err) {
      setActivity(`会话重置失败：${err}`);
    } finally {
      setLoading(false);
    }
    return;
  }

  if (currentMode === "tcm" && query.includes("继续补充症状")) {
    updateBridgeState("user", query);
    clearTcmQuestionnaire();
    setTcmFlowStep("collect");
    setActivity("请继续补充症状细节，我会重新生成辨证问卷。可补充寒热、汗出、口渴、睡眠、二便、舌苔。");
    renderSuggestions([]);
    return;
  }

  if (!inTcmMode) {
    createUserMessage(query);
  } else {
    updateBridgeState("user", query);
  }
  if (queryInput) queryInput.value = "";
  setLoading(true);
  renderSuggestions([]);
  if (!inTcmMode) {
    followToBottom();
  }

  try {
    if (currentMode === "tcm") {
      if (!tcmState.active || !tcmState.sessionId) {
        await switchMode("tcm", { notify: false, forceInit: true });
      }
      await sendTcmCollect(query);
    } else {
      const meta = await streamAssistantReply(query);
      renderSuggestions(meta.follow_ups || []);
    }
  } catch (err) {
    const msg = `网络错误: ${err}`;
    if (currentMode === "tcm") {
      setTcmProgressError("extract");
      setActivity(msg);
      setTcmSubmitStatus(msg, "error");
    } else {
      const errorShell = createAssistantStreamMessage();
      errorShell.bubble.textContent = msg;
      updateBridgeState("assistant", msg);
      errorShell.bubble.classList.remove("streaming");
    }
    renderSuggestions([]);
  } finally {
    if (currentMode === "tcm") {
      if (sendBtn) {
        sendBtn.disabled = false;
      }
    } else {
      setLoading(false);
    }
  }
}

if (form) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await sendQuery(queryInput ? queryInput.value : "");
  });
}

document.addEventListener("click", async (event) => {
  const target = event.target.closest("button[data-query]");
  if (!target) return;
  await sendQuery(target.dataset.query || "");
});

if (queryInput) {
  queryInput.addEventListener("keydown", async (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      await sendQuery(queryInput.value);
    }
  });
}

if (toggleDemoBtn) {
  toggleDemoBtn.addEventListener("click", () => {
    const open = demoPanel ? demoPanel.classList.contains("is-open") : false;
    setDemoCollapsed(open);
  });
}

if (demoCloseBtn) {
  demoCloseBtn.addEventListener("click", () => {
    setDemoCollapsed(true);
  });
}

if (drawerMask) {
  drawerMask.addEventListener("click", () => {
    setDemoCollapsed(true);
  });
}

if (toggleMetaBtn) {
  toggleMetaBtn.addEventListener("click", () => {
    setMetaVisibility(!showMeta);
  });
}

if (toggleOnlineBtn) {
  toggleOnlineBtn.addEventListener("click", () => {
    setOnlineSearch(!enableOnlineSearch);
  });
}

if (llmModelSelect) {
  llmModelSelect.addEventListener("change", () => {
    const picked = String(llmModelSelect.value || "").trim();
    if (!picked) return;
    selectedLlmModel = picked;
    selectedLlmProvider = "volcengine";
  });
}

if (toggleThinkingBtn) {
  toggleThinkingBtn.addEventListener("click", () => {
    setThinking(!enableThinking);
  });
}

if (modeNormalBtn) {
  modeNormalBtn.addEventListener("click", async () => {
    await switchMode("normal", { notify: true });
  });
}

if (modeTcmBtn) {
  modeTcmBtn.addEventListener("click", async () => {
    await switchMode("tcm", { notify: false });
  });
}

if (toggleModeInlineBtn) {
  toggleModeInlineBtn.addEventListener("click", async () => {
    if (currentMode === "tcm") {
      await switchMode("normal", { notify: true });
      return;
    }
    await switchMode("tcm", { notify: false });
  });
}

if (toggleTitleColorBtn) {
  toggleTitleColorBtn.addEventListener("click", () => {
    setBackgroundColored(!backgroundColored);
  });
}

if (tcmSubmitBtn) {
  tcmSubmitBtn.addEventListener("click", async () => {
    await submitTcmQuestionnaire();
  });
}

setDemoCollapsed(true);
setMetaVisibility(false);
setOnlineSearch(false);
if (llmModelSelect) {
  const first = String(llmModelSelect.value || "").trim();
  if (first) {
    selectedLlmModel = first;
  }
}
setThinking(false);
setBackgroundColored(false);
setTcmFlowStep("collect");
syncModeUi();
clearTcmProgress();
clearTcmAnalysisCard();
clearTcmQuestionnaire();
updateMessagesVisibility();
followToBottom();
captureCurrentModeView();

if (messages) {
  messages.addEventListener("scroll", () => {
    shouldAutoScroll = isNearMessageBottom();
    updateScrollBottomButton();
  });
}

if (scrollBottomBtn) {
  scrollBottomBtn.addEventListener("click", () => {
    followToBottom();
  });
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    setDemoCollapsed(true);
  }
});
