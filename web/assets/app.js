"use strict";

const state = {
  challenges: [],
  filter: "",
  difficulty: "all",
  current: null,
  detail: null,
  tab: "problem",
  job: null,
  pollTimer: null,
  startedAt: 0,
  authRequired: false,
  history: [],
  pendingJob: null,
  retryAfterToken: false,
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

function getToken() {
  return localStorage.getItem("xpu_token") || "";
}

function getServiceBase() {
  const stored = localStorage.getItem("xpu_service");
  if (stored) return stored.replace(/\/+$/, "");
  const config = window.XPU_RUNNER_CONFIG || {};
  return (config.apiBase || "").replace(/\/+$/, "");
}

function setServiceBase(value) {
  const cleaned = (value || "").trim().replace(/\/+$/, "");
  if (cleaned) localStorage.setItem("xpu_service", cleaned);
  else localStorage.removeItem("xpu_service");
}
function setToken(value) {
  if (value) localStorage.setItem("xpu_token", value);
  else localStorage.removeItem("xpu_token");
  updateTokenButton();
}

function updateTokenButton() {
  const button = $("#token-btn");
  if (!button) return;
  const hasToken = Boolean(getToken());
  button.textContent = hasToken ? "Token set" : "API token";
  button.classList.toggle("set", hasToken);
}

function openTokenDialog(message, retry = false) {
  const dialog = $("#token-dialog");
  if (!dialog) return;
  state.retryAfterToken = retry || state.retryAfterToken;
  if (message) $("#token-message").textContent = message;
  $("#token-input").value = getToken();
  $("#service-input").value = getServiceBase();
  if (!dialog.open) dialog.showModal();
  $("#token-input").focus();
  $("#token-input").select();
}

class AuthError extends Error {}

async function api(path, options = {}) {
  const headers = Object.assign({}, options.headers);
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (options.body) headers["Content-Type"] = "application/json";
  const res = await fetch(getServiceBase() + path, Object.assign({}, options, { headers }));
  if (res.status === 401) {
    openTokenDialog("This server requires an API token. Paste it and press Save — the run will retry.", true);
    throw new AuthError("API token required or invalid");
  }
  let payload = null;
  try {
    payload = await res.json();
  } catch (err) {
    payload = null;
  }
  if (!res.ok) {
    const detail = payload && (payload.detail || payload.error);
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return payload;
}

function showBanner(message) {
  const banner = $("#banner");
  if (!message) {
    banner.classList.add("hidden");
    banner.textContent = "";
    return;
  }
  banner.textContent = message;
  banner.classList.remove("hidden");
}

function toast(message, ms = 4000) {
  showBanner(message);
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => showBanner(""), ms);
}

function relativeTime(ts) {
  const seconds = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  return `${Math.round(seconds / 3600)}h ago`;
}

function fmtMs(value) {
  if (value === null || value === undefined) return "–";
  return value >= 10 ? `${value.toFixed(1)} ms` : `${value.toFixed(3)} ms`;
}

async function boot() {
  try {
    const health = await api("/healthz");
    state.authRequired = Boolean(health.auth_required);
    const dot = $("#status-dot");
    dot.classList.add("ok");
    $("#status-text").textContent = `${health.challenges} challenges · ${health.devices.length} GPU(s) · ${getServiceBase() || "this host"}`;
    if (state.authRequired && !getToken()) {
      showBanner("This server requires an API token. Click “API token” to enter it.");
      openTokenDialog("This server requires an API token. Paste it and press Save.");
    }
  } catch (err) {
    $("#status-dot").classList.add("bad");
    $("#status-text").textContent = "service unreachable";
    showBanner(`Cannot reach the runner service: ${err.message}`);
  }

  state.history = JSON.parse(localStorage.getItem("xpu_history") || "[]");
  renderHistory();
  await loadChallenges();
  window.addEventListener("hashchange", route);
  route();
}

async function loadChallenges() {
  try {
    const data = await api("/api/v1/challenges");
    state.challenges = data.challenges || [];
    renderSidebar();
  } catch (err) {
    showBanner(`Failed to load challenges: ${err.message}`);
  }
}

function renderSidebar() {
  const list = $("#challenge-list");
  list.innerHTML = "";
  const needle = state.filter.trim().toLowerCase();
  const rows = state.challenges.filter((challenge) => {
    if (state.difficulty !== "all" && challenge.difficultyLevel !== state.difficulty) return false;
    return !needle || challenge.title.toLowerCase().includes(needle) || String(challenge.id) === needle;
  });
  for (const challenge of rows) {
    const item = el("li");
    if (state.current && state.current.id === challenge.id) item.classList.add("active");
    item.appendChild(el("span", "title", challenge.title));
    const badge = el("span", `badge ${challenge.difficultyLevel}`, challenge.difficultyLevel);
    item.appendChild(badge);
    item.addEventListener("click", () => {
      window.location.hash = `#/c/${challenge.id}`;
    });
    list.appendChild(item);
  }
  if (!rows.length) {
    const empty = el("li", "muted", "No matching challenges");
    empty.style.cursor = "default";
    list.appendChild(empty);
  }
}

function route() {
  const hash = window.location.hash.replace(/^#\/?/, "");
  const match = hash.match(/^c\/([^/]+)(?:\/(code|problem))?$/);
  if (match) openChallenge(match[1], match[2] || "problem");
  else renderEmpty();
}

function renderEmpty() {
  state.current = null;
  state.detail = null;
  renderSidebar();
  $("#content").innerHTML = "";
  const box = el("div", "empty");
  box.appendChild(el("h1", "", "Intel XPU playground"));
  box.appendChild(el("p", "muted", "Pick a challenge on the left, write SYCL, and run it on a real Arc GPU."));
  $("#content").appendChild(box);
}

async function openChallenge(ref, tab = "problem") {
  try {
    const detail = await api(`/api/v1/challenges/${encodeURIComponent(ref)}`);
    state.detail = detail;
    state.current = detail;
    state.job = state.pendingJob || null;
    state.pendingJob = null;
    state.tab = tab;
    renderSidebar();
    renderChallenge();
  } catch (err) {
    showBanner(`Failed to open challenge: ${err.message}`);
  }
}

function syclStarter(detail) {
  const starters = detail.starterCode || [];
  return starters.find((entry) => entry.language === "sycl") || null;
}

function draftKey() {
  return `xpu_draft_${state.detail.id}`;
}

function renderChallenge() {
  const detail = state.detail;
  const content = $("#content");
  content.innerHTML = "";

  const head = el("div", "ch-head");
  head.appendChild(el("h1", "", detail.title));
  head.appendChild(el("span", `badge ${detail.difficultyLevel}`, detail.difficultyLevel));
  head.appendChild(el("span", "badge", detail.id));
  const starter = syclStarter(detail);
  head.appendChild(el("span", `badge ${starter ? "sycl" : ""}`, starter ? "SYCL" : "no SYCL starter"));
  content.appendChild(head);

  const tabs = el("div", "tabs");
  for (const [key, label] of [["problem", "Problem"], ["code", "Code"]]) {
    const tab = el("button", `tab ${state.tab === key ? "active" : ""}`, label);
    tab.type = "button";
    tab.addEventListener("click", () => {
      state.tab = key;
      window.location.hash = `#/c/${detail.id}/${key}`;
      renderChallenge();
    });
    tabs.appendChild(tab);
  }
  content.appendChild(tabs);

  if (state.tab === "problem") {
    const panel = el("div", "panel spec");
    panel.innerHTML = detail.spec || "<p class='muted'>No description.</p>";
    content.appendChild(panel);
  } else {
    content.appendChild(renderEditor(starter));
  }

  if (state.job) content.appendChild(renderResults());
}

function renderEditor(starter) {
  const wrap = el("div");

  const bar = el("div", "editor-bar");
  const label = el("span", "muted small", starter ? `starter: ${starter.fileName}` : "no SYCL starter");
  bar.appendChild(label);
  bar.appendChild(el("span", "spacer"));

  const reset = el("button", "ghost small", "Reset");
  reset.type = "button";
  reset.disabled = !starter;
  reset.addEventListener("click", () => {
    if (!starter) return;
    localStorage.removeItem(draftKey());
    const editor = $("#editor");
    if (editor) editor.value = starter.fileContent;
  });
  bar.appendChild(reset);

  const copy = el("button", "ghost small", "Copy");
  copy.type = "button";
  copy.addEventListener("click", async () => {
    const editor = $("#editor");
    if (!editor) return;
    try {
      await navigator.clipboard.writeText(editor.value);
      toast("Copied to clipboard");
    } catch (err) {
      toast("Clipboard blocked by the browser");
    }
  });
  bar.appendChild(copy);

  const run = el("button", "primary", "Run on XPU");
  run.type = "button";
  run.id = "run-btn";
  run.disabled = !starter;
  run.addEventListener("click", startRun);
  bar.appendChild(run);
  wrap.appendChild(bar);

  const editor = el("textarea", "code");
  editor.id = "editor";
  editor.spellcheck = false;
  const draft = localStorage.getItem(draftKey());
  editor.value = draft !== null ? draft : starter ? starter.fileContent : "";
  editor.addEventListener("input", () => localStorage.setItem(draftKey(), editor.value));
  editor.addEventListener("keydown", (event) => {
    if (event.key === "Tab") {
      event.preventDefault();
      const start = editor.selectionStart;
      const end = editor.selectionEnd;
      editor.setRangeText("    ", start, end, "end");
      localStorage.setItem(draftKey(), editor.value);
    }
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      startRun();
    }
  });
  wrap.appendChild(editor);

  const advanced = el("div", "advanced");
  advanced.appendChild(numberField("runs", "Timed runs", 15, 1, 100));
  advanced.appendChild(numberField("inner", "Inner repeats", 1, 1, 1000));
  advanced.appendChild(numberField("timeout", "Timeout (s)", 40, 5, 600));
  wrap.appendChild(advanced);

  return wrap;
}

function numberField(id, labelText, value, min, max) {
  const label = el("label");
  label.appendChild(document.createTextNode(labelText));
  const input = document.createElement("input");
  input.type = "number";
  input.id = `opt-${id}`;
  input.value = String(value);
  input.min = String(min);
  input.max = String(max);
  label.appendChild(input);
  return label;
}

async function startRun() {
  const editor = $("#editor");
  if (!editor || !state.detail) return;
  const runBtn = $("#run-btn");
  if (runBtn) runBtn.disabled = true;
  state.startedAt = Date.now();
  state.job = { status: "queued", result: null };
  state.tab = "code";
  renderChallenge();

  const payload = {
    challenge: state.detail.id,
    source: editor.value,
    language: "sycl",
    accelerator: "B70",
    runs: Number($("#opt-runs")?.value || 15),
    inner_repeats: Number($("#opt-inner")?.value || 1),
    timeout: Number($("#opt-timeout")?.value || 40),
  };

  try {
    const job = await api("/api/v1/submissions", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.job = job;
    renderChallenge();
    pollJob(job.id);
  } catch (err) {
    state.job = { status: "error", result: { status: "error", error: { stage: "client", message: err.message } } };
    renderChallenge();
    if (err instanceof AuthError) {
      showBanner("This server requires an API token — enter it in the dialog to retry.");
    }
  } finally {
    if (runBtn) runBtn.disabled = false;
  }
}

function pollJob(id) {
  window.clearTimeout(state.pollTimer);
  const tick = async () => {
    try {
      const job = await api(`/api/v1/submissions/${id}`);
      state.job = job;
      renderChallenge();
      if (["success", "error", "timeout", "oom", "interrupted"].includes(job.status)) {
        rememberRun(job);
        return;
      }
      state.pollTimer = window.setTimeout(tick, 800);
    } catch (err) {
      state.job = { status: "error", result: { status: "error", error: { stage: "client", message: err.message } } };
      renderChallenge();
    }
  };
  state.pollTimer = window.setTimeout(tick, 400);
}

function rememberRun(job) {
  const result = job.result || {};
  const perf = result.performance;
  const entry = {
    id: job.id,
    challenge: job.challenge,
    status: job.status,
    median: perf ? perf.median_ms : null,
    ts: Date.now(),
  };
  state.history = [entry, ...state.history.filter((item) => item.id !== job.id)].slice(0, 20);
  localStorage.setItem("xpu_history", JSON.stringify(state.history));
  renderHistory();
}

function renderHistory() {
  const list = $("#history-list");
  if (!list) return;
  list.innerHTML = "";
  if (!state.history.length) {
    list.appendChild(el("li", "muted small", "No runs yet"));
    return;
  }
  for (const entry of state.history.slice(0, 10)) {
    const item = el("li", "small");
    const left = el("span", "", `${entry.challenge.split("/").pop()} · ${entry.status}`);
    left.style.overflow = "hidden";
    left.style.textOverflow = "ellipsis";
    left.style.whiteSpace = "nowrap";
    const right = el("span", "muted", entry.median ? fmtMs(entry.median) : relativeTime(entry.ts));
    item.appendChild(left);
    item.appendChild(right);
    item.addEventListener("click", async () => {
      try {
        state.pendingJob = await api(`/api/v1/submissions/${entry.id}`);
      } catch (err) {
        state.pendingJob = null;
        toast("That run is no longer available on the server");
      }
      window.location.hash = `#/c/${entry.challenge.split("/").pop().split("_")[0]}/code`;
    });
    list.appendChild(item);
  }
}

function renderResults() {
  const job = state.job;
  const result = job.result || {};
  const wrap = el("div", "results panel");
  const head = el("div", "result-head");
  const status = job.status || "queued";
  head.appendChild(el("span", `pill ${status}`, status));
  if (status === "running" || status === "queued") {
    head.appendChild(el("span", "spinner"));
    head.appendChild(el("span", "muted small", `${Math.round((Date.now() - state.startedAt) / 1000)}s`));
  }
  if (result.compile_ms) head.appendChild(el("span", "muted small", `compile ${Math.round(result.compile_ms)} ms`));
  if (result.wall_ms) head.appendChild(el("span", "muted small", `wall ${(result.wall_ms / 1000).toFixed(1)} s`));
  wrap.appendChild(head);

  const metrics = el("div", "metrics");
  const functional = result.functional;
  if (functional) {
    metrics.appendChild(metric("Functional", `${functional.passed}/${functional.total}`));
  }
  const perf = result.performance;
  if (perf) {
    metrics.appendChild(metric("Median", fmtMs(perf.median_ms)));
    metrics.appendChild(metric("Min / Max", `${fmtMs(perf.min_ms)} / ${fmtMs(perf.max_ms)}`));
    metrics.appendChild(metric("Spread", `${perf.spread_pct.toFixed(1)} %`));
    metrics.appendChild(metric("Throughput", `${perf.io_gbps.toFixed(1)} GB/s`));
  }
  if (metrics.childElementCount) wrap.appendChild(metrics);

  if (functional && functional.cases && functional.cases.length) {
    const list = el("ul", "case-list");
    for (const testCase of functional.cases) {
      const item = el(
        "li",
        testCase.passed ? "ok" : "bad",
        `case ${testCase.index}: ${testCase.passed ? "pass" : `fail${testCase.failed_param ? ` (${testCase.failed_param})` : ""}`}`
      );
      list.appendChild(item);
    }
    wrap.appendChild(list);
  }

  if (perf && perf.median_ms < 0.2 && (perf.inner_repeats || 1) === 1) {
    wrap.appendChild(
      el("div", "muted small", "Small kernel: set Inner repeats to 10 for a stable median.")
    );
  }

  const error = result.error;
  if (error) {
    const box = el("div", "error-box");
    box.textContent = `[${error.stage}] ${error.message || ""}`;
    if (error.case !== undefined) box.textContent += `\ncase: ${error.case}`;
    wrap.appendChild(box);
  }
  if (result.logs) {
    const box = el("div", "error-box");
    box.textContent = result.logs;
    wrap.appendChild(box);
  }
  return wrap;
}

function metric(key, value) {
  const box = el("div", "metric");
  box.appendChild(el("div", "k", key));
  box.appendChild(el("div", "v", value));
  return box;
}

$("#search").addEventListener("input", (event) => {
  state.filter = event.target.value;
  renderSidebar();
});
for (const chip of document.querySelectorAll("#filters .chip")) {
  chip.addEventListener("click", () => {
    state.difficulty = chip.dataset.diff;
    for (const other of document.querySelectorAll("#filters .chip")) {
      other.classList.toggle("active", other === chip);
    }
    renderSidebar();
  });
}
$("#token-btn").addEventListener("click", () => openTokenDialog("Paste the token configured on the runner service."));
$("#token-cancel").addEventListener("click", () => $("#token-dialog").close());
$("#token-form").addEventListener("submit", (event) => {
  event.preventDefault();
  setServiceBase($("#service-input").value);
  setToken($("#token-input").value.trim());
  $("#token-dialog").close();
  showBanner("");
  toast(getToken() ? "Token saved" : "Token cleared");
  const retry = state.retryAfterToken;
  state.retryAfterToken = false;
  if (retry && state.detail) startRun();
  else loadChallenges();
});
updateTokenButton();

boot();
