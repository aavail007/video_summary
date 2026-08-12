const CHUNK_MS = 20_000;

const elements = {
  provider: document.querySelector("#provider"),
  apiKey: document.querySelector("#api-key"),
  language: document.querySelector("#language"),
  geminiModel: document.querySelector("#gemini-model"),
  openaiModel: document.querySelector("#openai-model"),
  geminiField: document.querySelector("#gemini-model-field"),
  openaiField: document.querySelector("#openai-model-field"),
  micButton: document.querySelector("#mic-button"),
  screenButton: document.querySelector("#screen-button"),
  mixedButton: document.querySelector("#mixed-button"),
  stopButton: document.querySelector("#stop-button"),
  clearButton: document.querySelector("#clear-button"),
  downloadButton: document.querySelector("#download-button"),
  statusDot: document.querySelector("#status-dot"),
  statusText: document.querySelector("#status-text"),
  timer: document.querySelector("#timer"),
  chunkStatus: document.querySelector("#chunk-status"),
  transcript: document.querySelector("#transcript"),
  keyHint: document.querySelector("#key-hint"),
};

const state = {
  active: false,
  recorder: null,
  sourceStreams: [],
  recorderStream: null,
  audioContext: null,
  cycleTimer: null,
  timerId: null,
  startedAt: 0,
  queue: Promise.resolve(),
  queuedChunks: 0,
  entries: [],
  config: {},
  configLoaded: false,
  activeProvider: "gemini",
  enteredKeys: { gemini: "", openai: "" },
};

function formatTime(totalSeconds) {
  const value = Math.max(0, Math.floor(totalSeconds));
  const hours = String(Math.floor(value / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((value % 3600) / 60)).padStart(2, "0");
  const seconds = String(value % 60).padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
}

function setStatus(message, kind = "idle") {
  elements.statusText.textContent = message;
  elements.statusDot.className = kind === "idle" ? "" : kind;
}

function hasProviderKey() {
  const provider = elements.provider.value;
  const enteredKey = elements.apiKey.value.trim();
  const configuredKey = provider === "gemini"
    ? state.config.gemini_key_configured
    : state.config.openai_key_configured;
  return Boolean(enteredKey || configuredKey);
}

function setControls(recording) {
  const canStart = state.configLoaded && hasProviderKey();
  elements.micButton.disabled = recording || !canStart;
  elements.screenButton.disabled = recording || !canStart;
  elements.mixedButton.disabled = recording || !canStart;
  elements.stopButton.disabled = !recording;
  elements.provider.disabled = recording;
  elements.apiKey.disabled = recording;
  elements.language.disabled = recording;
  elements.geminiModel.disabled = recording;
  elements.openaiModel.disabled = recording;

  if (!recording && state.configLoaded && !canStart) {
    const serviceName = elements.provider.value === "gemini" ? "Gemini" : "OpenAI";
    setStatus(`請先輸入 ${serviceName} API Key 才能開始收音`, "error");
  } else if (
    !recording
    && canStart
    && elements.statusText.textContent.startsWith("請先輸入")
  ) {
    setStatus("API Key 已就緒，可以開始收音");
  }
}

function updateQueueStatus() {
  elements.chunkStatus.textContent = state.queuedChunks
    ? `正在等待 ${state.queuedChunks} 段轉錄完成…`
    : "";
}

function appendEntry(entry) {
  state.entries.push(entry);
  elements.transcript.className = "";
  const article = document.createElement("article");
  article.className = `entry${entry.error ? " error" : ""}`;
  const time = document.createElement("time");
  time.textContent = formatTime(entry.offset);
  const paragraph = document.createElement("p");
  paragraph.textContent = entry.text;
  article.append(time, paragraph);
  elements.transcript.append(article);
  article.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function supportedMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
}

async function uploadChunk(blob, offset) {
  state.queuedChunks += 1;
  updateQueueStatus();
  try {
    const response = await fetch("/api/live/transcribe", {
      method: "POST",
      headers: {
        "Content-Type": blob.type || "audio/webm",
        "X-AI-Provider": elements.provider.value,
        "X-API-Key": encodeURIComponent(elements.apiKey.value.trim()),
        "X-Language": elements.language.value,
        "X-Gemini-Model": encodeURIComponent(elements.geminiModel.value.trim()),
        "X-OpenAI-Model": encodeURIComponent(elements.openaiModel.value),
      },
      body: blob,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "轉錄失敗");
    if (result.text) appendEntry({ offset, text: result.text, error: false });
  } catch (error) {
    appendEntry({ offset, text: `⚠ ${error.message}`, error: true });
    setStatus("部分音訊轉錄失敗，錄音仍可繼續", "error");
  } finally {
    state.queuedChunks -= 1;
    updateQueueStatus();
    if (!state.active && state.queuedChunks === 0) {
      setStatus("收音已結束");
    }
  }
}

function recordNextChunk() {
  if (!state.active || !state.recorderStream) return;
  const parts = [];
  const mimeType = supportedMimeType();
  const options = mimeType ? { mimeType } : undefined;
  const offset = (Date.now() - state.startedAt) / 1000;
  const recorder = new MediaRecorder(state.recorderStream, options);
  state.recorder = recorder;
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size) parts.push(event.data);
  });
  recorder.addEventListener("stop", () => {
    clearTimeout(state.cycleTimer);
    const blob = new Blob(parts, { type: recorder.mimeType || mimeType || "audio/webm" });
    if (blob.size) {
      state.queue = state.queue.then(() => uploadChunk(blob, offset));
    }
    if (state.active) recordNextChunk();
  }, { once: true });
  recorder.start();
  state.cycleTimer = setTimeout(() => {
    if (recorder.state === "recording") recorder.stop();
  }, CHUNK_MS);
}

async function startRecording(kind) {
  if (!state.configLoaded || !hasProviderKey()) {
    setControls(false);
    return;
  }
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    setStatus("此瀏覽器不支援錄音功能，請改用最新版 Chrome 或 Edge", "error");
    return;
  }
  const acquiredStreams = [];
  let audioContext = null;
  try {
    let recorderStream;
    if (kind === "microphone") {
      const microphoneStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      acquiredStreams.push(microphoneStream);
      recorderStream = new MediaStream(microphoneStream.getAudioTracks());
    } else {
      const displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: true,
      });
      acquiredStreams.push(displayStream);
      if (!displayStream.getAudioTracks().length) {
        throw new Error("分享來源沒有音訊，請選擇分頁並勾選分享音訊");
      }
      displayStream.getVideoTracks().forEach((track) => {
        track.addEventListener("ended", stopRecording, { once: true });
      });

      if (kind === "mixed") {
        const microphoneStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
        acquiredStreams.push(microphoneStream);
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) throw new Error("此瀏覽器不支援麥克風與螢幕混音");
        audioContext = new AudioContextClass();
        await audioContext.resume();
        const destination = audioContext.createMediaStreamDestination();
        const displaySource = audioContext.createMediaStreamSource(displayStream);
        const microphoneSource = audioContext.createMediaStreamSource(microphoneStream);
        const displayGain = audioContext.createGain();
        const microphoneGain = audioContext.createGain();
        displayGain.gain.value = 1;
        microphoneGain.gain.value = 1;
        displaySource.connect(displayGain).connect(destination);
        microphoneSource.connect(microphoneGain).connect(destination);
        recorderStream = destination.stream;
      } else {
        recorderStream = new MediaStream(displayStream.getAudioTracks());
      }
    }

    state.sourceStreams = acquiredStreams;
    state.recorderStream = recorderStream;
    state.audioContext = audioContext;
    state.active = true;
    state.startedAt = Date.now();
    setControls(true);
    const statusMessages = {
      microphone: "麥克風收音中",
      screen: "螢幕音訊收音中",
      mixed: "會議混合收音中（麥克風＋分享音訊）",
    };
    setStatus(statusMessages[kind], "recording");
    elements.timer.textContent = "00:00:00";
    state.timerId = setInterval(() => {
      elements.timer.textContent = formatTime((Date.now() - state.startedAt) / 1000);
    }, 1000);
    recordNextChunk();
  } catch (error) {
    acquiredStreams.forEach((stream) => {
      stream.getTracks().forEach((track) => track.stop());
    });
    if (audioContext && audioContext.state !== "closed") audioContext.close();
    setStatus(`無法開始收音：${error.message}`, "error");
    setControls(false);
  }
}

function stopRecording() {
  if (!state.active) return;
  state.active = false;
  clearTimeout(state.cycleTimer);
  clearInterval(state.timerId);
  if (state.recorder?.state === "recording") state.recorder.stop();
  state.sourceStreams.forEach((stream) => {
    stream.getTracks().forEach((track) => track.stop());
  });
  state.recorderStream?.getTracks().forEach((track) => track.stop());
  if (state.audioContext && state.audioContext.state !== "closed") {
    state.audioContext.close();
  }
  state.sourceStreams = [];
  state.recorderStream = null;
  state.audioContext = null;
  setControls(false);
  setStatus("收音已停止，正在完成剩餘轉錄");
}

function clearTranscript() {
  state.entries = [];
  elements.transcript.className = "transcript-empty";
  elements.transcript.textContent = "開始收音後，逐字稿會顯示在這裡。";
}

function downloadMarkdown() {
  if (!state.entries.length) {
    setStatus("目前沒有可下載的逐字稿", "error");
    return;
  }
  const lines = ["# 即時收音逐字稿", ""];
  state.entries.forEach((entry) => {
    lines.push(`## ${formatTime(entry.offset)}`, "", entry.text, "");
  });
  const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `live-transcript-${new Date().toISOString().replace(/[:.]/g, "-")}.md`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function updateProviderFields() {
  const gemini = elements.provider.value === "gemini";
  elements.geminiField.hidden = !gemini;
  elements.openaiField.hidden = gemini;
  const configured = gemini
    ? state.config.gemini_key_configured
    : state.config.openai_key_configured;
  elements.keyHint.textContent = configured
    ? ".env 已設定此服務的 Key，也可在此暫時覆寫"
    : "尚未在 .env 設定，請在此輸入自己的 API Key";
  setControls(state.active);
}

function handleProviderChange() {
  state.enteredKeys[state.activeProvider] = elements.apiKey.value;
  state.activeProvider = elements.provider.value;
  elements.apiKey.value = state.enteredKeys[state.activeProvider] || "";
  updateProviderFields();
}

function handleApiKeyInput() {
  state.enteredKeys[elements.provider.value] = elements.apiKey.value;
  setControls(state.active);
}

async function loadConfig() {
  try {
    const response = await fetch("/api/live/config");
    state.config = await response.json();
    elements.geminiModel.value = state.config.gemini_model || "gemini-3.6-flash";
    elements.openaiModel.value = state.config.openai_model || "gpt-4o-transcribe";
  } finally {
    state.configLoaded = true;
    updateProviderFields();
  }
}

elements.provider.addEventListener("change", handleProviderChange);
elements.apiKey.addEventListener("input", handleApiKeyInput);
elements.micButton.addEventListener("click", () => startRecording("microphone"));
elements.screenButton.addEventListener("click", () => startRecording("screen"));
elements.mixedButton.addEventListener("click", () => startRecording("mixed"));
elements.stopButton.addEventListener("click", stopRecording);
elements.clearButton.addEventListener("click", clearTranscript);
elements.downloadButton.addEventListener("click", downloadMarkdown);
window.addEventListener("beforeunload", () => {
  state.active = false;
  state.sourceStreams.forEach((stream) => {
    stream.getTracks().forEach((track) => track.stop());
  });
  if (state.audioContext && state.audioContext.state !== "closed") {
    state.audioContext.close();
  }
});

loadConfig();
