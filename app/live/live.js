// ReorderPilot Live: the live-call copilot page (SPEC section 17, V4b; DECISIONS D31-D33).
//
//   rep microphone  ─┐                     ┌─> POST /copilot/transcribe?speaker=SALES_REP ─┐
//                    ├─ voice detection ───┤                                               ├─> transcript
//   call tab audio  ─┘  (per stream)       └─> POST /copilot/transcribe?speaker=CLIENT ────┘      │
//                                                                                      after a CLIENT line
//                                                                       POST /copilot/turn (latest wins) ─> cards
//
// Two streams = two speakers, so no guessing who spoke. Plain ES module, no build step, no framework.

const $ = (id) => document.getElementById(id);
const TARGET_RATE = 16000;       // Whisper works at 16 kHz; smaller uploads, same accuracy
const VAD = {
  minThreshold: 0.012,           // RMS below this is always silence
  floorFactor: 3,                // speech = louder than 3x the measured background noise
  startFrames: 2,                // ~85 ms above threshold before an utterance starts (ignores clicks)
  endSilenceMs: 700,             // a pause this long ends the utterance
  maxMs: 15000,                  // cut very long monologues so cards keep coming
  minVoicedMs: 300,              // shorter bursts are noise, not words
  preRollMs: 300,                // keep audio from just before detection so first syllables are not lost
  gateMs: 300,                   // rep mic ignored while (and just after) the client speaks: speaker echo
};
const METER_BARS = 18;
const CUSTOMER_RE = /^[0-9A-Za-z]{1,20}$/;

const app = {
  asOf: "",
  audio: null,                                   // one AudioContext for both streams
  listeners: { SALES_REP: null, CLIENT: null },
  call: null,                                    // {id, mode, lines, turn, lastTurn, timer, ...}
  callSeq: 0,
};

// ---------- small helpers ----------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const secs = (ms) => `${(ms / 1000).toFixed(1)} s`;
const customer = () => $("customer-input").value.trim();
const other = (s) => (s === "CLIENT" ? "SALES_REP" : "CLIENT");
const isCurrent = (callId) => app.call !== null && app.call.id === callId;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;   // textContent everywhere: transcript/LLM text is untrusted
  return node;
}

function toast(message, kind = "info", ms = 7000) {
  const node = el("div", `toast ${kind}`, message);
  $("toasts").append(node);
  setTimeout(() => node.remove(), ms);
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  let body = null;
  try { body = await res.json(); } catch { /* empty or non-JSON body */ }
  if (!res.ok) {
    const d = body && body.detail;
    const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((e) => e.msg).join("; ") : `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return body;
}

const postJSON = (path, data) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

function remember(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private mode or blocked storage: not needed to work */ }
}
function recall(key) {
  try { return localStorage.getItem(key) || ""; } catch { return ""; }
}

// ---------- level meters ----------

function buildMeter(node) {
  node.history = new Array(METER_BARS).fill(0);
  for (let i = 0; i < METER_BARS; i++) node.append(el("i"));
}

function drawMeter(node, rms, speaking) {
  node.history.push(Math.min(1, rms * 14));
  node.history.shift();
  node.dataset.active = String(speaking);
  node.querySelectorAll("i").forEach((bar, i) => bar.style.setProperty("--h", node.history[i].toFixed(2)));
}

// ---------- audio: one Listener per stream ----------

async function ensureAudio() {
  if (!app.audio) {
    app.audio = new AudioContext();
    await app.audio.audioWorklet.addModule("frame-tap.js");
  }
  if (app.audio.state === "suspended") await app.audio.resume();
  return app.audio;
}

class Listener {
  /** Voice detection on one stream: emits utterances (raw samples) when someone speaks and then pauses. */
  constructor(speaker, stream) {
    this.speaker = speaker;
    this.stream = stream;
    this.meter = $(speaker === "CLIENT" ? "client-meter" : "rep-meter");
    this.active = false;        // only detect utterances during a live call
    this.phase = "idle";
    this.floor = 0.004;         // running estimate of background noise (RMS)
    this.hot = 0;
    this.pre = [];
    this.buf = [];
    this.lastEnd = 0;
    this.gate = () => false;
  }

  async start(ctx) {
    this.rate = ctx.sampleRate;
    this.src = ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(ctx, "frame-tap");
    this.sink = ctx.createGain();
    this.sink.gain.value = 0;   // the graph must reach the speakers to run, but we never play the audio back
    this.src.connect(this.node).connect(this.sink).connect(ctx.destination);
    this.node.port.onmessage = (e) => this.onFrame(e.data);
  }

  stop() {
    this.active = false;
    this.node?.port.close();
    this.src?.disconnect();
    this.node?.disconnect();
    this.stream.getTracks().forEach((t) => t.stop());
    drawMeter(this.meter, 0, false);
  }

  speaking() { return this.phase === "speaking"; }

  onFrame(frame) {
    const ms = (frame.length / this.rate) * 1000;
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    const rms = Math.sqrt(sum / frame.length);
    drawMeter(this.meter, rms, this.speaking());
    const threshold = Math.max(VAD.minThreshold, this.floor * VAD.floorFactor);

    if (this.phase === "idle") {
      this.pre.push(frame);
      while (this.pre.length * ms > VAD.preRollMs) this.pre.shift();
      if (this.active && rms > threshold && !this.gate()) {
        if (++this.hot >= VAD.startFrames) this.begin(ms);
      } else {
        this.hot = 0;
        this.floor = this.floor * 0.95 + Math.min(rms, 0.05) * 0.05;
      }
      return;
    }
    this.buf.push(frame);
    this.durMs += ms;
    if (rms > threshold) { this.voicedMs += ms; this.silenceMs = 0; } else { this.silenceMs += ms; }
    if (this.silenceMs >= VAD.endSilenceMs || this.durMs >= VAD.maxMs || !this.active) this.finish();
  }

  begin(ms) {
    this.phase = "speaking";
    this.buf = this.pre;
    this.pre = [];
    this.durMs = this.buf.length * ms;
    this.voicedMs = VAD.startFrames * ms;
    this.silenceMs = 0;
    this.line = onSpeechStart(this);
  }

  finish() {
    const chunks = this.buf;
    const line = this.line;
    const keep = this.voicedMs >= VAD.minVoicedMs && this.active;
    Object.assign(this, { phase: "idle", buf: [], hot: 0, line: null, lastEnd: performance.now() });
    onSpeechEnd(this, line, keep ? chunks : null);
  }
}

function toWav(chunks, inRate) {
  // Average-downsample to 16 kHz mono and wrap as 16-bit PCM WAV (Groq accepts wav; ~32 KB per second).
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const c of chunks) { merged.set(c, offset); offset += c.length; }
  const ratio = inRate / TARGET_RATE;
  const outLen = Math.floor(total / ratio);
  const view = new DataView(new ArrayBuffer(44 + outLen * 2));
  const text = (pos, s) => { for (let i = 0; i < s.length; i++) view.setUint8(pos + i, s.charCodeAt(i)); };
  text(0, "RIFF"); view.setUint32(4, 36 + outLen * 2, true); text(8, "WAVE");
  text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, TARGET_RATE, true); view.setUint32(28, TARGET_RATE * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, "data"); view.setUint32(40, outLen * 2, true);
  for (let i = 0; i < outLen; i++) {
    const from = Math.floor(i * ratio);
    const to = Math.min(total, Math.floor((i + 1) * ratio));
    let s = 0;
    for (let j = from; j < to; j++) s += merged[j];
    const v = Math.max(-1, Math.min(1, s / Math.max(1, to - from)));
    view.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

// ---------- sources (connect / disconnect) ----------

function micError(e) {
  if (!navigator.mediaDevices) return "Audio capture needs a secure page: open it via http://localhost.";
  if (e.name === "NotAllowedError") return "Microphone permission was denied. Allow it in the address bar and retry.";
  if (e.name === "NotFoundError") return "No microphone found.";
  return `Could not open the microphone: ${e.message}`;
}

async function toggleMic() {
  if (app.listeners.SALES_REP) return detach("SALES_REP");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    await attach("SALES_REP", stream);
  } catch (e) {
    toast(micError(e), "error");
  }
}

async function toggleTab() {
  if (app.listeners.CLIENT) return detach("CLIENT");
  try {
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: true,  // browsers require video to share a tab; it is disabled right away and never sent anywhere
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      selfBrowserSurface: "exclude", systemAudio: "include",
    });
    if (!stream.getAudioTracks().length) {
      stream.getTracks().forEach((t) => t.stop());
      toast("The shared tab has no audio. Share again and tick “Share tab audio”.", "error");
      return;
    }
    stream.getVideoTracks().forEach((t) => { t.enabled = false; });
    await attach("CLIENT", stream);
  } catch (e) {
    toast(e.name === "NotAllowedError" ? "Tab sharing was cancelled." : `Could not share the tab: ${e.message}`,
      e.name === "NotAllowedError" ? "warn" : "error");
  }
}

async function attach(speaker, stream) {
  const ctx = await ensureAudio();
  app.listeners[speaker]?.stop();
  const listener = new Listener(speaker, stream);
  await listener.start(ctx);
  if (speaker === "SALES_REP") {
    listener.gate = () => {
      const c = app.listeners.CLIENT;
      return !!c && (c.speaking() || performance.now() - c.lastEnd < VAD.gateMs);
    };
  }
  stream.getTracks().forEach((t) => t.addEventListener("ended", () => detach(speaker)));
  listener.active = app.call?.mode === "live";
  app.listeners[speaker] = listener;
  renderSources();
}

function detach(speaker) {
  const listener = app.listeners[speaker];
  if (!listener) return;
  app.listeners[speaker] = null;
  listener.stop();
  renderSources();
  if (app.call?.mode === "live" && !app.listeners.SALES_REP && !app.listeners.CLIENT) {
    toast("Both audio sources are disconnected; the call is paused until you reconnect one.", "warn");
  }
}

function renderSources() {
  const live = app.call?.mode === "live";
  for (const [speaker, ids] of Object.entries({
    SALES_REP: { box: "src-rep", state: "rep-state", btn: "connect-mic", on: "Disconnect", off: "Connect microphone", idle: "Microphone off" },
    CLIENT: { box: "src-client", state: "client-state", btn: "connect-tab", on: "Stop sharing", off: "Share call tab audio", idle: "Not shared" },
  })) {
    const connected = !!app.listeners[speaker];
    $(ids.box).dataset.live = String(connected && live);
    $(ids.btn).textContent = connected ? ids.on : ids.off;
    $(ids.state).textContent = !connected ? ids.idle : live ? "Listening" : "Connected · ready";
  }
  const micOnly = app.listeners.SALES_REP && !app.listeners.CLIENT;
  $("transcript-hint").textContent = micOnly
    ? "Microphone only: speakers are guessed. Click a line to swap it."
    : "Click a line to swap its speaker";
  renderButtons();
}

function renderButtons() {
  const okCustomer = CUSTOMER_RE.test(customer());
  const inCall = app.call !== null;
  const anySource = !!(app.listeners.SALES_REP || app.listeners.CLIENT);
  $("start-live").disabled = !okCustomer || !anySource;
  $("start-demo").disabled = !okCustomer;
  $("start-live").title = !okCustomer ? "Pick a customer first" : !anySource ? "Connect your microphone or the call tab first" : "";
  for (const id of ["start-live", "start-demo"]) $(id).hidden = inCall;
  $("demo-voice").parentElement.hidden = inCall;
  $("end-call").hidden = !inCall;
  $("customer-input").disabled = inCall;
}

// ---------- the call ----------

function startCall(mode) {
  app.call = {
    id: ++app.callSeq, mode, lines: [], lineSeq: 0, startedAt: Date.now(),
    turn: { inFlight: false, queued: false }, lastTurn: null,
    timer: setInterval(renderTimer, 500),
  };
  $("transcript").replaceChildren();
  $("cards").replaceChildren(el("p", "muted", mode === "live"
    ? "Listening… cards appear after the client's first sentence."
    : "Demo call starting… cards appear after the client's first line."));
  $("context").replaceChildren();
  $("why").hidden = true;
  setStatus("idle", "Listening");
  for (const l of Object.values(app.listeners)) if (l) l.active = mode === "live";
  renderSources();
  renderTimer();
}

function closeCall() {
  if (!app.call) return;
  clearInterval(app.call.timer);
  app.call = null;
  window.speechSynthesis?.cancel();
  for (const l of Object.values(app.listeners)) if (l) l.active = false;
  setStatus("idle", "Call ended");
  $("timer").dataset.live = "false";
  renderSources();
}

function renderTimer() {
  const t = $("timer");
  if (!app.call) return;
  const s = Math.floor((Date.now() - app.call.startedAt) / 1000);
  t.textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  t.dataset.live = "true";
}

function addLine(speaker, status, t0) {
  const line = { id: ++app.call.lineSeq, callId: app.call.id, speaker, text: "", status, t0, latency: null };
  app.call.lines.push(line);
  $("transcript-empty")?.remove();
  renderLine(line);
  return line;
}

function removeLine(line) {
  if (isCurrent(line.callId)) app.call.lines = app.call.lines.filter((l) => l !== line);
  document.querySelector(`[data-line="${line.id}"]`)?.remove();
}

function renderLine(line) {
  const box = $("transcript");
  let node = box.querySelector(`[data-line="${line.id}"]`);
  if (!node) {
    node = el("div", "line");
    node.dataset.line = String(line.id);
    node.dataset.t0 = String(line.t0);
    node.append(el("div", "meta"), el("div", "bubble"));
    node.querySelector(".bubble").addEventListener("click", () => flipSpeaker(line));
    // keep spoken order even when transcriptions come back out of order
    const after = [...box.children].find((c) => Number(c.dataset.t0) > line.t0);
    box.insertBefore(node, after || null);
  }
  node.className = `line ${line.speaker === "CLIENT" ? "client" : "rep"}`;
  node.dataset.status = line.status;
  const meta = node.querySelector(".meta");
  meta.replaceChildren(el("b", "", line.speaker === "CLIENT" ? "Client" : "You"));
  if (line.latency !== null) meta.append(el("span", "", `heard in ${secs(line.latency)}`));
  const bubble = node.querySelector(".bubble");
  if (line.status === "speaking" && !line.text) {
    bubble.replaceChildren(el("span", "typing"));
    bubble.firstChild.append(el("i"), el("i"), el("i"));
  } else if (line.status === "pending") {
    bubble.replaceChildren(el("span", "", "Transcribing "), el("span", "typing"));
    bubble.lastChild.append(el("i"), el("i"), el("i"));
  } else {
    bubble.textContent = line.text;
  }
  box.scrollTop = box.scrollHeight;
}

function transcriptLines() {
  return app.call.lines
    .filter((l) => l.status === "done" && l.text)
    .sort((a, b) => a.t0 - b.t0)
    .map((l) => `[${l.speaker}] ${l.text}`);
}

function flipSpeaker(line) {
  if (!app.call || line.status !== "done") return;
  line.speaker = other(line.speaker);
  renderLine(line);
  requestTurn();
}

// ---------- live audio -> transcript ----------

function onSpeechStart(listener) {
  if (!app.call || app.call.mode !== "live") return null;
  return addLine(listener.speaker, "speaking", performance.now() - VAD.preRollMs);
}

async function onSpeechEnd(listener, line, chunks) {
  if (!line) return;
  if (!chunks || !isCurrent(line.callId)) { removeLine(line); return; }
  const callId = line.callId;
  line.status = "pending";
  renderLine(line);
  // Mic only: the speaker is unknown, so the server guesses from cue words / turn-taking (D26 rules).
  const micOnly = !app.listeners.CLIENT;
  const previous = [...app.call.lines].reverse().find((l) => l.status === "done");
  const params = new URLSearchParams({ speaker: micOnly ? "AUTO" : listener.speaker });
  if (micOnly && previous) params.set("previous", previous.speaker);
  const sent = performance.now();
  try {
    const res = await api(`/copilot/transcribe?${params}`, {
      method: "POST", headers: { "Content-Type": "audio/wav" }, body: toWav(chunks, listener.rate),
    });
    if (!isCurrent(callId)) return;
    if (!res.line) { removeLine(line); return; }   // silence or a Whisper "thank you" hallucination
    const [, speaker, text] = res.line.match(/^\[(CLIENT|SALES_REP)\] (.*)$/s);
    Object.assign(line, { speaker, text, status: "done", latency: performance.now() - sent });
    renderLine(line);
    if (speaker === "CLIENT") requestTurn();
  } catch (e) {
    if (!isCurrent(callId)) return;
    Object.assign(line, { status: "error", text: e.message });
    renderLine(line);
    toast(e.message, "error");
  }
}

// ---------- transcript -> battle cards (one request at a time, latest wins) ----------

function requestTurn() {
  const call = app.call;
  if (!call) return;
  const lines = transcriptLines();
  if (!lines.some((l) => l.startsWith("[CLIENT]"))) return;
  if (call.turn.inFlight) { call.turn.queued = true; return; }   // run again with the newest lines when done
  call.turn.inFlight = true;
  setStatus("thinking", "Thinking…");
  $("cards").dataset.stale = String(!!call.lastTurn);
  if (!call.lastTurn) $("cards").replaceChildren(el("div", "skeleton"), el("div", "skeleton"));
  const started = performance.now();
  postJSON("/copilot/turn", { customer_id: customer(), transcript: lines })
    .then((res) => {
      if (!isCurrent(call.id)) return;
      call.lastTurn = res;
      renderTurn(res, performance.now() - started);
    })
    .catch((e) => {
      if (!isCurrent(call.id)) return;
      setStatus("error", "Copilot error");
      toast(e.message, "error");
    })
    .finally(() => {
      if (!isCurrent(call.id)) return;
      call.turn.inFlight = false;
      $("cards").dataset.stale = "false";
      if (call.turn.queued) { call.turn.queued = false; requestTurn(); }
    });
}

function renderTurn(res, roundTripMs) {
  const cards = $("cards");
  cards.replaceChildren();
  res.cards.forEach((c, i) => {
    const card = el("article", i === 0 ? "card lead" : "card");
    if (i === 0) card.append(el("div", "card-kicker", "Say this"));
    card.append(el("div", "card-title", c.title), el("div", "card-say", c.say));
    if (c.evidence) card.append(el("div", "card-evidence", c.evidence));
    cards.append(card);
  });
  if (!res.cards.length) cards.append(el("p", "muted", "No suggestion for this moment."));

  const fallback = res.stop_reason.startsWith("fallback");
  setStatus(fallback ? "error" : "ready", fallback ? `Fallback cards · ${secs(roundTripMs)}` : `Updated · ${secs(roundTripMs)}`);
  if (fallback) toast("The LLM was unavailable, so these cards come from the model prediction and memory only.", "warn");

  const context = $("context");
  context.replaceChildren();
  if (res.price_sensitive) context.append(el("span", "tag warn", "Price-sensitive"));
  res.declinable.slice(0, 3).forEach((p) => context.append(el("span", "tag", `Due · ${p.description.toLowerCase()}`)));

  const trail = $("trail");
  trail.replaceChildren();
  res.trail.forEach((node, i) => {
    if (i) trail.append(el("em", "", "›"));
    trail.append(el("span", "", node));
  });
  trail.append(el("em", "", ` stop: ${res.stop_reason} · server ${secs(res.latency_ms)}`));
  const removed = $("removed");
  removed.hidden = !res.removed_codes.length;
  removed.textContent = `Grounding guard removed cards citing unknown stock codes: ${res.removed_codes.join(", ")}`;
  const memories = $("memories");
  memories.replaceChildren();
  if (!res.memories.length) memories.append(el("li", "muted", "No notes or objections found for this customer."));
  res.memories.forEach((m) => {
    const li = el("li");
    li.append(el("small", "", `${m.source} · ${m.created_at} · distance ${m.distance}`), el("span", "", m.text));
    memories.append(li);
  });
  $("why").hidden = false;
}

function setStatus(state, text) {
  const pill = $("copilot-status");
  pill.dataset.state = state;
  pill.textContent = text;
}

// ---------- demo call (scripted lines, spoken by two browser voices) ----------

async function loadVoices(maxMs = 1500) {
  // Browsers load voices asynchronously; the first getVoices() call often returns [].
  const synth = window.speechSynthesis;
  if (!synth || synth.getVoices().length) return;
  await Promise.race([new Promise((r) => synth.addEventListener("voiceschanged", r, { once: true })), sleep(maxMs)]);
}

function pickVoices() {
  const all = window.speechSynthesis?.getVoices() || [];
  const english = all.filter((v) => v.lang.startsWith("en"));
  const pool = english.length ? english : all;
  const find = (names) => pool.find((v) => names.some((n) => v.name.includes(n)));
  const rep = find(["David", "Daniel", "Guy", "Alex", "Ryan", "Male"]) || pool[0];
  const client = find(["Zira", "Samantha", "Aria", "Jenny", "Sonia", "Female"]) || pool.find((v) => v !== rep) || rep;
  return { SALES_REP: rep, CLIENT: client };
}

function speakInto(line, text, voice, callId) {
  const words = text.split(/\s+/);
  const reveal = (upto) => { if (isCurrent(callId)) { line.text = upto; renderLine(line); } };
  const useVoice = $("demo-voice").checked && window.speechSynthesis && voice;
  if (!useVoice) {   // no audio: reveal word by word, like live captions
    return (async () => {
      for (let i = 1; i <= words.length; i++) { if (!isCurrent(callId)) return; reveal(words.slice(0, i).join(" ")); await sleep(170); }
    })();
  }
  return new Promise((resolve) => {
    const u = new SpeechSynthesisUtterance(text);
    u.voice = voice;
    u.rate = 1.08;
    u.onboundary = (e) => { if (e.name === "word") reveal(text.slice(0, e.charIndex + (e.charLength || 0))); };
    const done = () => { clearTimeout(guard); resolve(); };
    const guard = setTimeout(done, Math.max(4000, words.length * 600));   // some voices never fire onend
    u.onend = done;
    u.onerror = done;
    window.speechSynthesis.speak(u);
  });
}

async function runDemo() {
  startCall("demo");
  const callId = app.call.id;
  let lines;
  try {
    ({ lines } = await api("/copilot/demo"));
  } catch (e) {
    toast(e.message, "error");
    closeCall();
    return;
  }
  await loadVoices();
  const voices = pickVoices();
  for (const raw of lines) {
    if (!isCurrent(callId)) return;
    const [, speaker, text] = raw.match(/^\[(CLIENT|SALES_REP)\] (.*)$/);
    const line = addLine(speaker, "speaking", performance.now());
    await speakInto(line, text, voices[speaker], callId);
    if (!isCurrent(callId)) return;
    Object.assign(line, { text, status: "done" });
    renderLine(line);
    if (speaker === "CLIENT") requestTurn();
    await sleep(450);
  }
  if (isCurrent(callId)) toast("Demo call finished. Press End call to save what you learned.", "info");
}

// ---------- end call & persist ----------

async function waitForCards(maxMs = 20000) {
  const call = app.call;
  const until = performance.now() + maxMs;
  while (call.turn.inFlight || call.turn.queued) {
    if (performance.now() > until) break;
    await sleep(100);
  }
}

async function openEndDialog() {
  const call = app.call;
  if (!call) return;
  for (const l of Object.values(app.listeners)) if (l) l.active = false;   // stop listening while saving
  window.speechSynthesis?.cancel();
  setStatus("thinking", "Finishing…");
  await waitForCards();   // so the suggested note includes the client's last sentence
  if (!isCurrent(call.id)) return;
  const turn = call.lastTurn;
  $("end-customer").textContent = `customer ${customer()}`;
  $("end-note").value = turn?.suggested_note || `Live call ${app.asOf}.`;
  $("end-count").textContent = String($("end-note").value.length);
  const choices = $("end-declined");
  choices.replaceChildren();
  (turn?.declinable || []).forEach((p) => {
    const label = el("label", "choice");
    const box = el("input");
    box.type = "checkbox";
    box.value = p.stock_code;
    label.append(box, el("span", "", `${p.stock_code} · ${p.description.toLowerCase()}`));
    choices.append(label);
  });
  if (!choices.children.length) choices.append(el("span", "muted", "No product suggestions in this call."));
  $("end-result").hidden = true;
  $("end-save").hidden = false;
  $("end-save").disabled = false;
  $("end-done").hidden = true;
  setStatus("ready", "Call paused");
  $("end-dialog").showModal();
}

function resumeCall() {
  $("end-dialog").close();
  if (app.call?.mode === "live") for (const l of Object.values(app.listeners)) if (l) l.active = true;
  setStatus("ready", "Listening");
}

async function saveCall(event) {
  event.preventDefault();
  const call = app.call;
  if (!call) return;
  const save = $("end-save");
  save.disabled = true;
  save.textContent = "Saving…";
  const declined = [...$("end-declined").querySelectorAll("input:checked")].map((b) => b.value);
  const result = $("end-result");
  try {
    const res = await postJSON("/copilot/end_call", {
      customer_id: customer(), transcript: transcriptLines(), note: $("end-note").value.trim(),
      declined, reason: $("end-reason").value,
    });
    result.className = "result";
    result.replaceChildren(el("b", "", `Saved note #${res.note.id} (${res.note.created_at}).`));
    if (res.effects.length) {
      const ul = el("ul");
      res.effects.forEach((e) => ul.append(el("li", "", `Declined: ${e}`)));
      result.append(ul);
    }
    result.append(el("div", "", res.vector_error ? `Vector memory check failed: ${res.vector_error}`
      : res.vector_rank ? `Vector memory: the next call will retrieve this note (rank ${res.vector_rank}, distance ${res.vector_distance}).`
        : res.vector_indexed ? "Vector memory: indexed; retrieved when the client raises similar topics."
          : "Vector memory: the note is not in the index."));
    save.hidden = true;
    $("end-done").hidden = false;
  } catch (e) {
    result.className = "result bad";
    result.textContent = e.message;
    save.disabled = false;
  } finally {
    save.textContent = "Save to memory";
    result.hidden = false;
  }
}

function finishCall() {
  $("end-dialog").close();
  closeCall();
}

function discardCall() {
  if (!window.confirm("Discard this call? Nothing will be saved.")) return;
  finishCall();
}

// ---------- start-up ----------

async function loadHeader() {
  try {
    const health = await api("/health");
    app.asOf = health.as_of;
    $("asof-chip").textContent = `As of ${health.as_of}`;
    $("api-chip").dataset.ok = "true";
    $("api-text").textContent = "Connected";
  } catch (e) {
    $("api-chip").dataset.ok = "false";
    $("api-text").textContent = "API offline";
    toast(`API not reachable: ${e.message}`, "error");
  }
  try {
    const due = await api("/customers/due?limit=15");
    const list = $("customer-list");
    due.customers.forEach((c) => {
      const option = el("option");
      option.value = c.customer_id;
      option.label = `£${Math.round(c.expected_value)} expected · ${c.n_candidates} items due`;
      list.append(option);
    });
    if (!customer() && due.customers.length) $("customer-input").value = due.customers[0].customer_id;
  } catch { /* the field still accepts any ID */ }
  renderButtons();
}

function init() {
  buildMeter($("rep-meter"));
  buildMeter($("client-meter"));
  $("customer-input").value = recall("rp.customer");
  $("customer-input").addEventListener("input", () => { remember("rp.customer", customer()); renderButtons(); });
  $("connect-mic").addEventListener("click", toggleMic);
  $("connect-tab").addEventListener("click", toggleTab);
  $("start-live").addEventListener("click", async () => { await ensureAudio(); startCall("live"); });
  $("start-demo").addEventListener("click", runDemo);
  $("end-call").addEventListener("click", openEndDialog);
  $("end-form").addEventListener("submit", saveCall);
  $("end-cancel").addEventListener("click", resumeCall);
  $("end-done").addEventListener("click", finishCall);
  $("end-discard").addEventListener("click", discardCall);
  $("end-dialog").addEventListener("cancel", (e) => { e.preventDefault(); resumeCall(); });   // Esc = back to call
  $("end-note").addEventListener("input", () => { $("end-count").textContent = String($("end-note").value.length); });
  window.speechSynthesis?.getVoices();   // voices load asynchronously; ask early so the demo has them
  window.addEventListener("beforeunload", (e) => { if (app.call) e.preventDefault(); });
  renderSources();
  loadHeader();
}

init();
