/* Live caption client.
 *
 * Display model follows BBC/DCMP live-subtitle convention:
 *   - provisional text is dimmed + italic and may revise in place
 *   - it is replaced exactly once, at the utterance boundary, by full-contrast final text
 * Corrections are therefore batched at sentence boundaries rather than churning per word,
 * which is the behaviour recommended for deaf/HoH readers.
 */

const els = {
  captions: document.getElementById('captions'),
  placeholder: document.getElementById('placeholder'),
  dot: document.getElementById('dot'),
  state: document.getElementById('state'),
  meter: document.getElementById('meter'),
  meterFill: document.getElementById('meterFill'),
  latency: document.getElementById('latency'),
  toggle: document.getElementById('toggle'),
  toggleLabel: document.getElementById('toggleLabel'),
  bigger: document.getElementById('bigger'),
  smaller: document.getElementById('smaller'),
  clear: document.getElementById('clear'),
  speakerMenu: document.getElementById('speakerMenu'),
  speakerName: document.getElementById('speakerName'),
  speakerIsSelf: document.getElementById('speakerIsSelf'),
  speakerMerge: document.getElementById('speakerMerge'),
  speakerSave: document.getElementById('speakerSave'),
  speakerCancel: document.getElementById('speakerCancel'),
};

const MAX_LINES = 60;          // keep the DOM bounded during long sessions
const MIN_FONT = 18;
const MAX_FONT = 72;

let provisionalEl = null;
let currentId = null;
let socket = null;
let reconnectDelay = 500;
let running = true;

/* ---------- caption rendering ---------- */

const roster = new Map();   // speaker id -> { label, isSelf }

function clarityClass(score) {
  if (score >= 80) return 'good';
  if (score >= 55) return 'mid';
  return 'low';
}

/** Build/refresh one caption line: [speaker chip] text [clarity badge] */
function renderLine(el, msg, isFinal) {
  const id = msg.speaker_id;
  const info = id != null ? roster.get(id) : null;
  const isSelf = info ? info.isSelf : false;

  el.className = `line ${isFinal ? 'final' : 'provisional'}${isSelf ? ' self' : ''}`;
  el.textContent = '';
  if (id != null) el.dataset.speakerId = String(id);

  if (msg.speaker) {
    const chip = document.createElement('span');
    chip.className = `speaker spk-${(id ?? 0) % 8}`;
    chip.textContent = isSelf ? 'You' : msg.speaker;
    chip.title = 'Click to rename this speaker';
    if (id != null) chip.onclick = (ev) => openSpeakerMenu(ev, id);
    el.appendChild(chip);
  }

  const text = document.createElement('span');
  text.className = 'text';
  text.textContent = msg.text;
  el.appendChild(text);

  // Clarity is feedback for the user's own speech practice, so it's shown only on their
  // lines — on other people's it would just be noise.
  if (isFinal && isSelf && msg.clarity != null) {
    const badge = document.createElement('span');
    badge.className = `clarity ${clarityClass(msg.clarity)}`;
    badge.textContent = `clarity ${msg.clarity}%`;
    badge.title = 'How confidently the model decoded your speech. Relative, not absolute.';
    el.appendChild(badge);
  }
}

function hidePlaceholder() {
  if (els.placeholder) {
    els.placeholder.remove();
    els.placeholder = null;
  }
}

function atBottom() {
  const el = els.captions;
  return el.scrollHeight - el.scrollTop - el.clientHeight < 80;
}

function scrollIfFollowing(wasAtBottom) {
  if (wasAtBottom) els.captions.scrollTop = els.captions.scrollHeight;
}

function trimHistory() {
  while (els.captions.children.length > MAX_LINES) {
    els.captions.removeChild(els.captions.firstChild);
  }
}

function showProvisional(id, msg) {
  if (!msg.text) return;
  hidePlaceholder();
  const wasAtBottom = atBottom();

  if (!provisionalEl || currentId !== id) {
    provisionalEl = document.createElement('p');
    els.captions.appendChild(provisionalEl);
    currentId = id;
  }
  renderLine(provisionalEl, msg, false);
  trimHistory();
  scrollIfFollowing(wasAtBottom);
}

function commitFinal(id, msg) {
  const wasAtBottom = atBottom();

  if (!msg.text) {
    // Nothing intelligible in that utterance: drop the provisional line entirely.
    if (provisionalEl && currentId === id) provisionalEl.remove();
    provisionalEl = null;
    currentId = null;
    return;
  }

  hidePlaceholder();
  const line =
    provisionalEl && currentId === id ? provisionalEl : document.createElement('p');
  if (!line.parentNode) els.captions.appendChild(line);

  renderLine(line, msg, true);
  line.classList.add('recent');
  setTimeout(() => line.classList.remove('recent'), 300);

  provisionalEl = null;
  currentId = null;
  trimHistory();
  scrollIfFollowing(wasAtBottom);
}

/* ---------- status ---------- */

function setState(state, label) {
  els.dot.dataset.state = state;
  els.state.textContent = label || state;
}

function setRunning(next) {
  running = next;
  els.toggle.dataset.running = String(running);
  els.toggleLabel.textContent = running ? 'Pause' : 'Start';
  els.toggle.title = running
    ? 'Pause transcribing and release the microphone (Space)'
    : 'Start transcribing (Space)';
  els.captions.classList.toggle('stopped', !running);
  els.meter.classList.toggle('stopped', !running);
  if (!running) {
    // Drop the in-progress line: that audio is deliberately discarded server-side.
    if (provisionalEl) provisionalEl.remove();
    provisionalEl = null;
    currentId = null;
    els.meterFill.style.width = '0%';
    els.latency.textContent = '';
  }
}

function sendCommand(cmd, extra) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(Object.assign({ cmd }, extra || {})));
  }
}

function updateMeter(db, speaking) {
  if (!running) return;
  // Map -60..0 dBFS onto the bar.
  const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
  els.meterFill.style.width = pct + '%';
  els.meterFill.classList.toggle('hot', db > -3);
  els.dot.dataset.state = speaking ? 'speaking' : 'listening';
}

/* ---------- transport ---------- */

function handle(msg) {
  switch (msg.type) {
    case 'partial':
      showProvisional(msg.id, msg);
      els.latency.textContent = `~${Math.round(msg.latency_ms)} ms`;
      break;
    case 'final':
      commitFinal(msg.id, msg);
      els.latency.textContent = `${Math.round(msg.latency_ms)} ms final`;
      break;
    case 'roster':
      roster.clear();
      (msg.speakers || []).forEach((s) =>
        roster.set(s.id, { label: s.label, isSelf: !!s.is_self })
      );
      refreshRenderedSpeakers();
      break;
    case 'discard':
      if (provisionalEl && currentId === msg.id) provisionalEl.remove();
      provisionalEl = null;
      currentId = null;
      break;
    case 'level':
      updateMeter(msg.db, msg.speaking);
      break;
    case 'status':
      if (typeof msg.running === 'boolean') setRunning(msg.running);
      if (msg.state === 'loading') setState('loading', `loading ${msg.model || ''}`.trim());
      else if (msg.state === 'stopped') setState('stopped', 'paused — mic released');
      else if (msg.state === 'listening') setState('listening', msg.device || 'listening');
      else setState(msg.state, msg.state);
      break;
    case 'error':
      if (typeof msg.running === 'boolean') setRunning(msg.running);
      setState('error', msg.message || 'error');
      break;
  }
}

/** Re-label already-rendered lines after a rename, so history stays consistent. */
function refreshRenderedSpeakers() {
  els.captions.querySelectorAll('.speaker').forEach((chip) => {
    const cls = [...chip.classList].find((c) => c.startsWith('spk-'));
    if (!cls) return;
    const line = chip.parentElement;
    const id = Number(line.dataset.speakerId);
    if (Number.isNaN(id)) return;
    const info = roster.get(id);
    if (!info) return;
    chip.textContent = info.isSelf ? 'You' : info.label;
    line.classList.toggle('self', info.isSelf);
  });
}

/* ---------- speaker rename popover ---------- */

let menuSpeakerId = null;

function openSpeakerMenu(ev, id) {
  ev.stopPropagation();
  menuSpeakerId = id;
  const info = roster.get(id) || { label: '', isSelf: false };
  els.speakerName.value = info.label.startsWith('Speaker ') ? '' : info.label;
  els.speakerName.placeholder = info.label;
  els.speakerIsSelf.checked = info.isSelf;

  // Offer every other known speaker as a merge target.
  els.speakerMerge.innerHTML = '<option value="">— nobody —</option>';
  roster.forEach((other, otherId) => {
    if (otherId === id) return;
    const opt = document.createElement('option');
    opt.value = String(otherId);
    opt.textContent = other.isSelf ? 'You' : other.label;
    els.speakerMerge.appendChild(opt);
  });

  els.speakerMenu.hidden = false;
  const rect = ev.target.getBoundingClientRect();
  const menuW = els.speakerMenu.offsetWidth;
  els.speakerMenu.style.left =
    Math.max(8, Math.min(rect.left, window.innerWidth - menuW - 8)) + 'px';
  els.speakerMenu.style.top =
    Math.min(rect.bottom + 6, window.innerHeight - els.speakerMenu.offsetHeight - 8) + 'px';
  els.speakerName.focus();
}

function closeSpeakerMenu() {
  els.speakerMenu.hidden = true;
  menuSpeakerId = null;
}

function saveSpeakerMenu() {
  if (menuSpeakerId == null) return;
  const name = els.speakerName.value.trim();
  const mergeTarget = els.speakerMerge.value;

  if (name) sendCommand('rename_speaker', { id: menuSpeakerId, name });
  sendCommand('set_self', { id: menuSpeakerId, value: els.speakerIsSelf.checked });
  // Merge last: it renumbers ids, so any rename must be applied first.
  if (mergeTarget !== '') {
    sendCommand('merge_speakers', {
      source: menuSpeakerId,
      target: Number(mergeTarget),
    });
  }
  closeSpeakerMenu();
}

async function wsUrl() {
  let port = 8766;
  try {
    const res = await fetch('/config.json', { cache: 'no-store' });
    if (res.ok) port = (await res.json()).wsPort || port;
  } catch (_) {
    /* fall back to the default port */
  }
  return `ws://${location.hostname || '127.0.0.1'}:${port}`;
}

async function connect() {
  const url = await wsUrl();
  setState('connecting', 'connecting');
  socket = new WebSocket(url);

  socket.onopen = () => {
    reconnectDelay = 500;
    setState('listening', 'connected');
  };
  socket.onmessage = (ev) => {
    try {
      handle(JSON.parse(ev.data));
    } catch (err) {
      console.error('bad message', err);
    }
  };
  socket.onclose = () => {
    setState('disconnected', 'reconnecting…');
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 8000);
  };
  socket.onerror = () => socket.close();
}

/* ---------- controls ---------- */

function setFontSize(px) {
  const size = Math.max(MIN_FONT, Math.min(MAX_FONT, px));
  document.documentElement.style.setProperty('--caption-size', size + 'px');
  localStorage.setItem('captionSize', String(size));
}

function currentFontSize() {
  return parseInt(
    getComputedStyle(document.documentElement).getPropertyValue('--caption-size'),
    10
  );
}

els.bigger.onclick = () => setFontSize(currentFontSize() + 4);
els.smaller.onclick = () => setFontSize(currentFontSize() - 4);
els.toggle.onclick = () => sendCommand('toggle');
els.speakerSave.onclick = saveSpeakerMenu;
els.speakerCancel.onclick = closeSpeakerMenu;
els.speakerName.onkeydown = (e) => {
  if (e.key === 'Enter') saveSpeakerMenu();
  if (e.key === 'Escape') closeSpeakerMenu();
};
document.addEventListener('click', (e) => {
  if (!els.speakerMenu.hidden && !els.speakerMenu.contains(e.target)) closeSpeakerMenu();
});
els.clear.onclick = () => {
  els.captions.innerHTML = '';
  provisionalEl = null;
  currentId = null;
};

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && (e.key === '+' || e.key === '=')) setFontSize(currentFontSize() + 4);
  if (e.ctrlKey && e.key === '-') setFontSize(currentFontSize() - 4);
  // Space toggles capture, but not while typing a speaker name.
  if (e.key === ' ' && !e.ctrlKey && !e.altKey && e.target.tagName !== 'INPUT') {
    e.preventDefault();  // don't scroll the transcript
    sendCommand('toggle');
  }
});

const saved = parseInt(localStorage.getItem('captionSize') || '', 10);
if (!Number.isNaN(saved)) setFontSize(saved);

connect();
