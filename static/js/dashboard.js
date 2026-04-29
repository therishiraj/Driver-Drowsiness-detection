/**
 * DrowsyGuard Dashboard — Frontend Logic
 * Polls /agent_state every 300ms and updates all UI elements.
 * Plays audio alert via Web Audio API as fallback.
 */

'use strict';

// ── Config ───────────────────────────────────────────────────────────────
const POLL_INTERVAL = 300;   // ms
const LOG_MAX = 50;
let audioEnabled = true;
let lastState = null;
let audioCtx = null;
let alertActive = false;
let alertInterval = null;

// ── Initialise ───────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  pollAgentState();
  setInterval(pollAgentState, POLL_INTERVAL);
  log('Agent dashboard initialised.', 'info');
});

// ── Polling ──────────────────────────────────────────────────────────────
async function pollAgentState() {
  try {
    const res = await fetch('/agent_state');
    if (!res.ok) return;
    const data = await res.json();
    updateDashboard(data);
  } catch (e) {
    // Silently ignore network errors (camera may be loading)
  }
}

// ── Dashboard Update ─────────────────────────────────────────────────────
function updateDashboard(d) {
  const state = d.state;

  // State card
  const card = document.getElementById('stateCard');
  const sv   = document.getElementById('stateValue');
  const sub  = document.getElementById('stateSub');
  const ov   = document.getElementById('overlayState');

  card.className = 'state-card ' + state.toLowerCase().replace('_', '-');
  sv.textContent = state;
  ov.textContent = state;

  const stateMessages = {
    AWAKE:    'Driver appears alert and attentive.',
    DROWSY:   '⚠ Fatigue detected — please rest!',
    SLEEPING: '🚨 Driver asleep — EMERGENCY ALERT!',
    NO_FACE:  'No face detected in frame.'
  };
  sub.textContent = stateMessages[state] || '';

  // Colour state value
  const colours = { AWAKE:'#22c55e', DROWSY:'#f59e0b', SLEEPING:'#ef4444', NO_FACE:'#64748b' };
  sv.style.color = colours[state] || '#e2e8f0';

  // EAR bar
  const ear = d.ear || 0;
  const earPct = Math.min((ear / 0.4) * 100, 100);
  document.getElementById('earValue').textContent = ear.toFixed(3);
  const earBar = document.getElementById('earBar');
  earBar.style.width = earPct + '%';
  earBar.style.background = ear < 0.25 ? '#ef4444' : ear < 0.30 ? '#f59e0b' : '#3b82f6';

  // Consecutive closed frames bar
  const cons = d.consecutive_closed || 0;
  const thresh = d.alert_threshold_frames || 20;
  const consPct = Math.min((cons / thresh) * 100, 100);
  document.getElementById('consValue').textContent = `${cons} / ${thresh}`;
  const consBar = document.getElementById('consBar');
  consBar.style.width = consPct + '%';
  consBar.style.background = consPct > 80 ? '#ef4444' : consPct > 50 ? '#f59e0b' : '#f59e0b';

  // Stats
  document.getElementById('statBlinks').textContent = d.total_blinks || 0;
  document.getElementById('statFPS').textContent    = (d.fps || 0).toFixed(0);
  document.getElementById('statFrames').textContent = d.frame_count || 0;
  document.getElementById('statSession').textContent = formatDuration(d.session_duration || 0);
  document.getElementById('inferenceTime').textContent = `Inference: ~${(1000 / Math.max(d.fps || 1, 1)).toFixed(0)}ms`;

  // Alert handling
  handleAlerts(state, d.alert_active);

  // Log state transitions
  if (state !== lastState) {
    const logClass = { DROWSY:'warning', SLEEPING:'danger', AWAKE:'success', NO_FACE:'info' }[state] || 'info';
    log(`State → ${state}`, logClass);
    lastState = state;
  }
}

// ── Alert Handling ───────────────────────────────────────────────────────
function handleAlerts(state, serverAlertActive) {
  const banner = document.getElementById('alertBanner');
  const title  = document.getElementById('alertTitle');
  const msg    = document.getElementById('alertMsg');
  const icon   = document.getElementById('alertIcon');

  if (state === 'DROWSY' || state === 'SLEEPING') {
    banner.style.display = 'flex';
    if (state === 'SLEEPING') {
      title.textContent = '🚨 DRIVER SLEEPING — EMERGENCY';
      msg.textContent   = 'Vehicle should pull over immediately!';
      icon.textContent  = '🚨';
      banner.style.borderBottomColor = '#ef4444';
    } else {
      title.textContent = '⚠ DROWSINESS DETECTED';
      msg.textContent   = 'Eyes closed too long. Please take a break.';
      icon.textContent  = '⚠️';
      banner.style.borderBottomColor = '#f59e0b';
    }
    // Web audio beep as fallback
    if (audioEnabled && !alertActive) {
      alertActive = true;
      playWebAudioBeep(state === 'SLEEPING' ? 880 : 440);
      alertInterval = setInterval(() => {
        playWebAudioBeep(state === 'SLEEPING' ? 880 : 440);
      }, 2000);
    }
  } else {
    banner.style.display = 'none';
    if (alertActive) {
      clearInterval(alertInterval);
      alertActive = false;
    }
  }
}

function dismissAlert() {
  document.getElementById('alertBanner').style.display = 'none';
}

// ── Web Audio Beep ───────────────────────────────────────────────────────
function playWebAudioBeep(frequency = 440) {
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc  = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(frequency, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.0, audioCtx.currentTime);
    gain.gain.linearRampToValueAtTime(0.4, audioCtx.currentTime + 0.05);
    gain.gain.linearRampToValueAtTime(0.0, audioCtx.currentTime + 0.5);
    osc.start(audioCtx.currentTime);
    osc.stop(audioCtx.currentTime + 0.5);
  } catch (e) { /* audio not supported */ }
}

// ── Controls ─────────────────────────────────────────────────────────────
async function resetAgent() {
  try {
    await fetch('/reset_agent', { method: 'POST' });
    lastState = null;
    clearInterval(alertInterval);
    alertActive = false;
    document.getElementById('alertBanner').style.display = 'none';
    document.getElementById('logList').innerHTML = '';
    log('Agent state reset.', 'info');
  } catch (e) {}
}

function toggleAudio() {
  audioEnabled = !audioEnabled;
  const btn = document.getElementById('audioBtn');
  btn.textContent = audioEnabled ? '🔔 Audio ON' : '🔕 Audio OFF';
  if (!audioEnabled) { clearInterval(alertInterval); alertActive = false; }
}

function handleVideoError() {
  document.getElementById('overlayState').textContent = 'CAMERA ERROR';
  log('Camera feed unavailable.', 'danger');
}

// ── Event Log ─────────────────────────────────────────────────────────────
function log(message, type = 'info') {
  const list = document.getElementById('logList');
  const ts   = new Date().toLocaleTimeString();
  const li   = document.createElement('li');
  li.className = `log-item log-${type}`;
  li.textContent = `[${ts}] ${message}`;
  list.insertBefore(li, list.firstChild);
  // Trim old entries
  while (list.children.length > LOG_MAX) list.removeChild(list.lastChild);
}

// ── Helpers ───────────────────────────────────────────────────────────────
function formatDuration(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}
