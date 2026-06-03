/**
 * Report puller — ACAS side of the online pull pipeline.
 *
 * For an active remote online session, periodically pulls newly-produced output
 * files from the edge agent and writes them into the SAME local directory layout
 * ACAS uses for local runs (REPORT_PATH/report-<sessionId>/). Once the files are
 * there, every downstream consumer (AI predict, rule-based alert parsing, GUI)
 * works unchanged — local and remote converge on one code path.
 *
 * One poller per (edgeId, sessionId). Pull-only: ACAS initiates every request,
 * so only ACAS→edge connectivity is required.
 */
const fs = require('fs');
const path = require('path');
const { REPORT_PATH } = require('../constants');
const edgeClient = require('./edgeClient');

const DEFAULT_INTERVAL_MS = 3000;

// active pollers keyed by sessionId
const pollers = new Map();

function sessionDir(sessionId) {
  return path.join(REPORT_PATH, `report-${sessionId}`);
}

// Reject anything that isn't a plain file name (defence in depth; the agent is
// also expected to validate, but we never trust a remote-supplied path).
function safeName(name) {
  return typeof name === 'string' && name === path.basename(name) && !name.startsWith('.');
}

async function pollOnce(state) {
  const client = edgeClient.forEdge(state.edgeId);
  const { files = [], cursor } = await client.listReports(state.sessionId, state.cursor);
  for (const f of files) {
    if (!safeName(f.name)) continue;
    const bytes = await client.fetchReport(state.sessionId, f.name);
    const dest = path.join(sessionDir(state.sessionId), f.name);
    // resolved path must stay inside the session dir
    if (!path.resolve(dest).startsWith(path.resolve(sessionDir(state.sessionId)) + path.sep)) continue;
    fs.writeFileSync(dest, bytes);
  }
  if (typeof cursor !== 'undefined' && cursor !== null) state.cursor = cursor;
}

/**
 * Start pulling reports for a remote online session.
 * Idempotent: starting an already-running poller is a no-op.
 */
function start({ edgeId, sessionId, intervalMs = DEFAULT_INTERVAL_MS }) {
  if (pollers.has(sessionId)) return;
  fs.mkdirSync(sessionDir(sessionId), { recursive: true });

  const state = { edgeId, sessionId, cursor: 0, timer: null, busy: false };
  const tick = async () => {
    if (state.busy) return; // never overlap polls
    state.busy = true;
    try {
      await pollOnce(state);
    } catch (e) {
      // Surface but don't crash the poller; the session can recover on the next tick.
      console.warn(`[reportPuller] ${sessionId}: ${e.message}`);
    } finally {
      state.busy = false;
    }
  };
  state.timer = setInterval(tick, intervalMs);
  pollers.set(sessionId, state);
  tick(); // pull immediately so the GUI sees data without waiting a full interval
}

/** Stop pulling for a session and do one final drain. */
async function stop(sessionId) {
  const state = pollers.get(sessionId);
  if (!state) return;
  clearInterval(state.timer);
  pollers.delete(sessionId);
  try { await pollOnce(state); } catch (e) { /* best-effort final drain */ }
}

function isActive(sessionId) {
  return pollers.has(sessionId);
}

module.exports = { start, stop, isActive };
