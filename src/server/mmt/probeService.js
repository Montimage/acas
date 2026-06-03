/**
 * Probe service — single dispatch point for "run mmt on host X".
 *
 *   hostId omitted | 'local'  → the existing local mmt-connector (UNCHANGED)
 *   any other hostId          → a remote edge agent via edgeClient
 *
 * Local behaviour is delegated verbatim to mmt-connector so this feature adds
 * remote support without touching the local path. Remote online additionally
 * starts a report puller so the edge's output lands in the local report dir.
 */
const { getUniqueId } = require('../utils/utils');
const connector = require('./mmt-connector');
const { getEdge } = require('./edgeRegistry');
const edgeClient = require('./edgeClient');
const reportPuller = require('./reportPuller');

const isLocal = (hostId) => !hostId || hostId === 'local';

// Valid Linux interface name; also blocks shell metacharacters before the value
// is ever forwarded to the agent (defence in depth — the agent re-validates).
const VALID_IFACE = /^[A-Za-z0-9_.:-]{1,32}$/;

/** Promisified local online start (mmt-connector uses callbacks). */
function localOnline(netInf) {
  return new Promise((resolve) => connector.startMMTOnline(netInf, resolve));
}
function localStop() {
  return new Promise((resolve) => connector.stopMMT((s) => resolve(s || { isRunning: false })));
}

async function startOnline({ hostId, netInf, engine = 'both', options = {} }) {
  if (!VALID_IFACE.test(String(netInf || ''))) {
    return { error: `Invalid interface name: ${netInf}` };
  }

  if (isLocal(hostId)) {
    // Local path is unchanged; `engine` is implicit (mmt-probe) as today.
    return localOnline(netInf);
  }

  if (!getEdge(hostId)) return { error: `Unknown host: ${hostId}` };

  // ACAS owns the session id so both sides agree and the puller can find files.
  const sessionId = getUniqueId();
  const client = edgeClient.forEdge(hostId);
  const result = await client.startOnline({ sessionId, netInf, engine, options });
  reportPuller.start({ edgeId: hostId, sessionId });
  return { ...result, sessionId, hostId, isRunning: true, isOnlineMode: true, engine };
}

async function stop({ hostId, sessionId }) {
  if (isLocal(hostId)) return localStop();
  if (!getEdge(hostId)) return { error: `Unknown host: ${hostId}` };

  const client = edgeClient.forEdge(hostId);
  const result = await client.stop(sessionId);
  if (sessionId) await reportPuller.stop(sessionId);
  return { ...result, hostId, isRunning: false };
}

async function getStatus({ hostId, sessionId }) {
  if (isLocal(hostId)) return connector.getMMTStatus();
  if (!getEdge(hostId)) return { error: `Unknown host: ${hostId}` };
  return edgeClient.forEdge(hostId).status(sessionId);
}

module.exports = { startOnline, stop, getStatus };
