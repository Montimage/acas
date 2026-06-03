/**
 * Online capture sessions. Each session runs mmt-probe (AI-based features)
 * and/or mmt_security (rule-based alerts) on a NIC, writing output files into a
 * per-session directory that ACAS later pulls.
 *
 * Processes are spawned with argv arrays (no shell) and tracked by handle, so
 * stop is PID-scoped — we never `killall`.
 */
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { SESSIONS_DIR, RULES_DIR, USE_SUDO, resolveBin, SYNCED_PROBE_CONF, DEFAULT_PROBE_CONF } = require('./config');

const VALID_IFACE = /^[A-Za-z0-9_.:-]{1,32}$/;
const VALID_SESSION = /^[A-Za-z0-9_.\-]{1,128}$/;
// Prefer ACAS's synced config (matches the feature format the model expects);
// resolved per start so a freshly-synced config takes effect without a restart.
const probeConf = () => (fs.existsSync(SYNCED_PROBE_CONF) ? SYNCED_PROBE_CONF : DEFAULT_PROBE_CONF);
const SECURITY_BIN = resolveBin('mmt_security');

const sessions = new Map();

function sessionDir(sessionId) {
  return path.join(SESSIONS_DIR, sessionId);
}

// Build the privileged spawn (sudo -n <bin> ... when USE_SUDO, else <bin> ...).
function launch(bin, args, opts) {
  const [cmd, cmdArgs] = USE_SUDO ? ['sudo', ['-n', bin, ...args]] : [bin, args];
  return spawn(cmd, cmdArgs, { stdio: ['ignore', 'pipe', 'pipe'], ...opts });
}

function startEngine(name, bin, args, dir, opts = {}) {
  const logStream = fs.createWriteStream(path.join(dir, `${name}.log`), { flags: 'a' });
  const child = launch(bin, args, opts);
  child.stdout.pipe(logStream);
  child.stderr.pipe(logStream);
  return { name, child, exited: false, code: null };
}

// mmt_security tuning, validated so nothing unsafe reaches the argv array.
const VALID_MASK = /^[0-9,]{1,256}$/; // exclude-rule mask is digits + commas

/**
 * Start an online session.
 * @param {object} p { sessionId, netInf, engine, options }
 *   options (rule-based tuning): { intervalSec, verbose, excludeMask, cores }
 */
function startOnline({ sessionId, netInf, engine = 'both', options = {} }) {
  if (!VALID_SESSION.test(String(sessionId || ''))) throw new Error('invalid sessionId');
  if (!VALID_IFACE.test(String(netInf || ''))) throw new Error('invalid interface name');
  if (sessions.has(sessionId)) throw new Error('session already running');

  const dir = sessionDir(sessionId);
  fs.mkdirSync(dir, { recursive: true });

  const engines = [];
  const want = (e) => engine === e || engine === 'both';

  if (want('probe')) {
    const args = ['-c', probeConf(), '-i', netInf,
      '-X', 'input.mode=ONLINE',
      '-X', `file-output.output-dir=${dir}/`,
      '-X', 'file-output.sample-file=true'];
    engines.push(startEngine('probe', 'mmt-probe', args, dir));
  }

  if (want('security')) {
    // Mirror ACAS's local mmt_security flags: -i, [-v], [-x mask], [-c cores], -f dir/:interval.
    // cwd=RULES_DIR so mmt_security loads ACAS's ./rules (falls back to deb defaults).
    const intervalSec = Number(options.intervalSec) > 0 ? Number(options.intervalSec) : 5;
    const args = ['-i', netInf];
    if (options.verbose) args.push('-v');
    if (options.excludeMask && VALID_MASK.test(String(options.excludeMask))) args.push('-x', String(options.excludeMask));
    if (Number(options.cores) > 0) args.push('-c', String(Number(options.cores)));
    args.push('-f', `${dir}/:${intervalSec}`);
    engines.push(startEngine('security', SECURITY_BIN, args, dir, { cwd: RULES_DIR }));
  }

  if (engines.length === 0) throw new Error(`invalid engine: ${engine}`);

  const session = { sessionId, netInf, engine, startedAt: Date.now(), engines };
  for (const e of engines) {
    e.child.on('exit', (code) => { e.exited = true; e.code = code; });
    e.child.on('error', (err) => { e.exited = true; e.error = err.message; });
  }
  sessions.set(sessionId, session);
  return publicStatus(session);
}

function publicStatus(session) {
  if (!session) return null;
  return {
    sessionId: session.sessionId,
    netInf: session.netInf,
    engine: session.engine,
    startedAt: session.startedAt,
    isRunning: session.engines.some((e) => !e.exited),
    engines: session.engines.map((e) => ({ name: e.name, running: !e.exited, code: e.code, error: e.error })),
  };
}

function status(sessionId) {
  if (sessionId) return publicStatus(sessions.get(sessionId));
  return { sessions: [...sessions.values()].map(publicStatus) };
}

/** PID-scoped stop: SIGTERM the tracked children, escalate to SIGKILL. */
function stop(sessionId) {
  const session = sessions.get(sessionId);
  if (!session) return { stopped: false, reason: 'no such session' };

  for (const e of session.engines) {
    if (e.exited) continue;
    try {
      // When launched via sudo, signal the process group so the child under sudo dies too.
      e.child.kill('SIGTERM');
      setTimeout(() => { try { if (!e.exited) e.child.kill('SIGKILL'); } catch (x) { /* ignore */ } }, 4000);
    } catch (err) { /* ignore */ }
  }
  sessions.delete(sessionId);
  return { stopped: true, sessionId };
}

module.exports = { startOnline, stop, status, sessionDir };
