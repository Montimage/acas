const express = require('express');
const router = express.Router();
const { connect, StringCodec } = require('nats');
const readline = require('readline');
const { spawn, exec, execFile } = require('child_process');
const fs = require('fs');
const path = require('path');

const {
  NATS_URL,
  NATS_USER_INGESTOR,
  NATS_PASS_INGESTOR,
  NATS_SUBJECT,
  USE_SUDO,
} = process.env;

const { LOCAL_NATS_URL, TRAINING_PATH, REPORT_PATH, PCAP_PATH } = require('../constants');
const { identifyUser } = require('../middleware/userAuth');
const ruleManager = require('../utils/ruleManager');
const probeService = require('../mmt/probeService');
// Compile any rules whose .so is missing (idempotent); safe to call on boot.
ruleManager.bootstrap().catch(e => console.error('[ruleManager] bootstrap error:', e.message));
const { resolvePcapPath } = require('../utils/pcapResolver');
const { preparePcapForSecurity } = require('../utils/pcapConverter');
// Output folder for mmt_security CSVs
const SECURITY_OUT_DIR = path.join(__dirname, '../mmt/outputs');
// Import unified session manager
const sessionManager = require('../utils/sessionManager');
// Import queue functions
const { queueRuleBasedDetection, getJobStatus } = require('../queue/job-queue');
const { handleQueueError, isRedisError } = require('../utils/queueErrorHelper');

// Default to sudo unless explicitly disabled; use non-interactive to avoid blocking
const SUDO = USE_SUDO === 'false' ? '' : 'sudo -n ';

async function getNatsConnection(customUrl, customUsername, customPassword) {
  const servers = customUrl || NATS_URL || LOCAL_NATS_URL;
  // Prioritize custom credentials, then fall back to env variables
  const user = customUsername || NATS_USER_INGESTOR || undefined;
  const pass = customPassword || NATS_PASS_INGESTOR || undefined;
  const opts = user && pass
    ? { servers, user, pass }
    : { servers };
  return connect(opts);
}

function normalizeExcludeRules(value) {
  if (value == null) return null;
  const list = Array.isArray(value) ? value : [value];
  const filtered = list
    .map(v => (v == null ? '' : String(v).trim()))
    .filter(Boolean);
  return filtered.length > 0 ? filtered.join(',') : null;
}

function isValidIPv4(ip) {
  if (typeof ip !== 'string') return false;
  const m = ip.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
  if (!m) return false;
  return m.slice(1).every(o => {
    const n = Number(o);
    return n >= 0 && n <= 255 && String(n) === String(Number(o));
  });
}

// ------------------------------
// Rule-based detection (mmt_security)
// ------------------------------

// Attach user identification to enable user-specific PCAP resolution
router.use(identifyUser);

// State for running mmt_security instance (online)
let secState = {
  running: false,
  mode: null, // 'online' | 'offline'
  pid: null,
  child: null,
  iface: null,
  pcapFile: null,
  outputDir: null,
  outputFile: null,
  startedAt: null,
  intervalSec: null,
  ruleVerdicts: [], // [{ rule: 56, verdicts: 8 }]
  sessionId: null,
  viewers: 0,
  startedBy: null,
};

function ensureDir(dir) {
  try {
    fs.mkdirSync(dir, { recursive: true, mode: 0o777 });
    try { fs.chmodSync(dir, 0o777); } catch (e) { }
  } catch (e) { }
}

function resolveSecurityBin() {
  const candidates = [
    '/opt/mmt/security/bin/mmt_security',
    '/opt/mmt/security/bin/mmt-security',
    'mmt_security',
    'mmt-security',
  ];
  for (const p of candidates) {
    try {
      if (p.includes('/') && fs.existsSync(p)) return p;
    } catch (e) { }
  }
  // Fallback to first candidate; spawn will fail and we report error
  return candidates[0];
}

function findLatestSecurityCsv(dir) {
  try {
    if (!dir || !fs.existsSync(dir)) return null;
    const all = fs.readdirSync(dir).filter(f => f.endsWith('.csv'));
    // mmt_security alert files are named "sec-*.csv". Prefer those so a shared
    // dir (e.g. a remote session running engine "both") never returns a
    // mmt-probe "*_data.csv" feature file by mistake. Fall back to any .csv.
    const secFiles = all.filter(f => f.startsWith('sec-'));
    const candidates = secFiles.length > 0 ? secFiles : all;
    const files = candidates
      .map(f => ({ f, full: path.join(dir, f), st: fs.statSync(path.join(dir, f)) }))
      .sort((a, b) => b.st.mtimeMs - a.st.mtimeMs);
    return files.length > 0 ? files[0].full : null;
  } catch (e) {
    return null;
  }
}

// Summarise a security alert CSV as [{ rule, verdicts }], mirroring the
// ruleVerdicts the local online flow derives from mmt_security's stdout. Used to
// give remote (edge) sessions the same response shape as local ones.
function ruleVerdictsFromFile(file) {
  if (!file) return [];
  const counts = new Map();
  for (const a of parseSecurityCsv(file, 1000000)) {
    if (a && a.code != null) counts.set(a.code, (counts.get(a.code) || 0) + 1);
  }
  return [...counts.entries()].map(([rule, verdicts]) => ({ rule, verdicts })).sort((a, b) => a.rule - b.rule);
}

/**
 * Aggregate a session's alerts across ALL its sec-*.csv files.
 *
 * Each session has its own output dir, created fresh at start, so every
 * sec-*.csv in it belongs to this session — the folder IS the session boundary
 * (no timestamp filtering needed). We read every file in chronological order,
 * combine, dedupe, and return up to `limit` (the most recent). A single file is
 * just a one-element aggregate — still valid.
 */
const NO_LIMIT = 10_000_000; // per-file read cap (files are per-interval, far smaller)

function collectSessionAlerts(dir, limit = 500, raw = false) {
  if (!dir || !fs.existsSync(dir)) return { alerts: [], files: [], total: 0 };
  const all = fs.readdirSync(dir).filter(f => f.endsWith('.csv'));
  // Prefer sec-*.csv (alert files); fall back to any .csv for legacy dirs.
  let secFiles = all.filter(f => f.startsWith('sec-'));
  if (secFiles.length === 0) secFiles = all;
  const ordered = secFiles
    .map(f => { const full = path.join(dir, f); return { f, full, mt: fs.statSync(full).mtimeMs }; })
    .sort((a, b) => a.mt - b.mt); // oldest first → chronological

  const combined = [];
  for (const { full } of ordered) combined.push(...parseSecurityCsv(full, NO_LIMIT));
  const files = ordered.map(o => o.f);
  // raw=true → every alert (most-recent `limit`); default → aggregated signatures.
  if (raw) return { alerts: combined.slice(-limit), files, total: combined.length };
  return { alerts: aggregateAlerts(combined), files, total: combined.length };
}

function parseSecurityCsvLine(line) {
  // Example line:
  // 10,0,"",1758916808,56,"detected","attack","Probable SYN flooding attack (Half TCP handshake without TCP RST)",{"event_1":{...}}
  // We'll parse basic CSV fields until JSON blob, then parse JSON.
  if (!line || !line.trim()) return null;
  // Split by commas but preserve JSON at the end; do a naive split by first 8 commas
  const parts = [];
  let current = '';
  let inQuotes = false;
  let commas = 0;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      inQuotes = !inQuotes;
      current += ch;
    } else if (ch === ',' && !inQuotes && commas < 8) {
      parts.push(current);
      current = '';
      commas++;
    } else {
      current += ch;
    }
  }
  parts.push(current);

  if (parts.length < 9) return null;
  const [sev, unknown0, emptyQ, ts, code, status, category, desc, jsonStr] = parts;
  let details = null;
  try { details = JSON.parse(jsonStr); } catch (e) { details = null; }

  // Try to derive src/dst from details
  const attrs = [];
  function scanAttrs(obj) {
    try {
      const events = Object.values(obj || {});
      for (const ev of events) {
        const kv = ev && ev.attributes;
        if (Array.isArray(kv)) {
          for (const [k, v] of kv) {
            attrs.push({ k: String(k), v });
          }
        }
      }
    } catch (e) { }
  }
  if (details) scanAttrs(details);
  const attrMap = new Map(attrs.map(x => [x.k, x.v]));
  const srcIp = attrMap.get('ip.src') || null;
  const dstIp = attrMap.get('ip.dst') || null;
  // A distinguishing "target" for HTTP/app rules (e.g. the probed path or the
  // response code) so dedup keeps distinct targets separate — important on
  // loopback where src/dst IPs are absent.
  const target = attrMap.get('http.uri') || attrMap.get('http.response') || null;

  return {
    probeId: Number(sev),
    timestamp: Number(ts),
    code: Number(code),
    status: (status || '').replace(/"/g, ''),
    category: (category || '').replace(/"/g, ''),
    description: (desc || '').replace(/"/g, ''),
    srcIp,
    dstIp,
    target,
    raw: line,
    details,
  };
}

function parseSecurityCsv(filePath, limit = 500) {
  try {
    if (!filePath || !fs.existsSync(filePath)) return [];
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split(/\r?\n/).filter(Boolean);
    const out = [];
    for (const line of lines.slice(-limit)) {
      const obj = parseSecurityCsvLine(line);
      if (obj) out.push(obj);
    }
    return out;
  } catch (e) {
    return [];
  }
}

function parseRuleVerdictsFromText(text) {
  try {
    const map = new Map();
    String(text || '')
      .split(/\r?\n/)
      .forEach((line) => {
        const m = line.match(/-\s*rule\s+(\d+)\s+generated\s+(\d+)\s+verdicts/i);
        if (m) {
          const rule = Number(m[1]);
          const verdicts = Number(m[2]);
          map.set(rule, verdicts);
        }
      });
    return Array.from(map.entries()).map(([rule, verdicts]) => ({ rule, verdicts }));
  } catch {
    return [];
  }
}

const TARGET_CAP = 20; // max distinct targets listed per aggregated alert

/**
 * Aggregate raw alerts into one entry per threat signature.
 *
 * Key = (code, category, srcIp, dstIp) — "who did what to whom". The target
 * (path / response code / port) is deliberately NOT in the key, because for
 * volumetric attacks (scans, brute force) it would explode into thousands of
 * rows; instead each aggregate carries `count`, `firstSeen`/`lastSeen`, and a
 * capped list of distinct `targets`. This keeps the volume + breadth signal
 * without flooding. Raw events remain on disk for drill-down (`?raw=true`).
 */
function aggregateAlerts(list) {
  try {
    const groups = new Map();
    for (const a of list || []) {
      const code = a && typeof a.code !== 'undefined' ? String(a.code) : '';
      const category = String((a && a.category) || '').trim().toLowerCase();
      const src = String((a && a.srcIp) || '').trim().toLowerCase();
      const dst = String((a && a.dstIp) || '').trim().toLowerCase();
      const key = [code, category, src, dst].join('|');
      let g = groups.get(key);
      if (!g) {
        // Seed from the first alert (keeps a representative raw/details sample).
        g = {
          probeId: a.probeId, code: a.code, status: a.status, category: a.category,
          description: a.description, srcIp: a.srcIp, dstIp: a.dstIp,
          count: 0, firstSeen: a.timestamp, lastSeen: a.timestamp,
          targets: [], distinctTargets: 0, raw: a.raw, details: a.details,
          _seenTargets: new Set(),
        };
        groups.set(key, g);
      }
      g.count += 1;
      if (typeof a.timestamp === 'number') {
        if (g.firstSeen == null || a.timestamp < g.firstSeen) g.firstSeen = a.timestamp;
        if (g.lastSeen == null || a.timestamp > g.lastSeen) g.lastSeen = a.timestamp;
      }
      if (a.target && !g._seenTargets.has(a.target)) {
        g._seenTargets.add(a.target);
        if (g.targets.length < TARGET_CAP) g.targets.push(a.target);
      }
    }
    return [...groups.values()].map((g) => {
      g.distinctTargets = g._seenTargets.size;
      delete g._seenTargets;
      return g;
    });
  } catch (_) {
    return Array.isArray(list) ? list : [];
  }
}

router.get('/rule-based/status', async (req, res) => {
  try {
    const { sessionId } = req.query;
    const isAdmin = req.isAdmin || false;

    if (sessionId) {
      // Get specific session status
      const session = sessionManager.getSession('attacks', sessionId);
      if (!session) {
        return res.status(404).json({ error: 'Session not found', sessionId });
      }
      const lastFile = session.outputFile || findLatestSecurityCsv(session.outputDir);
      if (lastFile && !session.outputFile) {
        sessionManager.updateSession('attacks', sessionId, { outputFile: lastFile });
      }
      return res.send({
        ok: true,
        ...session,
        isAdmin,
        canStartOnline: isAdmin
      });
    }

    // Legacy: return global secState (for backward compatibility)
    const lastFile = secState.outputFile || findLatestSecurityCsv(secState.outputDir);
    if (lastFile) secState.outputFile = lastFile;
    res.send({
      ok: true,
      ...secState,
      isAdmin,
      canStartOnline: isAdmin
    });
  } catch (e) {
    res.status(500).send(e.message || 'Failed to get status');
  }
});

router.get('/rule-based/alerts', async (req, res) => {
  try {
    const { sessionId, limit: limitParam } = req.query;
    const limit = limitParam ? Number(limitParam) : 500;

    let outputDir;
    if (sessionId) {
      // Get alerts for a specific session (local or remote edge)
      const session = sessionManager.getSession('attacks', sessionId);
      if (!session) {
        return res.status(404).json({ error: 'Session not found', sessionId });
      }
      outputDir = session.outputDir;
    } else {
      // Legacy: use global secState
      outputDir = secState.outputDir;
    }

    // Aggregate across all of the session's sec-*.csv files (covers the whole
    // session). ?raw=true returns every alert instead of aggregated signatures.
    const raw = req.query.raw === 'true';
    const { alerts, files, total } = collectSessionAlerts(outputDir, limit, raw);
    res.send({ ok: true, files, count: alerts.length, total, aggregated: !raw, alerts });
  } catch (e) {
    res.status(500).send(e.message || 'Failed to read alerts');
  }
});

// List available rules (catalog for the selection table)
router.get('/rule-based/rules', async (req, res) => {
  try {
    const refresh = req.query.refresh === 'true';
    res.json({ ok: true, rules: await ruleManager.listRules({ refresh }) });
  } catch (e) {
    res.status(500).send(e.message || 'Failed to list rules');
  }
});

// Raw XML source of a single rule
router.get('/rule-based/rules/:id/xml', (req, res) => {
  const xml = ruleManager.getRuleXml(req.params.id);
  if (!xml) return res.status(404).send('No XML source for this rule');
  res.type('application/xml').send(xml);
});

// Add a user rule: { filename, xml }. Compiles and validates before saving.
router.post('/rule-based/rules', async (req, res) => {
  try {
    const { filename, xml } = req.body || {};
    if (!filename || !xml) return res.status(400).send('Missing filename or xml');
    const r = await ruleManager.addRule(filename, xml);
    if (!r.ok) return res.status(400).send(r.error);
    res.json({ ok: true, id: r.id, rules: await ruleManager.listRules({ refresh: true }) });
  } catch (e) {
    res.status(500).send(e.message || 'Failed to add rule');
  }
});

// Edit a user-added rule's XML. Predefined rules are protected by ruleManager.
router.put('/rule-based/rules/:id', async (req, res) => {
  try {
    const { xml } = req.body || {};
    if (!xml) return res.status(400).send('Missing xml');
    const r = await ruleManager.updateRule(req.params.id, xml);
    if (!r.ok) return res.status(400).send(r.error);
    res.json({ ok: true, id: r.id, rules: await ruleManager.listRules({ refresh: true }) });
  } catch (e) {
    res.status(500).send(e.message || 'Failed to update rule');
  }
});

// Remove a user-added rule. Predefined rules are protected by ruleManager.
router.delete('/rule-based/rules/:id', async (req, res) => {
  try {
    const r = ruleManager.deleteRule(req.params.id);
    if (!r.ok) return res.status(400).send(r.error);
    res.json({ ok: true, rules: await ruleManager.listRules({ refresh: true }) });
  } catch (e) {
    res.status(500).send(e.message || 'Failed to remove rule');
  }
});

router.post('/rule-based/online/start', async (req, res) => {
  try {
    const { interface: ifaceParam, iface: ifaceLegacy, intervalSec = 5, verbose = true, excludeRules, cores, hostId } = req.body || {};
    const iface = ifaceParam || ifaceLegacy; // accept "interface" (consistent with /predict/online); "iface" kept for compatibility
    if (!iface) return res.status(400).send('Missing interface');

    // Validate every value that reaches the mmt_security command line, to prevent
    // OS command injection (these flow into args / the spawned process).
    if (!/^[A-Za-z0-9_.:-]{1,32}$/.test(String(iface))) {
      return res.status(400).send('Invalid interface name');
    }
    if (cores != null && cores !== '' && !(Number.isInteger(Number(cores)) && Number(cores) > 0 && Number(cores) <= 1024)) {
      return res.status(400).send('Invalid cores (must be a positive integer)');
    }
    if (intervalSec != null && intervalSec !== '' && !(Number(intervalSec) > 0 && Number(intervalSec) <= 86400)) {
      return res.status(400).send('Invalid intervalSec (must be a positive number)');
    }
    const excludeMaskChecked = normalizeExcludeRules(excludeRules);
    if (excludeMaskChecked && !/^[0-9,]{1,256}$/.test(excludeMaskChecked)) {
      return res.status(400).send('Invalid excludeRules (rule ids only)');
    }

    // Remote edge: capture runs on the agent (engine "security"); ACAS pulls the
    // sec-*.csv alerts into report-<sessionId>/. Register the session so the
    // existing /rule-based/alerts and /stop paths resolve it.
    if (hostId && hostId !== 'local') {
      // Forward the same rule-based tuning the local path uses, so edge sessions
      // honour intervalSec/verbose/excludeRules/cores too.
      const excludeMask = normalizeExcludeRules(excludeRules);
      const r = await probeService.startOnline({
        hostId,
        netInf: iface,
        engine: 'security',
        options: { intervalSec, verbose, excludeMask, cores },
      });
      if (r.error) return res.status(400).json({ error: r.error });
      const outputDir = path.join(REPORT_PATH, `report-${r.sessionId}`);
      const startedAt = new Date().toISOString();
      sessionManager.createSession('attacks', r.sessionId, 'online', {
        outputDir, remote: true, hostId, iface, startedAt, intervalSec,
      });
      // Same shape as a local start so the GUI can treat both uniformly.
      return res.json({
        ok: true, running: true, mode: 'online', remote: true,
        pid: null, child: null, iface, pcapFile: null,
        outputDir, outputFile: null, startedAt, intervalSec,
        ruleVerdicts: [], hostId, sessionId: r.sessionId,
        message: `Remote rule-based detection started on edge ${hostId} (${iface})`,
      });
    }

    // Check if online rule-based detection is already running (local)
    if (secState.running && secState.mode === 'online') {
      return res.status(409).json({
        error: 'Online rule-based detection already running',
        message: 'Rule-based detection is already running. Please stop the current session first.',
        currentSession: secState.sessionId,
        sessionId: secState.sessionId
      });
    }

    // If sudo is enabled, verify non-interactive sudo works; otherwise return guidance
    if (SUDO && SUDO.trim().length > 0) {
      try {
        await new Promise((resolve, reject) => {
          exec(`${SUDO.trim()} -v`, (err) => {
            if (err) reject(err); else resolve();
          });
        });
      } catch (e) {
        return res.status(403).send(
          'sudo is configured but requires a password. Either set USE_SUDO=false and grant capabilities to mmt_security (setcap cap_net_raw,cap_net_admin+eip), or configure sudoers NOPASSWD for the binary.'
        );
      }
    }

    const outDirBase = SECURITY_OUT_DIR;
    // Use a unique folder per run to avoid reading previous alerts
    const runDir = path.join(outDirBase, `run-${Date.now()}`);
    ensureDir(runDir);
    const bin = resolveSecurityBin();

    const args = [];
    args.push('-i', iface);
    if (verbose) args.push('-v');
    const excludeMask = normalizeExcludeRules(excludeRules);
    if (excludeMask) args.push('-x', excludeMask);
    if (cores) args.push('-c', String(cores));
    // ensure trailing slash and include rotation interval so files are created under the run folder
    args.push('-f', `${runDir}/:${Number(intervalSec)}`);

    // Spawn with an argv array (NO shell) so user-supplied values (iface, cores,
    // excludeRules) can never be interpreted as shell syntax — prevents OS command
    // injection. sudo -n stays non-interactive without a shell.
    const useSudo = SUDO.trim().length > 0;
    const spawnCmd = useSudo ? 'sudo' : bin;
    const spawnArgs = useSudo ? ['-n', bin, ...args] : args;
    console.log('[SECURITY][rule-based][online] Executing:', spawnCmd, spawnArgs.join(' '));
    const child = spawn(spawnCmd, spawnArgs, { stdio: ['ignore', 'pipe', 'pipe'], cwd: ruleManager.WORKSPACE });
    const verdictMap = new Map();
    let combinedLog = '';
    const parseAndUpdate = (txt) => {
      const lines = String(txt || '').split(/\r?\n/);
      for (const line of lines) {
        const m = line.match(/-\s*rule\s+(\d+)\s+generated\s+(\d+)\s+verdicts/i);
        if (m) {
          verdictMap.set(Number(m[1]), Number(m[2]));
        }
      }
      secState.ruleVerdicts = Array.from(verdictMap.entries()).map(([rule, verdicts]) => ({ rule, verdicts }));
    };
    child.stdout.on('data', d => {
      const raw = d.toString();
      combinedLog += raw;
      const txt = raw.trim();
      if (txt) console.log('[mmt_security][stdout]', txt);
      parseAndUpdate(raw);
    });
    child.stderr.on('data', d => {
      const raw = d.toString();
      combinedLog += raw;
      const txt = raw.trim();
      if (txt) console.log('[mmt_security][stderr]', txt);
      parseAndUpdate(raw);
    });
    child.on('exit', (code, signal) => {
      console.log(`[SECURITY][rule-based] mmt_security exited code=${code} signal=${signal}`);
      secState.running = false;
      secState.pid = null;
      secState.child = null;
      // finalize ruleVerdicts from combined logs on exit
      const finalVerdicts = parseRuleVerdictsFromText(combinedLog);
      secState.ruleVerdicts = finalVerdicts && finalVerdicts.length > 0
        ? finalVerdicts
        : Array.from(verdictMap.entries()).map(([rule, verdicts]) => ({ rule, verdicts }));
    });

    // Generate unique session ID
    const sessionId = `security_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    secState = {
      running: true,
      mode: 'online',
      pid: child.pid,
      child,
      iface,
      pcapFile: null,
      outputDir: runDir,
      outputFile: null,
      startedAt: new Date().toISOString(),
      intervalSec: Number(intervalSec),
      ruleVerdicts: [],
      sessionId,
      startedBy: req.ip || 'unknown',
    };

    // Create session in session manager
    sessionManager.createSession('attacks', sessionId, 'online', {
      pid: child.pid,
      iface,
      pcapFile: null,
      outputDir: runDir,
      outputFile: null,
      intervalSec: Number(intervalSec),
      ruleVerdicts: [],
      startedBy: req.ip || 'unknown',
    });

    res.send({
      ok: true,
      sessionId,
      message: 'Online rule-based detection started',
      running: true,
      mode: 'online',
      iface,
      pid: child.pid,
      outputDir: runDir,
      startedAt: secState.startedAt,
      intervalSec: secState.intervalSec,
      ruleVerdicts: secState.ruleVerdicts
    });
  } catch (e) {
    console.error('rule-based online start error:', e);
    res.status(500).send(e.message || 'Failed to start mmt_security');
  }
});

router.post('/rule-based/online/stop', async (req, res) => {
  try {
    const { hostId, sessionId } = req.body || {};

    // Remote edge session: stop the agent's capture and the report puller, then
    // return the same shape as a local stop (the puller's final drain has run, so
    // the latest pulled alerts are available for outputFile/ruleVerdicts).
    if (hostId && hostId !== 'local') {
      const r = await probeService.stop({ hostId, sessionId });
      if (r.error) return res.status(400).json({ error: r.error });
      const session = sessionId ? sessionManager.getSession('attacks', sessionId) : null;
      const outputDir = session ? session.outputDir : null;
      const outputFile = findLatestSecurityCsv(outputDir);
      if (sessionId) sessionManager.updateSession('attacks', sessionId, { isRunning: false, outputFile });
      return res.json({
        ok: true, stopped: true, running: false, mode: 'online', remote: true,
        pid: null, child: null,
        iface: session ? session.iface : null,
        pcapFile: null,
        outputDir,
        outputFile,
        startedAt: session ? session.startedAt : null,
        intervalSec: session ? session.intervalSec : null,
        ruleVerdicts: ruleVerdictsFromFile(outputFile),
        hostId, sessionId,
      });
    }

    if (!secState.running) return res.send({ ok: true, stopped: false });

    const stoppedPid = secState.pid;
    const child = secState.child;
    try {
      process.kill(stoppedPid, 'SIGINT');
    } catch (e) {
      console.warn('[SECURITY] Failed SIGINT by PID, trying pkill');
      exec(`${SUDO}pkill -f mmt_security || ${SUDO}pkill -f mmt-security`, () => { });
    }
    // Wait briefly for the process to exit so ruleVerdicts can be finalized
    if (child && typeof child.once === 'function') {
      await Promise.race([
        new Promise((resolve) => child.once('exit', resolve)),
        new Promise((resolve) => setTimeout(resolve, 4000)),
      ]);
    } else {
      // Fallback wait
      await new Promise((r) => setTimeout(r, 1000));
    }
    secState.running = false;
    secState.pid = null;
    const lastFile = findLatestSecurityCsv(secState.outputDir);
    if (lastFile) secState.outputFile = lastFile;

    // Update session manager
    if (secState.sessionId) {
      sessionManager.updateSession('attacks', secState.sessionId, {
        isRunning: false,
        outputFile: lastFile,
        pid: null
      });
    }

    res.send({ ok: true, stopped: true, ...secState });
  } catch (e) {
    res.status(500).send(e.message || 'Failed to stop mmt_security');
  }
});

router.post('/rule-based/offline', async (req, res) => {
  try {
    const { pcapFile, filePath, verbose = false, excludeRules, cores, useQueue } = req.body || {};
    // Validate values that reach the mmt_security command line (injection guard).
    if (cores != null && cores !== '' && !(Number.isInteger(Number(cores)) && Number(cores) > 0 && Number(cores) <= 1024)) {
      return res.status(400).send('Invalid cores (must be a positive integer)');
    }
    const offlineMaskChecked = normalizeExcludeRules(excludeRules);
    if (offlineMaskChecked && !/^[0-9,]{1,256}$/.test(offlineMaskChecked)) {
      return res.status(400).send('Invalid excludeRules (rule ids only)');
    }
    const userId = req.userId;
    let inputPath = null;
    if (filePath) {
      inputPath = filePath;
    }
    if (!inputPath && pcapFile) {
      // Use unified resolver: checks user uploads -> samples -> legacy PCAP_PATH
      inputPath = resolvePcapPath(pcapFile, userId);
    }
    if (!inputPath) return res.status(400).send('Missing pcapFile or filePath');
    if (!fs.existsSync(inputPath)) return res.status(404).send(`PCAP not found: ${inputPath}`);

    // Generate unique session ID for this offline detection
    const sessionId = `security_offline_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    // Queue-based approach is ENABLED BY DEFAULT
    const useQueueDefault = process.env.USE_QUEUE_BY_DEFAULT !== 'false';
    const shouldUseQueue = useQueue !== undefined ? useQueue : useQueueDefault;
    let fallbackToSync = false;

    if (shouldUseQueue) {
      console.log('[SECURITY][rule-based][offline] Using queue-based processing for session:', sessionId);

      let result;
      try {
        // Queue the job
        result = await queueRuleBasedDetection({
          pcapFile,
          filePath: inputPath,
          sessionId,
          verbose,
          excludeRules,
          cores,
          priority: 5
        });
      } catch (error) {
        // Check if it's a Redis connection error
        if (isRedisError(error)) {
          console.warn('[SECURITY] Redis unavailable, automatically falling back to sync mode');
          fallbackToSync = true;
          // Fall through to sync processing below
        } else {
          // For non-Redis errors, return the error
          return handleQueueError(res, error, 'Rule-based detection queue');
        }
      }

      if (!fallbackToSync) {
        const jobId = result.jobId;
      console.log('[SECURITY] Job queued:', jobId, 'Waiting for completion...');

      // Create session in session manager
      sessionManager.createSession('attacks', sessionId, 'offline', {
        pcapFile,
        filePath: inputPath,
        outputDir: null, // Will be set by worker
        outputFile: null,
        excludeRules,
        cores,
        queued: true,
        jobId: jobId
      });

      // Poll for job completion
      const maxWaitTime = 5 * 60 * 1000; // 5 minutes max
      const pollInterval = 2000; // 2 seconds
      const startTime = Date.now();

      while (Date.now() - startTime < maxWaitTime) {
        const status = await getJobStatus(jobId, 'rule-based-detection');

        if (status.status === 'completed') {
          console.log('[SECURITY] Job completed:', jobId);

          // Update session with results
          sessionManager.updateSession('attacks', sessionId, {
            isRunning: false,
            outputDir: status.result?.outputDir,
            outputFile: status.result?.outputFile,
            ruleVerdicts: status.result?.ruleVerdicts
          });

          // Parse and return alerts
          const file = status.result?.outputFile;
          const alerts = file ? parseSecurityCsv(file, 2000) : [];
          const uniqueAlerts = aggregateAlerts(alerts);

          return res.send({
            ok: true,
            sessionId,
            file,
            count: uniqueAlerts.length,
            alerts: uniqueAlerts,
            ruleVerdicts: status.result?.ruleVerdicts || [],
            queued: true,
            jobId: jobId
          });
        } else if (status.status === 'failed') {
          console.error('[SECURITY] Job failed:', jobId, status.failedReason);
          sessionManager.updateSession('attacks', sessionId, {
            isRunning: false,
            error: status.failedReason
          });
          return res.status(500).send(status.failedReason || 'Rule-based detection failed');
        }

        // Still processing, wait and check again
        await new Promise(resolve => setTimeout(resolve, pollInterval));
      }

        // Timeout
        return res.status(408).send('Request timeout: Rule-based detection took too long');
      }
    }

    // Direct processing (blocking) - used when useQueue=false OR when Redis is unavailable
    console.log('[SECURITY][rule-based][offline] Using direct processing (blocking) for session:', sessionId);

    const outDir = path.join(SECURITY_OUT_DIR, `offline-${sessionId}`);
    ensureDir(outDir);
    const bin = resolveSecurityBin();

    // Prepare pcap file (convert if needed for LINUX_SLL or other non-Ethernet formats)
    let pcapPrep;
    try {
      pcapPrep = await preparePcapForSecurity(inputPath);
      if (pcapPrep.converted) {
        console.log(`[SECURITY][rule-based][offline] Converted ${pcapPrep.linkType} to Ethernet format`);
      }
    } catch (convError) {
      console.warn('[SECURITY][rule-based][offline] PCAP conversion failed, using original file:', convError.message);
      // Fallback to original file if conversion fails
      pcapPrep = {
        path: inputPath,
        converted: false,
        cleanup: () => {}
      };
    }

    const processPath = pcapPrep.path;

    const args = [];
    args.push('-t', processPath);
    const excludeMask = normalizeExcludeRules(excludeRules);
    if (excludeMask) args.push('-x', excludeMask);
    if (cores) args.push('-c', String(cores));
    args.push('-f', `${outDir}/`);

    console.log('[SECURITY][rule-based][offline] Executing:', bin, args.join(' '));

    // Create session in session manager
    sessionManager.createSession('attacks', sessionId, 'offline', {
      pcapFile,
      filePath: inputPath,
      processedPath: processPath,
      converted: pcapPrep.converted,
      outputDir: outDir,
      outputFile: null,
      excludeRules,
      cores,
    });

    // execFile (NO shell): args passed as a discrete argv array, so excludeRules/
    // cores/pcap path can't be interpreted as shell syntax — prevents injection.
    execFile(bin, args, { cwd: ruleManager.WORKSPACE, maxBuffer: 200 * 1024 * 1024 }, (error, stdout, stderr) => {
      // Always cleanup temporary converted file
      pcapPrep.cleanup();

      if (error) {
        console.error('mmt_security offline error:', stderr || error.message);
        sessionManager.updateSession('attacks', sessionId, {
          isRunning: false,
          error: stderr || error.message
        });
        return res.status(500).send(stderr || error.message);
      }
      const file = findLatestSecurityCsv(outDir);
      const alerts = parseSecurityCsv(file, 2000);
      const uniqueAlerts = aggregateAlerts(alerts);
      const ruleVerdicts = parseRuleVerdictsFromText(`${stdout}\n${stderr}`);

      sessionManager.updateSession('attacks', sessionId, {
        isRunning: false,
        outputFile: file,
        ruleVerdicts,
        alertCount: uniqueAlerts.length
      });

      const response = {
        ok: true,
        sessionId,
        file,
        count: uniqueAlerts.length,
        alerts: uniqueAlerts,
        ruleVerdicts,
        converted: pcapPrep.converted
      };

      if (fallbackToSync) {
        response.warning = 'Redis/Valkey service is unavailable. Automatically switched to synchronous processing mode.';
        response.message = 'Rule-based detection completed in sync mode (Redis unavailable, automatic fallback)';
      }

      if (pcapPrep.converted) {
        response.conversionInfo = `PCAP was automatically converted from ${pcapPrep.linkType} to Ethernet format`;
      }

      res.send(response);
    });
  } catch (e) {
    res.status(500).send(e.message || 'Failed to run offline rule-based detection');
  }
});

// Publish multiple flow records to NATS in bulk
// Body: { payloads: array, subject?: string, natsUrl?: string, username?: string, password?: string }
router.post('/nats-publish/bulk', async (req, res) => {
  try {
    const { payloads, subject, natsUrl, username, password } = req.body || {};
    if (!Array.isArray(payloads) || payloads.length === 0) {
      return res.status(400).send('Missing or empty payloads array');
    }

    const nc = await getNatsConnection(natsUrl, username, password);
    const sc = StringCodec();
    const subj = subject || (process.env.NATS_SUBJECT && process.env.NATS_SUBJECT.trim());
    if (!subj) {
      await nc.close();
      return res.status(400).send('Missing NATS subject (set NATS_SUBJECT env or pass subject)');
    }

    let published = 0;
    let failed = 0;

    for (const payload of payloads) {
      try {
        const data = JSON.stringify({ type: 'flow', payload });
        await nc.publish(subj, sc.encode(data));
        published += 1;
      } catch (e) {
        console.error('Failed to publish flow:', e);
        failed += 1;
      }
    }

    await nc.flush();
    await nc.close();

    res.send({ ok: true, published, failed, subject: subj });
  } catch (e) {
    console.error('NATS bulk publish error:', e);
    res.status(500).send(e.message || 'NATS bulk publish failed');
  }
});

// Stream extracted flows (features CSV) to NATS in chunks
// Body: { sessionId?: string, reportId?: string, fileName: string, chunkLines?: number, subject?: string, natsUrl?: string, username?: string, password?: string }
router.post('/nats-publish/flows', async (req, res) => {
  try {
    const { sessionId, reportId, fileName, chunkLines = 1000, subject, natsUrl, username, password } = req.body || {};
    if (!fileName) return res.status(400).send('Missing fileName');
    const baseDir = reportId ? path.join(REPORT_PATH, reportId) : (sessionId ? path.join(REPORT_PATH, `report-${sessionId}`) : null);
    if (!baseDir) return res.status(400).send('Missing sessionId or reportId');
    let filePath = path.join(baseDir, fileName);
    if (!fs.existsSync(filePath)) {
      console.warn(`[NATS flows] File not found at expected path: ${filePath}. Attempting fallback search under ${REPORT_PATH}`);
      // Fallback: recursive search for fileName under REPORT_PATH
      const stack = [REPORT_PATH];
      let foundPath = null;
      while (stack.length) {
        const dir = stack.pop();
        let entries = [];
        try {
          entries = fs.readdirSync(dir, { withFileTypes: true });
        } catch (e) {
          continue;
        }
        for (const ent of entries) {
          const p = path.join(dir, ent.name);
          if (ent.isDirectory()) {
            // Only descend a few levels to avoid huge scans
            if ((p.match(/\//g) || []).length - (REPORT_PATH.match(/\//g) || []).length <= 6) {
              stack.push(p);
            }
          } else if (ent.isFile() && ent.name === fileName) {
            foundPath = p;
            break;
          }
        }
        if (foundPath) break;
      }
      if (!foundPath) {
        return res.status(404).send(`Flows CSV file not found. Tried: ${path.join(baseDir, fileName)} and no fallback match under ${REPORT_PATH}`);
      }
      console.log(`[NATS flows] Using fallback match for flows file: ${foundPath}`);
      filePath = foundPath;
    }

    const nc = await getNatsConnection(natsUrl, username, password);
    const sc = StringCodec();
    const subj = subject || (process.env.NATS_SUBJECT && process.env.NATS_SUBJECT.trim());
    if (!subj) {
      await nc.close();
      return res.status(400).send('Missing NATS subject (set NATS_SUBJECT env or pass subject)');
    }

    let published = 0;
    let headerLine = null;
    let buffer = [];
    let seq = 0;
    let headerSent = false;

    const publishChunk = async (lines) => {
      if (!lines || lines.length === 0) return;
      const includeHeader = !headerSent && !!headerLine;
      const columns = includeHeader ? headerLine.split(',').map(s => String(s).trim()) : undefined;
      const payload = {
        type: 'flows',
        reportId: reportId || (sessionId ? `report-${sessionId}` : undefined),
        sessionId,
        fileName,
        seq,
        hasHeader: includeHeader,
        header: includeHeader ? headerLine : undefined,
        columns,
        lines,
      };
      const data = JSON.stringify(payload);
      await nc.publish(subj, sc.encode(data));
      published += 1;
      seq += 1;
      if (includeHeader) headerSent = true;
    };

    const rl = readline.createInterface({ input: fs.createReadStream(filePath), crlfDelay: Infinity });
    let lineIndex = 0;
    for await (const line of rl) {
      if (lineIndex === 0) {
        headerLine = line;
      } else {
        buffer.push(line);
        if (buffer.length >= Number(chunkLines)) {
          await publishChunk(buffer);
          buffer = [];
        }
      }
      lineIndex += 1;
    }
    if (buffer.length > 0) {
      await publishChunk(buffer);
    }
    await nc.flush();
    await nc.close();
    res.send({ ok: true, published, chunks: published, fileName });
  } catch (e) {
    console.error('NATS flows publish error:', e);
    res.status(500).send(e.message || 'NATS flows publish failed');
  }
});

// Stream a model dataset to NATS in chunks to avoid large HTTP payloads
// Body: { modelId: string, datasetType: 'train'|'test', chunkLines?: number, subject?: string, natsUrl?: string, username?: string, password?: string }
router.post('/nats-publish/dataset', async (req, res) => {
  try {
    const { modelId, datasetType = 'train', chunkLines = 500, subject, natsUrl, username, password } = req.body || {};
    if (!modelId || !datasetType) return res.status(400).send('Missing modelId or datasetType');

    const datasetName = `${String(datasetType).charAt(0).toUpperCase() + String(datasetType).slice(1)}_samples.csv`;
    const datasetFilePath = path.join(TRAINING_PATH, modelId.replace('.h5', ''), 'datasets', datasetName);
    if (!fs.existsSync(datasetFilePath)) {
      return res.status(404).send(`Dataset file not found: ${datasetFilePath}`);
    }

    const nc = await getNatsConnection(natsUrl, username, password);
    const sc = StringCodec();
    const subj = subject || (process.env.NATS_SUBJECT && process.env.NATS_SUBJECT.trim());
    if (!subj) {
      await nc.close();
      return res.status(400).send('Missing NATS subject (set NATS_SUBJECT env or pass subject)');
    }

    let published = 0;
    let headerLine = null;
    let buffer = [];
    let seq = 0;
    let headerSent = false;

    const publishChunk = async (lines) => {
      if (!lines || lines.length === 0) return;
      const includeHeader = !headerSent && !!headerLine;
      const columns = includeHeader ? headerLine.split(',').map(s => String(s).trim()) : undefined;
      const payload = {
        type: 'dataset',
        modelId,
        datasetType,
        seq,
        hasHeader: includeHeader,
        header: includeHeader ? headerLine : undefined,
        columns,
        lines,
      };
      const data = JSON.stringify(payload);
      await nc.publish(subj, sc.encode(data));
      published += 1;
      seq += 1;
      if (includeHeader) headerSent = true;
    };

    const rl = readline.createInterface({ input: fs.createReadStream(datasetFilePath), crlfDelay: Infinity });
    let lineIndex = 0;
    for await (const line of rl) {
      if (lineIndex === 0) {
        headerLine = line;
      } else {
        buffer.push(line);
        if (buffer.length >= Number(chunkLines)) {
          await publishChunk(buffer);
          buffer = [];
        }
      }
      lineIndex += 1;
    }
    if (buffer.length > 0) {
      await publishChunk(buffer);
    }
    await nc.flush();
    await nc.close();
    res.send({ ok: true, published, chunks: published, modelId, datasetType });
  } catch (e) {
    console.error('NATS dataset publish error:', e);
    res.status(500).send(e.message || 'NATS dataset publish failed');
  }
});

router.post('/block-ip-port', async (req, res) => {
  try {
    const { ip, port, protocol = 'tcp' } = req.body || {};
    const p = Number(port);
    if (!isValidIPv4(ip)) return res.status(400).send('Invalid IPv4 address');
    if (!p || p < 1 || p > 65535) return res.status(400).send('Invalid port');
    const proto = ['tcp', 'udp'].includes(String(protocol).toLowerCase()) ? String(protocol).toLowerCase() : 'tcp';
    const qip = ip.replace(/"/g, '');
    const cmd = `${SUDO}iptables -I INPUT -s ${qip} -p ${proto} --dport ${p} -j DROP && ${SUDO}iptables -I OUTPUT -d ${qip} -p ${proto} --sport ${p} -j DROP`;
    console.log('[SECURITY] Executing:', cmd);
    exec(cmd, (error, stdout, stderr) => {
      if (error) {
        console.error('iptables error:', stderr || error.message);
        return res.status(500).send(stderr || error.message);
      }
      res.send({ ok: true, command: cmd, stdout });
    });
  } catch (e) {
    console.error('block-ip-port error:', e);
    res.status(500).send(e.message || 'block-ip-port failed');
  }
});

router.post('/drop-session', async (req, res) => {
  try {
    const { ip } = req.body || {};
    if (!isValidIPv4(ip)) return res.status(400).send('Invalid IPv4 address');
    const cmd = `${SUDO}iptables -I INPUT -s ${ip} -j DROP && ${SUDO}iptables -I OUTPUT -d ${ip} -j DROP`;
    console.log('[SECURITY] Executing:', cmd);
    exec(cmd, (error, stdout, stderr) => {
      if (error) {
        console.error('iptables error:', stderr || error.message);
        return res.status(500).send(stderr || error.message);
      }
      res.send({ ok: true, command: cmd, stdout });
    });
  } catch (e) {
    res.status(500).send(e.message || 'drop-session failed');
  }
});

router.post('/rate-limit', async (req, res) => {
  try {
    const { ip, port, protocol = 'tcp', limit = '5/sec', burst = 10, direction = 'in' } = req.body || {};
    if (!isValidIPv4(ip)) return res.status(400).send('Invalid IPv4 address');
    const p = Number(port);
    if (!p || p < 1 || p > 65535) return res.status(400).send('Invalid port');
    const proto = ['tcp', 'udp'].includes(String(protocol).toLowerCase()) ? String(protocol).toLowerCase() : 'tcp';
    // Validate limit/burst — they reach the iptables command line (run as root).
    if (!/^[0-9]{1,6}\/(sec|second|min|minute|hour)$/i.test(String(limit))) {
      return res.status(400).send('Invalid limit (e.g. "5/sec")');
    }
    if (!(Number.isInteger(Number(burst)) && Number(burst) > 0 && Number(burst) <= 100000)) {
      return res.status(400).send('Invalid burst (positive integer)');
    }
    if (!['in', 'out', 'both'].includes(String(direction))) {
      return res.status(400).send('Invalid direction (in|out|both)');
    }

    const rules = [];
    if (direction === 'in' || direction === 'both') {
      rules.push(`${SUDO}iptables -I INPUT -s ${ip} -p ${proto} --dport ${p} -m limit --limit ${limit} --limit-burst ${burst} -j ACCEPT`);
      rules.push(`${SUDO}iptables -I INPUT -s ${ip} -p ${proto} --dport ${p} -j DROP`);
    }
    if (direction === 'out' || direction === 'both') {
      rules.push(`${SUDO}iptables -I OUTPUT -d ${ip} -p ${proto} --sport ${p} -m limit --limit ${limit} --limit-burst ${burst} -j ACCEPT`);
      rules.push(`${SUDO}iptables -I OUTPUT -d ${ip} -p ${proto} --sport ${p} -j DROP`);
    }

    const cmd = rules.join(' && ');
    console.log('[SECURITY] Executing rate-limit:', cmd);
    exec(cmd, (error, stdout, stderr) => {
      if (error) {
        console.error('iptables error:', stderr || error.message);
        return res.status(500).send(stderr || error.message);
      }
      res.send({ ok: true, command: cmd, stdout });
    });
  } catch (e) {
    res.status(500).send(e.message || 'rate-limit failed');
  }
});

module.exports = router;