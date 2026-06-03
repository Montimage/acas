/**
 * mmt-agent — runs MMT on an edge host under ACAS control.
 *
 * Probe-only: it captures and serves output files; it performs no AI/ML. Every
 * route except /health requires the ACAS bearer token. Serves HTTPS when TLS
 * material is configured, otherwise HTTP (loopback/trusted networks only).
 */
const fs = require('fs');
const http = require('http');
const https = require('https');
const express = require('express');

const config = require('./lib/config');
const { requireToken } = require('./lib/auth');
const { listInterfaces } = require('./lib/interfaces');
const { getVersions, captureCapable } = require('./lib/versions');
const rules = require('./lib/rules');
const sessions = require('./lib/sessions');
const reports = require('./lib/reports');

if (!config.TOKEN) {
  console.error('[mmt-agent] FATAL: ACAS_TOKEN is not set. Refusing to start.');
  process.exit(1);
}
for (const d of [config.WORK_DIR, config.SESSIONS_DIR, config.RULES_DIR, config.TMP_DIR]) {
  fs.mkdirSync(d, { recursive: true });
}

const app = express();
app.use(express.json({ limit: '200mb' })); // debs arrive base64-encoded

const wrap = (fn) => (req, res) => Promise.resolve(fn(req, res)).catch((e) => {
  res.status(e.status && e.status >= 400 && e.status < 600 ? e.status : 500)
    .json({ error: e.message || 'agent error' });
});

// Liveness + capabilities (unauthenticated so ACAS can detect a reachable-but-unconfigured agent).
app.get('/health', (req, res) => {
  res.json({ ok: true, versions: getVersions(), captureCapable: captureCapable() });
});

// Everything below requires the token.
app.use(requireToken);

app.get('/interfaces', (req, res) => res.json({ interfaces: listInterfaces() }));

app.post('/rules', wrap((req, res) => {
  const result = rules.syncRules((req.body && req.body.files) || []);
  res.json({ ok: true, ...result });
}));

// Return the mmt-probe config the agent will use (synced if present, else the
// deb default) so ACAS can show what this edge is running.
app.get('/probe-config', wrap((req, res) => {
  const synced = fs.existsSync(config.SYNCED_PROBE_CONF);
  const p = synced ? config.SYNCED_PROBE_CONF : config.DEFAULT_PROBE_CONF;
  const conf = fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : '';
  res.json({ config: conf, source: synced ? 'synced' : 'default', path: p });
}));

// Receive ACAS's mmt-probe config (defines the feature report format the model
// needs). Written to the work dir; sessions.js prefers it over the deb default.
app.post('/probe-config', wrap((req, res) => {
  const conf = req.body && req.body.conf;
  if (typeof conf !== 'string' || !conf) return res.status(400).json({ error: 'missing conf' });
  fs.writeFileSync(config.SYNCED_PROBE_CONF, conf, 'utf8');
  res.json({ ok: true, path: config.SYNCED_PROBE_CONF, bytes: conf.length });
}));

app.post('/online', wrap((req, res) => {
  const { sessionId, netInf, engine, options } = req.body || {};
  res.json({ ok: true, ...sessions.startOnline({ sessionId, netInf, engine, options }) });
}));

app.get('/status', (req, res) => res.json(sessions.status(req.query.sessionId)));

app.post('/stop', (req, res) => {
  const { sessionId } = req.body || {};
  res.json(sessions.stop(sessionId));
});

app.get('/reports/:sessionId', (req, res) => {
  res.json(reports.list(req.params.sessionId, req.query.since || 0));
});

app.get('/reports/:sessionId/:fileName', wrap((req, res) => {
  const bytes = reports.read(req.params.sessionId, req.params.fileName);
  res.type('application/octet-stream').send(bytes);
}));

function start() {
  const useTls = config.TLS_CERT && config.TLS_KEY;
  if (useTls) {
    const server = https.createServer(
      { cert: fs.readFileSync(config.TLS_CERT), key: fs.readFileSync(config.TLS_KEY) },
      app,
    );
    server.listen(config.PORT, () => console.log(`[mmt-agent] HTTPS listening on :${config.PORT}`));
  } else {
    console.warn('[mmt-agent] WARNING: TLS not configured — serving plain HTTP. Use only on a trusted network.');
    http.createServer(app).listen(config.PORT, () => console.log(`[mmt-agent] HTTP listening on :${config.PORT}`));
  }
}

start();
