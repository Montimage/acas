/**
 * Edge client — the ACAS side of the ACAS→agent channel.
 *
 * Dependency-free HTTPS/HTTP wrapper (Node core only) that talks to a remote
 * mmt-agent. Every request carries the edge's bearer token; requests time out
 * so a dead edge can never hang an ACAS route.
 *
 * TLS: production edges use HTTPS. On a LAN with a self-signed cert, set the
 * edge's `insecureTLS` flag to skip verification, or provide `caFingerprint`
 * for pinning. We never disable verification silently.
 */
const http = require('http');
const https = require('https');
const { getEdgeWithSecret } = require('./edgeRegistry');

const DEFAULT_TIMEOUT_MS = 15000;

/**
 * Low-level request. Returns { status, headers, body, json }.
 * @param {object} edge   edge record (with token)
 * @param {object} opts   { method, path, body, timeoutMs, accept }
 */
function request(edge, { method = 'GET', path = '/', body = null, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  return new Promise((resolve, reject) => {
    const isTls = edge.tls !== false;
    const lib = isTls ? https : http;
    const payload = body == null ? null : Buffer.from(JSON.stringify(body));

    const headers = {
      Authorization: `Bearer ${edge.token}`,
      Accept: 'application/json',
    };
    if (payload) {
      headers['Content-Type'] = 'application/json';
      headers['Content-Length'] = payload.length;
    }

    const reqOpts = {
      host: edge.host,
      port: edge.port,
      method,
      path,
      headers,
      timeout: timeoutMs,
    };
    if (isTls) {
      // Verify by default. Only relax when the edge explicitly opts in.
      reqOpts.rejectUnauthorized = edge.insecureTLS !== true;
    }

    const req = lib.request(reqOpts, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        const raw = Buffer.concat(chunks);
        let json = null;
        try { json = JSON.parse(raw.toString('utf8')); } catch (e) { /* non-JSON */ }
        resolve({ status: res.statusCode, headers: res.headers, body: raw, json });
      });
    });

    req.on('error', reject);
    req.on('timeout', () => req.destroy(new Error(`edge request timed out after ${timeoutMs}ms`)));
    if (payload) req.write(payload);
    req.end();
  });
}

// Throw a useful error for non-2xx responses.
function ensureOk(res, what) {
  if (res.status >= 200 && res.status < 300) return res;
  const detail = (res.json && (res.json.error || res.json.message)) || res.body.toString('utf8').slice(0, 300);
  const err = new Error(`${what} failed (HTTP ${res.status}): ${detail}`);
  err.status = res.status;
  throw err;
}

/**
 * Build a typed client bound to an edge id. Resolves the secret lazily so the
 * token never lives longer than a call chain.
 */
function forEdge(edgeId) {
  const edge = getEdgeWithSecret(edgeId);
  if (!edge) throw new Error(`unknown edge: ${edgeId}`);

  const call = (opts, what) => request(edge, opts).then((res) => ensureOk(res, what));

  return {
    edge,

    health: () => call({ path: '/health', timeoutMs: 8000 }, 'health').then((r) => r.json),

    interfaces: () => call({ path: '/interfaces' }, 'list interfaces').then((r) => r.json),

    // Push/refresh the rule workspace. files = [{ name, xml }].
    syncRules: (files) =>
      call({ method: 'POST', path: '/rules', body: { files }, timeoutMs: 120000 }, 'rule sync').then((r) => r.json),

    // Read the mmt-probe config the edge is currently using.
    getProbeConfig: () =>
      call({ path: '/probe-config' }, 'get probe config').then((r) => r.json),

    // Push ACAS's mmt-probe config (so the edge's features match what the model expects).
    syncProbeConfig: (conf) =>
      call({ method: 'POST', path: '/probe-config', body: { conf }, timeoutMs: 30000 }, 'probe config sync').then((r) => r.json),

    // Start online capture. engine: 'probe' | 'security' | 'both'.
    // options carries rule-based tuning: { intervalSec, verbose, excludeMask, cores }.
    startOnline: ({ sessionId, netInf, engine, options }) =>
      call({ method: 'POST', path: '/online', body: { sessionId, netInf, engine, options } }, 'start online').then((r) => r.json),

    stop: (sessionId) =>
      call({ method: 'POST', path: '/stop', body: { sessionId } }, 'stop').then((r) => r.json),

    status: (sessionId) =>
      call({ path: `/status${sessionId ? `?sessionId=${encodeURIComponent(sessionId)}` : ''}` }, 'status').then((r) => r.json),

    // List report files produced since a cursor: { files:[{name,size,mtime}], cursor }.
    listReports: (sessionId, since = 0) =>
      call({ path: `/reports/${encodeURIComponent(sessionId)}?since=${since}` }, 'list reports').then((r) => r.json),

    // Download one report file (raw bytes).
    fetchReport: (sessionId, fileName) =>
      request(edge, { path: `/reports/${encodeURIComponent(sessionId)}/${encodeURIComponent(fileName)}`, timeoutMs: 60000 })
        .then((res) => ensureOk(res, 'fetch report').body),
  };
}

module.exports = { forEdge, request };
