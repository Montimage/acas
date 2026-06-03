/**
 * Edge registry — the set of remote machines that run an mmt-agent.
 *
 * Backed by a small JSON file (edges.json, gitignored) following the same
 * pattern as the rule manager's user-rules.json. No database is needed: an
 * edge is just { id, name, host, port, tls, token } plus cached runtime info.
 *
 * Security: the bearer token is a secret. It is stored here but NEVER returned
 * by the listing helpers — only `getEdgeWithSecret()` (used internally by the
 * edge client) exposes it.
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// Anchor to the code location (this file lives in src/server/mmt/), matching
// ruleManager's workspace pattern. This is correct both in and out of Docker,
// unlike the DOCKER_ENV-aware MMT_PATH which hard-codes the container path.
const REGISTRY_FILE = path.join(__dirname, 'edges.json');

// Keep only recognised SSH fields; coerce types. Returns null if no user given.
function normalizeSsh(ssh) {
  if (!ssh || typeof ssh !== 'object' || !ssh.user) return null;
  return {
    user: String(ssh.user),
    port: Number(ssh.port) || 22,
    privateKey: ssh.privateKey ? String(ssh.privateKey) : undefined,
    keyPath: ssh.keyPath ? String(ssh.keyPath) : undefined,
    passphrase: ssh.passphrase ? String(ssh.passphrase) : undefined,
    password: ssh.password ? String(ssh.password) : undefined,
    sudo: Boolean(ssh.sudo),
  };
}

function readAll() {
  try {
    const raw = fs.readFileSync(REGISTRY_FILE, 'utf8');
    const j = JSON.parse(raw);
    return Array.isArray(j) ? j : [];
  } catch (e) {
    return [];
  }
}

function writeAll(edges) {
  fs.mkdirSync(path.dirname(REGISTRY_FILE), { recursive: true });
  fs.writeFileSync(REGISTRY_FILE, JSON.stringify(edges, null, 2), 'utf8');
}

// Strip secrets before returning an edge to any API caller (token + SSH creds).
function publicView(edge) {
  if (!edge) return edge;
  const { token, ssh, ...rest } = edge;
  const sshPublic = ssh
    ? { user: ssh.user, port: ssh.port, sudo: Boolean(ssh.sudo), hasKey: Boolean(ssh.privateKey || ssh.keyPath), hasPassword: Boolean(ssh.password) }
    : null;
  return { ...rest, hasToken: Boolean(token), ssh: sshPublic };
}

/** List all edges (without secrets). */
function listEdges() {
  return readAll().map(publicView);
}

/** Get one edge (without secret), or null. */
function getEdge(id) {
  const e = readAll().find((x) => x.id === id);
  return e ? publicView(e) : null;
}

/** Internal: get one edge INCLUDING its token (used by the edge client). */
function getEdgeWithSecret(id) {
  return readAll().find((x) => x.id === id) || null;
}

/**
 * Add an edge. Generates the id and a strong bearer token.
 * Returns { edge: <public view>, token }. The token is stored internally and
 * injected into the agent during SSH provisioning; the route does not expose it.
 */
function addEdge({ name, host, port = 8443, tls = true, insecureTLS = false, ssh = null }) {
  if (!host || typeof host !== 'string') {
    throw new Error('host is required');
  }
  const edges = readAll();
  const token = crypto.randomBytes(32).toString('hex');
  const edge = {
    id: crypto.randomUUID(),
    name: name || host,
    host,
    port: Number(port) || 8443,
    tls: Boolean(tls),
    // Skip TLS certificate verification for this edge (e.g. a self-signed cert
    // on a trusted LAN). The provisioned agent uses a self-signed cert, so this
    // defaults to true when SSH provisioning is used.
    insecureTLS: Boolean(insecureTLS),
    // SSH credentials for automatic provisioning (operator-owned; edges.json is
    // gitignored). { user, port, privateKey | keyPath, password, sudo }.
    ssh: normalizeSsh(ssh),
    token,
    createdAt: Date.now(),
    versions: null, // filled in after a /health call
    provisionedAt: null,
  };
  edges.push(edge);
  writeAll(edges);
  return { edge: publicView(edge), token };
}

/** Patch mutable fields of an edge (name/host/port/tls/versions). */
function updateEdge(id, patch = {}) {
  const edges = readAll();
  const idx = edges.findIndex((x) => x.id === id);
  if (idx === -1) return null;
  const allowed = ['name', 'host', 'port', 'tls', 'insecureTLS', 'versions', 'provisionedAt'];
  for (const key of allowed) {
    if (key in patch) edges[idx][key] = patch[key];
  }
  // Coerce types so untrusted patch values can't inject (port reaches the edge
  // shell command during provisioning; tls flags reach the TLS client).
  if ('port' in patch) edges[idx].port = Number(patch.port) || 8443;
  if ('tls' in patch) edges[idx].tls = Boolean(patch.tls);
  if ('insecureTLS' in patch) edges[idx].insecureTLS = Boolean(patch.insecureTLS);
  if ('ssh' in patch) edges[idx].ssh = normalizeSsh(patch.ssh);
  writeAll(edges);
  return publicView(edges[idx]);
}

/** Remove an edge. Returns true if it existed. */
function removeEdge(id) {
  const edges = readAll();
  const next = edges.filter((x) => x.id !== id);
  if (next.length === edges.length) return false;
  writeAll(next);
  return true;
}

module.exports = {
  REGISTRY_FILE,
  listEdges,
  getEdge,
  getEdgeWithSecret,
  addEdge,
  updateEdge,
  removeEdge,
};
