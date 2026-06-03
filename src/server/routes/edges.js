/**
 * Edge (remote MMT host) management — mounted under /mmt/hosts.
 *
 * Thin handlers over edgeRegistry + edgeClient. The bearer token is write-only:
 * returned exactly once on creation, never listed afterwards.
 */
const express = require('express');
const fs = require('fs');
const path = require('path');
const {
  listEdges, getEdge, addEdge, updateEdge, removeEdge,
} = require('../mmt/edgeRegistry');
const edgeClient = require('../mmt/edgeClient');
const sshProvisioner = require('../mmt/sshProvisioner');
const ruleManager = require('../utils/ruleManager');
const { MMT_PROBE_CONFIG_PATH } = require('../constants');

const router = express.Router();

// Collect the ACAS rule XML workspace as [{ name, xml }] to push to an edge.
function loadRuleFiles() {
  const dir = ruleManager.XML_DIR;
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir)
    .filter((f) => f.endsWith('.xml'))
    .map((f) => ({ name: f, xml: fs.readFileSync(path.join(dir, f), 'utf8') }));
}

// ACAS's mmt-probe config — pushed to the edge so its feature CSVs match the
// format the ML feature extractor / model expects.
function loadProbeConfig() {
  return fs.existsSync(MMT_PROBE_CONFIG_PATH) ? fs.readFileSync(MMT_PROBE_CONFIG_PATH, 'utf8') : null;
}

// Push ACAS's rules + probe config to an edge (best-effort, non-fatal).
async function syncAcasAssets(id) {
  const client = edgeClient.forEdge(id);
  try { await client.syncRules(loadRuleFiles()); } catch (e) { /* non-fatal */ }
  const conf = loadProbeConfig();
  if (conf) { try { await client.syncProbeConfig(conf); } catch (e) { /* non-fatal */ } }
}

const wrap = (fn) => (req, res) => Promise.resolve(fn(req, res)).catch((e) => {
  res.status(e.status && e.status >= 400 && e.status < 600 ? e.status : 500)
    .json({ error: e.message || 'edge operation failed' });
});

// List edges (no secrets)
router.get('/', (req, res) => res.json({ ok: true, hosts: listEdges() }));

// Add an edge → returns token + bootstrap command ONCE
router.post('/', wrap(async (req, res) => {
  const { name, host, port, tls, insecureTLS, ssh, provision } = req.body || {};
  if (!host) return res.status(400).json({ error: 'host is required' });
  // The SSH-provisioned agent uses a self-signed cert, so default insecureTLS
  // to true when SSH creds are supplied (unless the caller overrides it).
  const insecure = insecureTLS !== undefined ? insecureTLS : Boolean(ssh);
  // The bearer token is generated and stored internally; ACAS injects it into the
  // agent during provisioning, so it is never returned to the caller.
  const { edge } = addEdge({ name, host, port, tls, insecureTLS: insecure, ssh });

  // One-call path: register + provision + sync rules & probe config (needs SSH creds).
  if (provision && edge.ssh) {
    const summary = await sshProvisioner.provision(edge.id);
    await syncAcasAssets(edge.id);
    return res.json({ ok: true, host: getEdge(edge.id), provisioned: true, summary });
  }

  res.json({
    ok: true,
    host: edge,
    nextStep: edge.ssh
      ? `POST /mmt/hosts/${edge.id}/provision`
      : 'Add SSH credentials (PUT /mmt/hosts/{id}) then provision.',
  });
}));

// Provision (or re-provision) the agent on the edge over SSH: ensure Docker,
// build the image, run the container. This is the "one step" deploy.
router.post('/:id/provision', wrap(async (req, res) => {
  if (!getEdge(req.params.id)) return res.status(404).json({ error: 'host not found' });
  const summary = await sshProvisioner.provision(req.params.id);
  // Seed the edge with ACAS's current rules + probe config once the agent is up.
  await syncAcasAssets(req.params.id);
  res.json({ ok: true, summary });
}));

// Tear down the agent container on the edge (keeps the registration).
router.post('/:id/deprovision', wrap(async (req, res) => {
  if (!getEdge(req.params.id)) return res.status(404).json({ error: 'host not found' });
  res.json({ ok: true, ...(await sshProvisioner.deprovision(req.params.id)) });
}));

router.put('/:id', (req, res) => {
  const updated = updateEdge(req.params.id, req.body || {});
  if (!updated) return res.status(404).json({ error: 'host not found' });
  res.json({ ok: true, host: updated });
});

router.delete('/:id', (req, res) => {
  if (!removeEdge(req.params.id)) return res.status(404).json({ error: 'host not found' });
  res.json({ ok: true });
});

// Health check (also caches the reported versions on the edge record)
router.get('/:id/health', wrap(async (req, res) => {
  if (!getEdge(req.params.id)) return res.status(404).json({ error: 'host not found' });
  const health = await edgeClient.forEdge(req.params.id).health();
  if (health && health.versions) updateEdge(req.params.id, { versions: health.versions });
  res.json({ ok: true, health });
}));

// List the edge's NICs
router.get('/:id/interfaces', wrap(async (req, res) => {
  if (!getEdge(req.params.id)) return res.status(404).json({ error: 'host not found' });
  res.json({ ok: true, ...(await edgeClient.forEdge(req.params.id).interfaces()) });
}));

// Re-push the current ACAS rule workspace AND probe config to the edge.
// No image rebuild needed — the agent picks both up for the next session.
router.post('/:id/sync-rules', wrap(async (req, res) => {
  if (!getEdge(req.params.id)) return res.status(404).json({ error: 'host not found' });
  const client = edgeClient.forEdge(req.params.id);
  const rules = await client.syncRules(loadRuleFiles());
  const conf = loadProbeConfig();
  const probeConfig = conf ? await client.syncProbeConfig(conf) : { skipped: 'no config found' };
  res.json({ ok: true, rules, probeConfig });
}));

module.exports = router;
