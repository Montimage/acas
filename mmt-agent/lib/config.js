/**
 * Agent configuration, entirely from the environment (set by the container at
 * `docker run` time), so the deployment is the single source of truth.
 */
const path = require('path');

const WORK_DIR = process.env.AGENT_WORK_DIR || '/var/lib/mmt-agent';

module.exports = {
  PORT: Number(process.env.AGENT_PORT) || 8443,
  // Bearer token ACAS must present. Required — the agent refuses to start without it.
  TOKEN: process.env.ACAS_TOKEN || '',
  // TLS material (PEM). If both are present the agent serves HTTPS; otherwise HTTP
  // (only acceptable on a trusted/loopback network — logged loudly).
  TLS_CERT: process.env.AGENT_TLS_CERT || '',
  TLS_KEY: process.env.AGENT_TLS_KEY || '',

  WORK_DIR,
  SESSIONS_DIR: path.join(WORK_DIR, 'sessions'), // per-session output files
  RULES_DIR: path.join(WORK_DIR, 'mmt-rules'),   // synced rule workspace (xml + compiled rules)
  TMP_DIR: path.join(WORK_DIR, 'tmp'),           // staged .deb uploads
  // mmt-probe config: ACAS pushes its custom config (which defines the feature
  // report format the ML extractor needs); fall back to the deb default.
  SYNCED_PROBE_CONF: path.join(WORK_DIR, 'mmt-probe.conf'),
  DEFAULT_PROBE_CONF: process.env.AGENT_PROBE_CONF || '/opt/mmt/probe/mmt-probe.conf',

  // Use sudo -n for privileged capture unless capabilities are granted on the binaries.
  USE_SUDO: process.env.USE_SUDO !== 'false',

  // Resolve an MMT binary: prefer the deb install location, fall back to PATH.
  resolveBin(name) {
    const fs = require('fs');
    const candidate = `/opt/mmt/security/bin/${name}`;
    if (fs.existsSync(candidate)) return candidate;
    return name; // assume on PATH (e.g. mmt-probe)
  },
};
