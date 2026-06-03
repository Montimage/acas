/**
 * SSH provisioner — turns "operator entered an IP + SSH creds" into a running
 * agent container on the edge, with zero manual steps on the edge.
 *
 * Flow (all over one SSH connection):
 *   1. ensure Docker is present (install via get.docker.com if missing),
 *   2. ship a small build context (Dockerfile + MMT debs + agent source),
 *   3. `docker build` the image on the edge (pinned ubuntu:20.04 base → host OS
 *      is irrelevant; the MMT runtime libs live in the image),
 *   4. `docker run` the agent with the bearer token injected,
 *   5. verify the agent answers /health.
 *
 * SSH is used only for provisioning. All runtime control (online/stop/status/
 * reports/rule-sync) goes through the agent's HTTPS API as before.
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const { Client } = require('ssh2');
const { getEdgeWithSecret, updateEdge } = require('./edgeRegistry');

const AGENT_DIR = path.join(__dirname, '../../../mmt-agent'); // repo-root mmt-agent
const PACKAGES_DIR = path.join(__dirname, '../mmt-packages');
const IMAGE_TAG = 'mmt-agent:latest';
const CONTAINER = 'mmt-agent';

// ---- build context (local, no Docker needed) -------------------------------

/**
 * Assemble the Docker build context into a single .tgz and return its path.
 * Layout: Dockerfile, entrypoint.sh, packages/*.deb, agent/<source>.
 */
function buildContextTarball() {
  const out = path.join(os.tmpdir(), `mmt-agent-context-${process.pid}-${Date.now()}.tgz`);
  // One bash script: stage the files, then tar them. Uses only coreutils/tar.
  const script = `
    set -e
    STAGE="$(mktemp -d)"
    cp "${AGENT_DIR}/Dockerfile" "${AGENT_DIR}/entrypoint.sh" "$STAGE/"
    mkdir "$STAGE/packages"
    cp "${PACKAGES_DIR}"/*.deb "$STAGE/packages/"
    mkdir "$STAGE/agent"
    tar cf - -C "${AGENT_DIR}" --exclude=node_modules --exclude=tls --exclude='*.log' \
        server.js lib package.json | tar xf - -C "$STAGE/agent"
    tar czf "${out}" -C "$STAGE" .
    rm -rf "$STAGE"
  `;
  execFileSync('bash', ['-c', script], { stdio: ['ignore', 'ignore', 'pipe'] });
  return out;
}

// ---- ssh helpers ------------------------------------------------------------

function connect(edge) {
  const ssh = edge.ssh;
  if (!ssh || !ssh.user) throw new Error('edge has no SSH credentials');
  const cfg = {
    host: edge.host,
    port: ssh.port || 22,
    username: ssh.user,
    readyTimeout: 20000,
  };
  if (ssh.privateKey) cfg.privateKey = ssh.privateKey;
  else if (ssh.keyPath) cfg.privateKey = fs.readFileSync(ssh.keyPath);
  if (ssh.passphrase) cfg.passphrase = ssh.passphrase;
  if (ssh.password) cfg.password = ssh.password;

  return new Promise((resolve, reject) => {
    const conn = new Client();
    conn.on('ready', () => resolve(conn));
    conn.on('error', reject);
    conn.connect(cfg);
  });
}

// Run a command; resolve { code, stdout, stderr }. Never rejects on non-zero.
function exec(conn, cmd) {
  return new Promise((resolve, reject) => {
    conn.exec(cmd, (err, stream) => {
      if (err) return reject(err);
      let stdout = '';
      let stderr = '';
      stream.on('data', (d) => { stdout += d; });
      stream.stderr.on('data', (d) => { stderr += d; });
      stream.on('close', (code) => resolve({ code, stdout, stderr }));
    });
  });
}

// Run a command and throw with context if it fails.
async function run(conn, cmd, what) {
  const r = await exec(conn, cmd);
  if (r.code !== 0) {
    const detail = (r.stderr || r.stdout || '').trim().slice(-500);
    throw new Error(`${what} failed (exit ${r.code}): ${detail}`);
  }
  return r.stdout;
}

function putFile(conn, localPath, remotePath) {
  return new Promise((resolve, reject) => {
    conn.sftp((err, sftp) => {
      if (err) return reject(err);
      sftp.fastPut(localPath, remotePath, (e) => (e ? reject(e) : resolve()));
    });
  });
}

// ---- provisioning -----------------------------------------------------------

/**
 * Provision (or re-provision) the agent on an edge over SSH.
 * @param {string} edgeId
 * @returns {object} summary { dockerInstalled, built, started, health }
 */
async function provision(edgeId) {
  const edge = getEdgeWithSecret(edgeId);
  if (!edge) throw new Error(`unknown edge: ${edgeId}`);
  if (!edge.ssh) throw new Error('edge has no SSH credentials; cannot provision');

  const sudo = edge.ssh.sudo ? 'sudo ' : '';
  const docker = `${sudo}docker`;
  const tarball = buildContextTarball();
  const remoteTar = `/tmp/mmt-agent-context-${Date.now()}.tgz`;
  const remoteDir = `/tmp/mmt-agent-build-${Date.now()}`;

  let conn;
  const summary = { dockerInstalled: false, built: false, started: false, health: null };
  try {
    conn = await connect(edge);

    // 1. Ensure Docker is present AND usable as this SSH user.
    const canRun = (await exec(conn, `${docker} version >/dev/null 2>&1 && echo ok`)).stdout.includes('ok');
    if (!canRun) {
      const hasBinary = (await exec(conn, 'command -v docker >/dev/null 2>&1 && echo yes')).stdout.includes('yes');
      if (hasBinary) {
        // Installed but not usable → permission/daemon issue, not a missing binary.
        // Don't try to reinstall; give the operator the actionable fix.
        throw new Error(
          "Docker is installed but not usable as SSH user. Add the user to the 'docker' group "
          + '(`sudo usermod -aG docker <user>` then reconnect), or set the host\'s ssh.sudo=true with passwordless sudo.',
        );
      }
      await run(conn, `curl -fsSL https://get.docker.com | ${sudo}sh`, 'docker install');
      summary.dockerInstalled = true;
    }

    // 2. Ship the build context.
    await putFile(conn, tarball, remoteTar);
    await run(conn, `mkdir -p ${remoteDir} && tar xzf ${remoteTar} -C ${remoteDir}`, 'extract context');

    // 3. Build the image on the edge. Capture full output to a log and surface
    // its tail on failure (Docker's final summary line otherwise hides the real
    // error behind truncation).
    const buildLog = `${remoteDir}/build.log`;
    const buildRes = await exec(conn, `${docker} build -t ${IMAGE_TAG} ${remoteDir} > ${buildLog} 2>&1; echo EXIT:$?`);
    const buildExit = Number((buildRes.stdout.match(/EXIT:(\d+)/) || [])[1] ?? 1);
    if (buildExit !== 0) {
      const tail = (await exec(conn, `tail -n 40 ${buildLog}`)).stdout.trim();
      throw new Error(`docker build failed (exit ${buildExit}):\n${tail}`);
    }
    summary.built = true;

    // 4. (Re)start the container with the token injected.
    await exec(conn, `${docker} rm -f ${CONTAINER} 2>/dev/null || true`);
    const runCmd = [
      docker, 'run', '-d',
      '--name', CONTAINER,
      '--restart', 'unless-stopped',
      '--network', 'host',
      '--cap-add=NET_RAW', '--cap-add=NET_ADMIN',
      '-e', `ACAS_TOKEN=${edge.token}`,
      '-e', `AGENT_PORT=${edge.port}`,
      IMAGE_TAG,
    ].join(' ');
    await run(conn, runCmd, 'docker run');
    summary.started = true;

    // 5. Verify the agent is answering locally on the edge.
    const health = await exec(conn, `sleep 2; curl -sk https://localhost:${edge.port}/health`);
    summary.health = health.stdout.trim().slice(0, 500);

    // cleanup remote staging
    await exec(conn, `rm -rf ${remoteTar} ${remoteDir}`);

    updateEdge(edgeId, { provisionedAt: Date.now() });
    return summary;
  } finally {
    if (conn) conn.end();
    try { fs.unlinkSync(tarball); } catch (e) { /* ignore */ }
  }
}

/** Tear down the agent container on the edge (does not remove the registration). */
async function deprovision(edgeId) {
  const edge = getEdgeWithSecret(edgeId);
  if (!edge) throw new Error(`unknown edge: ${edgeId}`);
  if (!edge.ssh) throw new Error('edge has no SSH credentials');
  const docker = edge.ssh.sudo ? 'sudo docker' : 'docker';
  let conn;
  try {
    conn = await connect(edge);
    await exec(conn, `${docker} rm -f ${CONTAINER} 2>/dev/null || true`);
    updateEdge(edgeId, { provisionedAt: null });
    return { ok: true };
  } finally {
    if (conn) conn.end();
  }
}

module.exports = { provision, deprovision };
