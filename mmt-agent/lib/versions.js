/**
 * Report installed MMT tool versions and capture-capability status so ACAS can
 * pin/verify an edge before using it.
 */
const fs = require('fs');
const { execSync } = require('child_process');
const { resolveBin } = require('./config');

function tryExec(cmd) {
  try { return execSync(cmd, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim(); }
  catch (e) { return null; }
}

// Resolve a binary on PATH (or absolute path), or null if absent.
function which(bin) {
  if (bin.includes('/')) return tryExec(`test -x "${bin}" && echo "${bin}"`);
  return tryExec(`command -v "${bin}"`);
}

function probeVersion(bin) {
  // Only probe if the binary actually exists, so we never capture a shell
  // "not found" message as if it were a version.
  if (!which(bin)) return null;
  const out = tryExec(`"${bin}" --version 2>/dev/null || "${bin}" -h 2>/dev/null | head -n1`);
  if (!out) return null;
  const m = out.match(/\d+\.\d+\.\d+/);
  return m ? m[0] : out.split('\n')[0].slice(0, 60);
}

function getVersions() {
  return {
    probe: probeVersion('mmt-probe'),
    dpi: tryExec("dpkg-query -W -f='${Version}' mmt-dpi 2>/dev/null") || null,
    security: probeVersion(resolveBin('mmt_security')),
  };
}

function captureCapable() {
  // 1. File capabilities on the probe binary.
  const caps = tryExec('getcap "$(command -v mmt-probe)" 2>/dev/null') || '';
  if (/cap_net_raw|cap_net_admin/.test(caps)) return true;

  // 2. This process's effective capabilities — covers a container started with
  //    --cap-add=NET_RAW/NET_ADMIN running as root (file caps may be absent).
  try {
    const m = fs.readFileSync('/proc/self/status', 'utf8').match(/CapEff:\s*([0-9a-fA-F]+)/);
    if (m) {
      const eff = BigInt(`0x${m[1]}`);
      const NET_RAW = 13n; // CAP_NET_RAW
      const NET_ADMIN = 12n; // CAP_NET_ADMIN
      if (((eff >> NET_RAW) & 1n) === 1n || ((eff >> NET_ADMIN) & 1n) === 1n) return true;
    }
  } catch (e) { /* ignore */ }

  // 3. Fallback: passwordless sudo.
  return tryExec('sudo -n true && echo ok') === 'ok';
}

module.exports = { getVersions, captureCapable };
