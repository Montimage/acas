/**
 * Enumerate the host's network interfaces for the ACAS NIC selector.
 * Reads /sys/class/net (authoritative on Linux, includes down interfaces),
 * falling back to os.networkInterfaces().
 */
const fs = require('fs');
const os = require('os');

function listInterfaces() {
  try {
    const names = fs.readdirSync('/sys/class/net').filter((n) => n !== 'lo' || true);
    return names.map((name) => {
      let state = 'unknown';
      try { state = fs.readFileSync(`/sys/class/net/${name}/operstate`, 'utf8').trim(); } catch (e) { /* ignore */ }
      return { name, state, loopback: name === 'lo' };
    });
  } catch (e) {
    return Object.keys(os.networkInterfaces()).map((name) => ({ name, state: 'unknown', loopback: name === 'lo' }));
  }
}

module.exports = { listInterfaces };
