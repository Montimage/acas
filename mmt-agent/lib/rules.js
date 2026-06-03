/**
 * Rule workspace on the edge. ACAS pushes its rule XML set here; we compile each
 * into RULES_DIR/rules/*.so. mmt_security is then run with cwd=RULES_DIR so it
 * loads ./rules (ACAS's rules) in preference to the deb defaults — mirroring how
 * ACAS itself runs locally.
 */
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { RULES_DIR, resolveBin } = require('./config');

const XML_DIR = path.join(RULES_DIR, 'xml');
const SO_DIR = path.join(RULES_DIR, 'rules'); // must be named "rules" for mmt_security
const COMPILE_BIN = resolveBin('compile_rule');

// rule file names look like "200.my_rule.xml"; keep only safe basenames.
function safeXmlName(name) {
  const base = path.basename(String(name || ''));
  return /^[\w.\-]+\.xml$/.test(base) ? base : null;
}

function compileOne(xmlPath) {
  const out = path.join(SO_DIR, path.basename(xmlPath).replace(/\.xml$/i, '.so'));
  try {
    execFileSync(COMPILE_BIN, [out, xmlPath], { stdio: ['ignore', 'pipe', 'pipe'] });
    return fs.existsSync(out);
  } catch (e) {
    return false;
  }
}

/**
 * Replace the rule workspace with the provided files and recompile.
 * files = [{ name, xml }].
 */
function syncRules(files) {
  if (!Array.isArray(files)) throw new Error('files must be an array');
  // Reset the workspace so it mirrors ACAS exactly (no stale rules).
  fs.rmSync(XML_DIR, { recursive: true, force: true });
  fs.rmSync(SO_DIR, { recursive: true, force: true });
  fs.mkdirSync(XML_DIR, { recursive: true });
  fs.mkdirSync(SO_DIR, { recursive: true });

  let compiled = 0;
  let failed = 0;
  for (const f of files) {
    const name = safeXmlName(f && f.name);
    if (!name || typeof f.xml !== 'string') { failed++; continue; }
    const xmlPath = path.join(XML_DIR, name);
    fs.writeFileSync(xmlPath, f.xml, 'utf8');
    if (compileOne(xmlPath)) compiled++; else failed++;
  }

  // If nothing compiled, drop ./rules so mmt_security falls back to deb defaults.
  if (fs.readdirSync(SO_DIR).filter((f) => f.endsWith('.so')).length === 0) {
    fs.rmSync(SO_DIR, { recursive: true, force: true });
  }
  return { compiled, failed, workspace: RULES_DIR };
}

module.exports = { syncRules, RULES_DIR, XML_DIR, SO_DIR };
