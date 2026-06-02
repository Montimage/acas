/**
 * mmt-security rule management.
 *
 * Rules live in an app-owned workspace so users can add their own. mmt_security
 * loads `./rules` (relative to its cwd) first, falling back to the deb-installed
 * /opt/mmt/security/rules only when ./rules is ABSENT. So we run detection with
 * cwd=WORKSPACE and keep compiled .so files in WORKSPACE/rules; if nothing
 * compiles we remove that folder to fall back to the predefined rules.
 */
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

const WORKSPACE = path.join(__dirname, '../mmt-rules');
const XML_DIR = path.join(WORKSPACE, 'xml');
const SO_DIR = path.join(WORKSPACE, 'rules'); // must be named "rules" for mmt_security
const USER_RULES_FILE = path.join(WORKSPACE, 'user-rules.json'); // ids added at runtime (gitignored)

function resolveBin(name) {
  const candidates = [`/opt/mmt/security/bin/${name}`, name];
  return candidates.find(p => p.includes('/') && fs.existsSync(p)) || candidates[0];
}
const SECURITY_BIN = resolveBin('mmt_security');
const COMPILE_BIN = resolveBin('compile_rule');

let rulesCache = null;

// rule id = numeric prefix of the file name, e.g. "200.detect_x.xml" -> "200"
const ruleIdOf = (file) => (file.match(/^(\d+)\./) || [])[1];

// Track rules added at runtime so predefined rules can't be removed.
function readUserRuleIds() {
  try { return new Set(JSON.parse(fs.readFileSync(USER_RULES_FILE, 'utf8')).map(Number)); }
  catch (e) { return new Set(); }
}
function writeUserRuleIds(set) {
  fs.writeFileSync(USER_RULES_FILE, JSON.stringify([...set]), 'utf8');
}
// Find a rule's xml/so file (by id prefix) in a directory.
const fileForId = (dir, id, ext) =>
  (fs.existsSync(dir) ? fs.readdirSync(dir) : []).find(f => f.endsWith(ext) && ruleIdOf(f) === String(id));

/** Compile one XML rule into SO_DIR. Returns { ok, error }. */
async function compileRule(xmlPath) {
  const base = path.basename(xmlPath).replace(/\.xml$/i, '');
  const out = path.join(SO_DIR, `${base}.so`);
  try {
    await execAsync(`"${COMPILE_BIN}" "${out}" "${xmlPath}"`);
    if (!fs.existsSync(out)) return { ok: false, error: 'compile produced no .so' };
    return { ok: true };
  } catch (e) {
    return { ok: false, error: (e.stderr || e.message || 'compile failed').trim() };
  }
}

/** Compile every XML whose .so is missing. Idempotent; safe to call on each boot. */
async function bootstrap() {
  fs.mkdirSync(XML_DIR, { recursive: true });
  fs.mkdirSync(SO_DIR, { recursive: true });
  const xmls = fs.readdirSync(XML_DIR).filter(f => f.endsWith('.xml'));
  // Prune compiled rules whose XML source was removed, so rules/ mirrors xml/.
  const wanted = new Set(xmls.map(f => f.replace(/\.xml$/i, '.so')));
  for (const so of fs.readdirSync(SO_DIR).filter(f => f.endsWith('.so'))) {
    if (!wanted.has(so)) { try { fs.unlinkSync(path.join(SO_DIR, so)); } catch (e) {} }
  }
  let compiled = 0, failed = 0;
  for (const xml of xmls) {
    const so = path.join(SO_DIR, xml.replace(/\.xml$/i, '.so'));
    if (fs.existsSync(so)) { compiled++; continue; }
    const r = await compileRule(path.join(XML_DIR, xml));
    if (r.ok) compiled++;
    else { failed++; console.warn(`[ruleManager] compile failed: ${xml} — ${r.error}`); }
  }
  // No usable rule -> drop the empty folder so mmt_security uses predefined rules.
  if (fs.readdirSync(SO_DIR).filter(f => f.endsWith('.so')).length === 0) {
    try { fs.rmdirSync(SO_DIR); } catch (e) { /* ignore */ }
  }
  console.log(`[ruleManager] bootstrap done: ${compiled} ready, ${failed} failed (workspace ${WORKSPACE})`);
  rulesCache = null;
}

/** Parse `mmt_security -l` text into rule objects. */
function parseRules(text) {
  const rules = [];
  let cur = null;
  for (const line of String(text).split(/\r?\n/)) {
    const head = line.match(/^\d+\s*-\s*Rule id:\s*(\d+)/);
    if (head) { cur = { id: Number(head[1]), type: '', description: '', events: [] }; rules.push(cur); continue; }
    if (!cur) continue;
    const m = line.match(/^\s*-\s*(type|description|event \d+)\s*:\s*(.*)$/);
    if (!m) continue;
    if (m[1] === 'type') cur.type = m[2].trim();
    else if (m[1] === 'description') cur.description = m[2].trim();
    else cur.events.push(m[2].trim());
  }
  return rules;
}

/** List available rules (from the workspace), annotated with XML availability. */
async function listRules({ refresh = false } = {}) {
  if (rulesCache && !refresh) return rulesCache;
  const { stdout, stderr } = await execAsync(`"${SECURITY_BIN}" -l`, { cwd: WORKSPACE });
  const xmlIds = new Set(
    (fs.existsSync(XML_DIR) ? fs.readdirSync(XML_DIR) : [])
      .filter(f => f.endsWith('.xml')).map(ruleIdOf).filter(Boolean)
  );
  const userIds = readUserRuleIds();
  rulesCache = parseRules(`${stdout}\n${stderr}`)
    .map(r => ({ ...r, hasXml: xmlIds.has(String(r.id)), userAdded: userIds.has(r.id) }))
    .sort((a, b) => a.id - b.id);
  return rulesCache;
}

/** Return the raw XML source of a rule by id, or null. */
function getRuleXml(id) {
  if (!/^\d+$/.test(String(id)) || !fs.existsSync(XML_DIR)) return null;
  const file = fs.readdirSync(XML_DIR).find(f => f.endsWith('.xml') && ruleIdOf(f) === String(id));
  return file ? fs.readFileSync(path.join(XML_DIR, file), 'utf8') : null;
}

/** Add a user rule: write XML, compile it, refresh cache. Returns { ok, error }. */
async function addRule(filename, xml) {
  const safe = path.basename(String(filename || '')).replace(/[^\w.\-]/g, '_');
  if (!safe.endsWith('.xml') || !ruleIdOf(safe)) {
    return { ok: false, error: 'filename must be "<id>.<name>.xml", e.g. 300.my_rule.xml' };
  }
  // The filename's id prefix must match the rule's property_id, otherwise the
  // catalog (keyed on property_id reported by mmt_security) won't match the XML.
  const fileId = ruleIdOf(safe);
  const propMatch = String(xml).match(/property_id\s*=\s*["'](\d+)["']/);
  if (!propMatch) return { ok: false, error: 'could not find property_id="..." in the XML' };
  if (propMatch[1] !== fileId) {
    return { ok: false, error: `Rule ID mismatch: filename has id ${fileId} but the XML has property_id="${propMatch[1]}". Make them identical (rename to ${propMatch[1]}.<name>.xml or set property_id="${fileId}").` };
  }
  // Reject collisions with an existing rule id (predefined or already added).
  const existing = fileForId(XML_DIR, fileId, '.xml');
  if (existing) {
    return { ok: false, error: `Rule id ${fileId} already exists (${existing}). Choose a different id, or remove the existing rule first.` };
  }
  fs.mkdirSync(XML_DIR, { recursive: true });
  fs.mkdirSync(SO_DIR, { recursive: true });
  const xmlPath = path.join(XML_DIR, safe);
  fs.writeFileSync(xmlPath, xml, 'utf8');
  const r = await compileRule(xmlPath);
  if (!r.ok) { try { fs.unlinkSync(xmlPath); } catch (e) {} return r; }
  const userIds = readUserRuleIds();
  userIds.add(Number(fileId));
  writeUserRuleIds(userIds);
  rulesCache = null;
  return { ok: true, id: Number(fileId) };
}

/** Edit a user-added rule's XML (id is fixed). Rolls back if it won't compile. */
async function updateRule(id, xml) {
  const sid = String(id);
  if (!readUserRuleIds().has(Number(id))) return { ok: false, error: 'Only user-added rules can be edited' };
  const m = String(xml).match(/property_id\s*=\s*["'](\d+)["']/);
  if (!m) return { ok: false, error: 'could not find property_id="..." in the XML' };
  if (m[1] !== sid) return { ok: false, error: `Rule ID mismatch: this rule's id is ${sid} but the XML has property_id="${m[1]}". Keep property_id="${sid}".` };
  const file = fileForId(XML_DIR, id, '.xml');
  if (!file) return { ok: false, error: `Rule ${sid} not found` };
  const xmlPath = path.join(XML_DIR, file);
  const backup = fs.readFileSync(xmlPath, 'utf8');
  fs.writeFileSync(xmlPath, xml, 'utf8');
  const r = await compileRule(xmlPath);
  if (!r.ok) { fs.writeFileSync(xmlPath, backup, 'utf8'); await compileRule(xmlPath); return r; }
  rulesCache = null;
  return { ok: true, id: Number(id) };
}

/** Remove a user-added rule (xml + so). Predefined rules cannot be removed. */
function deleteRule(id) {
  const userIds = readUserRuleIds();
  if (!userIds.has(Number(id))) return { ok: false, error: 'Only user-added rules can be removed' };
  for (const [dir, ext] of [[XML_DIR, '.xml'], [SO_DIR, '.so']]) {
    const f = fileForId(dir, id, ext);
    if (f) { try { fs.unlinkSync(path.join(dir, f)); } catch (e) {} }
  }
  userIds.delete(Number(id));
  writeUserRuleIds(userIds);
  rulesCache = null;
  return { ok: true };
}

module.exports = { WORKSPACE, XML_DIR, SO_DIR, bootstrap, listRules, getRuleXml, addRule, updateRule, deleteRule };
