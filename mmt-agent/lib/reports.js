/**
 * Serve a session's output files to ACAS (the pull side of the pipeline).
 *
 * `list` returns files modified at/after a cursor (epoch ms) so ACAS can pull
 * incrementally; the returned cursor is the newest mtime seen. `read` returns
 * one file's bytes, with strict path-traversal protection.
 */
const fs = require('fs');
const path = require('path');
const { sessionDir } = require('./sessions');

function list(sessionId, since = 0) {
  const dir = sessionDir(sessionId);
  if (!fs.existsSync(dir)) return { files: [], cursor: since };
  let cursor = Number(since) || 0;
  const files = [];
  for (const name of fs.readdirSync(dir)) {
    if (name.endsWith('.log')) continue; // logs stay on the edge
    const full = path.join(dir, name);
    let st;
    try { st = fs.statSync(full); } catch (e) { continue; }
    if (!st.isFile()) continue;
    const mtime = Math.floor(st.mtimeMs);
    if (mtime >= Number(since)) {
      files.push({ name, size: st.size, mtime });
      if (mtime > cursor) cursor = mtime;
    }
  }
  // +1 so the next poll doesn't re-list files at exactly the cursor boundary.
  return { files, cursor: cursor + 1 };
}

function read(sessionId, fileName) {
  const dir = path.resolve(sessionDir(sessionId));
  const target = path.resolve(path.join(dir, fileName));
  // Must resolve to a direct child of the session dir.
  if (path.dirname(target) !== dir) throw Object.assign(new Error('invalid path'), { status: 400 });
  if (!fs.existsSync(target)) throw Object.assign(new Error('not found'), { status: 404 });
  return fs.readFileSync(target);
}

module.exports = { list, read };
