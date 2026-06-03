/**
 * Bearer-token authentication. Every request except /health must present
 * `Authorization: Bearer <token>` matching the configured ACAS_TOKEN. The
 * comparison is constant-time to avoid leaking the token via timing.
 */
const crypto = require('crypto');
const { TOKEN } = require('./config');

function timingSafeEqual(a, b) {
  const ab = Buffer.from(String(a));
  const bb = Buffer.from(String(b));
  if (ab.length !== bb.length) return false;
  return crypto.timingSafeEqual(ab, bb);
}

function requireToken(req, res, next) {
  const header = req.headers.authorization || '';
  const match = header.match(/^Bearer\s+(.+)$/i);
  if (!match || !timingSafeEqual(match[1], TOKEN)) {
    return res.status(401).json({ error: 'invalid or missing token' });
  }
  next();
}

module.exports = { requireToken };
