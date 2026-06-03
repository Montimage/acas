const express = require('express');
const {
  startMMTOffline,
  startMMTForDataset,
} = require('../mmt/mmt-connector');
const fs = require('fs');
const path = require('path');
const { PCAP_PATH, MMT_PROBE_CONFIG_PATH } = require('../constants');
const { identifyUser } = require('../middleware/userAuth');
const probeService = require('../mmt/probeService');
const edgeClient = require('../mmt/edgeClient');
const { getEdge } = require('../mmt/edgeRegistry');
const edgesRouter = require('./edges');

const router = express.Router();

// Attach user identification to enable user-specific PCAP resolution in mmt-connector
router.use(identifyUser);

// Remote edge (host) management lives under /mmt/hosts
router.use('/hosts', edgesRouter);

// Read hostId from body or query; absent/"local" means the local machine.
const hostIdOf = (req) => req.body?.hostId || req.query?.hostId || 'local';
const isLocal = (hostId) => !hostId || hostId === 'local';

/* View the mmt-probe config. Local = ACAS's config file; remote = the config the
   edge agent is currently using. ?hostId= selects the target. */
router.get('/probe-config', async (req, res) => {
  const hostId = hostIdOf(req);
  try {
    if (isLocal(hostId)) {
      const config = fs.existsSync(MMT_PROBE_CONFIG_PATH) ? fs.readFileSync(MMT_PROBE_CONFIG_PATH, 'utf8') : '';
      return res.send({ ok: true, hostId: 'local', path: MMT_PROBE_CONFIG_PATH, config });
    }
    if (!getEdge(hostId)) return res.status(404).send({ error: `Unknown host: ${hostId}` });
    const r = await edgeClient.forEdge(hostId).getProbeConfig();
    res.send({ ok: true, hostId, ...r });
  } catch (e) {
    res.status(502).send({ error: e.message || 'Failed to read probe config' });
  }
});

/* Edit the mmt-probe config. Local writes ACAS's config file (backing up the
   previous one); remote pushes it to the edge agent. Takes effect on the next
   monitoring session. Body: { config, hostId? }. */
router.put('/probe-config', async (req, res) => {
  const hostId = hostIdOf(req);
  const { config } = req.body || {};
  if (typeof config !== 'string' || !config.trim()) {
    return res.status(400).send({ error: 'config (non-empty string) is required' });
  }
  try {
    if (isLocal(hostId)) {
      if (fs.existsSync(MMT_PROBE_CONFIG_PATH)) fs.copyFileSync(MMT_PROBE_CONFIG_PATH, `${MMT_PROBE_CONFIG_PATH}.bak`);
      fs.writeFileSync(MMT_PROBE_CONFIG_PATH, config, 'utf8');
      return res.send({ ok: true, hostId: 'local', path: MMT_PROBE_CONFIG_PATH, bytes: config.length });
    }
    if (!getEdge(hostId)) return res.status(404).send({ error: `Unknown host: ${hostId}` });
    const r = await edgeClient.forEdge(hostId).syncProbeConfig(config);
    res.send({ ok: true, hostId, ...r });
  } catch (e) {
    res.status(502).send({ error: e.message || 'Failed to update probe config' });
  }
});

/* GET status. Optional ?hostId=&sessionId= targets a remote edge. */
router.get('/', async (req, res) => {
  try {
    const mmtStatus = await probeService.getStatus({
      hostId: hostIdOf(req),
      sessionId: req.query.sessionId,
    });
    res.send({ mmtStatus });
  } catch (e) {
    res.status(502).send({ error: e.message || 'Failed to get status' });
  }
});

/* Start online monitoring. Local (default) or a remote edge via hostId.
   For a remote edge, `engine` selects 'probe' | 'security' | 'both'. */
router.post('/online', async (req, res) => {
  const { netInf, engine } = req.body || {};
  try {
    const mmtStatus = await probeService.startOnline({ hostId: hostIdOf(req), netInf, engine });
    if (mmtStatus.error) return res.status(401).send({ error: mmtStatus.error });
    res.send(mmtStatus);
  } catch (e) {
    res.status(502).send({ error: e.message || 'Failed to start online monitoring' });
  }
});

router.post('/offline', (req, res) => {
  const {
    fileName,
    filePath,
    outputSessionId,
  } = req.body;

  // If filePath is provided (e.g., /tmp/ndr_xxx.pcap), copy it into PCAP_PATH and use its basename
  if (filePath && !fileName) {
    try {
      const basename = path.basename(filePath);
      const dest = path.join(PCAP_PATH, basename);
      if (!fs.existsSync(dest)) {
        fs.copyFileSync(filePath, dest);
      }
      return startMMTOffline(basename, (mmtStatus) => {
        if (mmtStatus.error) {
          res.status(401).send({ error: mmtStatus.error });
        } else {
          console.log(mmtStatus);
          res.send(mmtStatus);
        }
      }, outputSessionId || null, false, req.userId || null);
    } catch (e) {
      return res.status(401).send({ error: e.message || 'Failed to copy pcap' });
    }
  }

  startMMTOffline(fileName, (mmtStatus) => {
    if (mmtStatus.error) {
      res.status(401).send({
        error: mmtStatus.error,
      });
    } else {
      console.log(mmtStatus);
      res.send(mmtStatus);
    }
  }, outputSessionId || null, false, req.userId || null);
});

/* Stop monitoring. Local (default) or a remote edge via ?hostId=&sessionId=. */
router.get('/stop', async (req, res) => {
  try {
    const mmtStatus = await probeService.stop({
      hostId: hostIdOf(req),
      sessionId: req.query.sessionId,
    });
    if (!mmtStatus) return res.send({ error: 'No mmt-probe is running' });
    res.send(mmtStatus);
  } catch (e) {
    res.status(502).send({ error: e.message || 'Failed to stop monitoring' });
  }
});

router.post('/dataset', (req, res) => {
  const {
    datasetName,
  } = req.body;
  startMMTForDataset(datasetName, (mmtStatus) => {
    if (mmtStatus.error) {
      res.status(401).send({
        error: mmtStatus.error,
      });
    } else {
      console.log(mmtStatus);
      res.send(mmtStatus);
    }
  });
});

module.exports = router;
