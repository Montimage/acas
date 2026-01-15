const express = require('express');
const fs = require('fs');
const path = require('path');
const {
  getPredictingStatus,
  startPredicting,
  stopOnlinePrediction,
} = require('../deep-learning/deep-learning-connector');
const { listNetworkInterfaces } = require('../utils/utils');
const { queuePrediction, getJobStatus } = require('../queue/job-queue');
const { handleQueueError, isRedisError } = require('../utils/queueErrorHelper');
const { v4: uuidv4 } = require('uuid');
const { startMMTOffline, getMMTStatus } = require('../mmt/mmt-connector');
const { listFiles } = require('../utils/file-utils');
const { REPORT_PATH } = require('../constants');
const { resolvePcapPath } = require('../utils/pcapResolver');
const { identifyUser } = require('../middleware/userAuth');

const router = express.Router();

// Apply user identification middleware to all routes
router.use(identifyUser);

/**
 * Wait for MMT to finish processing
 * @param {number} timeout - Maximum wait time in ms (default: 5 minutes)
 * @param {number} pollInterval - Polling interval in ms (default: 500ms)
 * @returns {Promise<void>}
 */
const waitForMMTCompletion = (timeout = 300000, pollInterval = 500) => {
  return new Promise((resolve, reject) => {
    const startTime = Date.now();

    const checkStatus = () => {
      const status = getMMTStatus();

      if (!status.isRunning) {
        resolve();
        return;
      }

      if (Date.now() - startTime > timeout) {
        reject(new Error('MMT processing timeout'));
        return;
      }

      setTimeout(checkStatus, pollInterval);
    };

    checkStatus();
  });
};

/**
 * Get the first CSV file from a report directory
 * @param {string} reportId - The report ID (directory name)
 * @returns {Promise<string|null>} - The CSV filename or null
 */
const getReportCsvFile = (reportId) => {
  return new Promise((resolve) => {
    const reportDir = path.join(REPORT_PATH, reportId);
    listFiles(reportDir, '.csv', (files) => {
      if (files && files.length > 0) {
        resolve(files[0]);
      } else {
        resolve(null);
      }
    });
  });
};

router.get('/stop', (req, res) => {
  stopOnlinePrediction((predictingStatus) => {
    res.send({
      predictingStatus,
    });
  });

});
/**
 * Use a selectedModel (modelId) to classify an input traffic
 * The input traffic can be from a pcap file, a dataset or a network interface
 * Example of the predictConfig
 * - Analyze a pcap file
 * {
 *  selectedModel: 'model-001',
 *  inputTraffic: {
 *    type: 'pcapFile',
 *    value: 'my-pcap-file.pcap'
 *  }
 * }
 * - Analyze a dataset
 * {
 *  selectedModel: 'model-001',
 *  inputTraffic: {
 *    type: 'dataset',
 *    value: 'my-dataset-01'
 *  }
 * }
 * - Analyze a live traffic
 * {
 *  selectedModel: 'model-001',
 *  inputTraffic: {
 *    type: 'net',
 *    value: 'eth0'
 *  }
 * }
 */
router.post('/', async (req, res) => {
  const {
    predictConfig,
  } = req.body;
  if (!predictConfig) {
    res.status(401).send({
      error: 'Missing predicting configuration. Please read the docs',
    });
  } else {
    startPredicting(predictConfig, (predictingStatus) => {
      res.send(predictingStatus);
    });
  }
});

router.get('/', (req, res) => {
  res.send({
    predictingStatus: getPredictingStatus(),
  });
});

router.get('/interfaces', (req, res) => {
  const networkInterfaces = listNetworkInterfaces();
  const ipv4Addresses = Object.keys(networkInterfaces).reduce((addresses, interfaceName) => {
    const ipv4Interface = networkInterfaces[interfaceName].find((interface) => interface.family === 'IPv4');

    if (ipv4Interface) {
      addresses.push(`${interfaceName} - ${ipv4Interface.address}`);
    }

    return addresses;
  }, []);

  console.log(ipv4Addresses);

  res.send({
    interfaces: ipv4Addresses,
  });
});

/**
 * POST /api/predict/offline
 * Simplified offline prediction - server handles MMT analysis automatically
 *
 * Body: { modelId: string, pcapFile: string, useQueue?: boolean }
 */
router.post('/offline', async (req, res) => {
  try {
    const { modelId, pcapFile, useQueue } = req.body || {};
    const userId = req.userId;

    if (!modelId) {
      return res.status(400).json({
        error: 'Missing required parameter',
        message: 'modelId is required'
      });
    }

    if (!pcapFile) {
      return res.status(400).json({
        error: 'Missing required parameter',
        message: 'pcapFile is required'
      });
    }

    // Resolve pcap path: checks user uploads -> samples -> legacy PCAP_PATH
    const resolvedPcapPath = resolvePcapPath(pcapFile, userId);
    if (!resolvedPcapPath || !fs.existsSync(resolvedPcapPath)) {
      return res.status(404).json({
        error: 'PCAP file not found',
        message: `The pcap file ${pcapFile} was not found in uploads or samples`
      });
    }

    console.log(`[Prediction] Running MMT analysis on ${pcapFile} (resolved: ${resolvedPcapPath})`);

    // Start MMT offline analysis with userId for path resolution
    const mmtResult = await new Promise((resolve) => {
      startMMTOffline(pcapFile, (status) => {
        resolve(status);
      }, null, false, userId);
    });

    if (mmtResult.error) {
      return res.status(400).json({
        error: 'MMT analysis failed',
        message: mmtResult.error
      });
    }

    console.log(`[Prediction] MMT started with sessionId: ${mmtResult.sessionId}`);

    // Wait for MMT to complete
    try {
      await waitForMMTCompletion();
      console.log('[Prediction] MMT analysis completed');
    } catch (error) {
      return res.status(500).json({
        error: 'MMT analysis timeout',
        message: 'MMT processing took too long'
      });
    }

    // Get the generated report
    const finalReportId = `report-${mmtResult.sessionId}`;
    const finalReportFileName = await getReportCsvFile(finalReportId);

    if (!finalReportFileName) {
      return res.status(500).json({
        error: 'Report generation failed',
        message: 'No CSV file was generated by MMT analysis'
      });
    }

    console.log(`[Prediction] Using report: ${finalReportId}/${finalReportFileName}`)

    // Queue-based approach is ENABLED BY DEFAULT
    const useQueueDefault = process.env.USE_QUEUE_BY_DEFAULT !== 'false';
    const shouldUseQueue = useQueue !== undefined ? useQueue : useQueueDefault;
    let fallbackToSync = false;

    if (shouldUseQueue) {
      console.log('[Prediction] Using queue-based processing for model:', modelId);

      // Generate unique prediction ID
      const predictionId = `predict-${Date.now()}-${uuidv4().substring(0, 8)}`;

      let jobInfo;
      try {
        // Queue the prediction job
        jobInfo = await queuePrediction({
          modelId,
          reportId: finalReportId,
          reportFileName: finalReportFileName,
          predictionId,
          priority: 5
        });
        
        return res.json({
          success: true,
          useQueue: true,
          predictionId,
          jobId: jobInfo.jobId,
          queueName: jobInfo.queueName,
          position: jobInfo.position,
          estimatedWait: jobInfo.estimatedWait,
          message: 'Prediction job queued successfully'
        });
      } catch (error) {
        // Check if it's a Redis connection error
        if (isRedisError(error)) {
          console.warn('[Prediction] Redis unavailable, automatically falling back to sync mode');
          fallbackToSync = true;
          // Fall through to sync processing below
        } else {
          // For non-Redis errors, return the error
          return handleQueueError(res, error, 'Prediction queue');
        }
      }
    }

    // Direct processing (blocking) - used when useQueue=false OR when Redis is unavailable
    console.log('[Prediction] Using direct processing (blocking) for model:', modelId);

    // Use existing direct prediction method (legacy)
    const predictConfig = {
      modelId,
      inputTraffic: {
        type: 'report',
        value: { reportId: finalReportId, reportFileName: finalReportFileName }
      }
    };

    startPredicting(predictConfig, (predictingStatus) => {
      if (predictingStatus.error) {
        return res.status(500).json({
          success: false,
          error: predictingStatus.error
        });
      }

      const response = {
        success: true,
        useQueue: false,
        predictionId: predictingStatus.lastPredictedId,
        ...predictingStatus,
        message: fallbackToSync 
          ? 'Prediction started in sync mode (Redis unavailable, automatic fallback)' 
          : 'Prediction started (blocking mode)'
      };

      if (fallbackToSync) {
        response.warning = 'Redis/Valkey service is unavailable. Automatically switched to synchronous processing mode.';
      }

      res.json(response);
    });

  } catch (error) {
    console.error('[Prediction] Error:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

/**
 * GET /api/predict/job/:jobId
 * Get status of a queued prediction job
 */
router.get('/job/:jobId', async (req, res) => {
  try {
    const { jobId } = req.params;
    const status = await getJobStatus(jobId, 'prediction');
    res.json(status);
  } catch (error) {
    res.status(500).json({
      error: 'Failed to get job status',
      message: error.message
    });
  }
});

/**
 * POST /api/predict/online
 * Start online prediction - combines MMT-probe monitoring with AI prediction
 * Body: { modelId: string, interface: string }
 * 
 * This endpoint:
 * 1. Starts MMT-probe on the specified network interface
 * 2. Automatically runs AI prediction on generated CSV reports
 * 3. Returns prediction results in real-time
 * 
 * No need to manually run tcpdump or orchestrate multiple API calls
 */
router.post('/online', async (req, res) => {
  try {
    const { modelId, interface: netInf } = req.body || {};

    if (!modelId) {
      return res.status(400).json({
        error: 'Missing required parameter: modelId',
        message: 'Please provide a trained model ID'
      });
    }

    if (!netInf) {
      return res.status(400).json({
        error: 'Missing required parameter: interface',
        message: 'Please provide a network interface name (e.g., eth0, lo)'
      });
    }

    // Use the existing startPredicting with type='online'
    const predictConfig = {
      modelId,
      inputTraffic: {
        type: 'online',
        value: {
          netInf
        }
      }
    };

    startPredicting(predictConfig, (predictingStatus) => {
      if (predictingStatus.error) {
        return res.status(500).json({
          success: false,
          error: predictingStatus.error,
          message: 'Failed to start online prediction'
        });
      }

      res.json({
        success: true,
        mode: 'online',
        interface: netInf,
        modelId,
        predictionId: predictingStatus.lastPredictedId,
        sessionId: predictingStatus.config?.inputTraffic?.value?.sessionId,
        isRunning: predictingStatus.isRunning,
        startedAt: predictingStatus.lastPredictedAt,
        message: `Online prediction started on interface ${netInf} using model ${modelId}`
      });
    });

  } catch (error) {
    console.error('[Online Prediction] Error:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

/**
 * GET /api/predict/online/status
 * Get status of online prediction
 */
router.get('/online/status', (req, res) => {
  try {
    const status = getPredictingStatus();
    
    // Check if it's actually an online prediction
    const isOnlineMode = status.config?.inputTraffic?.type === 'online';
    
    res.json({
      success: true,
      mode: isOnlineMode ? 'online' : 'offline',
      isRunning: status.isRunning,
      predictionId: status.lastPredictedId,
      interface: isOnlineMode ? status.config?.inputTraffic?.value?.netInf : null,
      modelId: isOnlineMode ? status.config?.modelId : null,
      startedAt: status.lastPredictedAt,
      config: status.config
    });
  } catch (error) {
    console.error('[Online Prediction Status] Error:', error);
    res.status(500).json({
      error: 'Failed to get online prediction status',
      message: error.message
    });
  }
});

/**
 * POST /api/predict/online/stop
 * Stop online prediction
 */
router.post('/online/stop', (req, res) => {
  try {
    stopOnlinePrediction((predictingStatus) => {
      res.json({
        success: true,
        message: 'Online prediction stopped',
        isRunning: predictingStatus.isRunning,
        predictionId: predictingStatus.lastPredictedId,
        stoppedAt: Date.now()
      });
    });
  } catch (error) {
    console.error('[Online Prediction Stop] Error:', error);
    res.status(500).json({
      error: 'Failed to stop online prediction',
      message: error.message
    });
  }
});

module.exports = router;
