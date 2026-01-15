/* eslint-disable no-unused-vars */
const express = require('express');

const router = express.Router();
const {
  PREDICTION_PATH,
} = require('../constants');
const {
  listFiles, readTextFile, isFileExist,
} = require('../utils/file-utils');

/** Download a prediction .csv file */
router.get('/:predictionId/download', (req, res, next) => {
  const { predictionId } = req.params;
  const sessionManager = require('../utils/sessionManager');
  const session = sessionManager.getSession('prediction', predictionId);
  const isOnlineMode = session?.mode === 'online';

  // Disable caching for online predictions
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');

  // For offline mode, wait until completion
  if (session && session.isRunning && !isOnlineMode) {
    return res.status(202).json({
      status: 'processing',
      message: 'Prediction is still in progress, file not yet available'
    });
  }

  const predictionFilePath = `${PREDICTION_PATH}${predictionId}/predictions.csv`;
  isFileExist(predictionFilePath, (ret) => {
    if (!ret) {
      if (isOnlineMode) {
        return res.status(200).json({
          status: session?.isRunning ? 'running' : 'completed',
          message: 'No predictions file generated yet'
        });
      }
      res.status(404).json({
        error: 'File not found',
        message: `The prediction file of ${predictionId} does not exist`
      });
    } else {
      res.sendFile(predictionFilePath);
    }
  });
});

/**
 * Get a prediction result content (singular - returns CSV file)
 */
router.get('/:predictionId/attack', (req, res, next) => {
  const { predictionId } = req.params;
  const sessionManager = require('../utils/sessionManager');
  const session = sessionManager.getSession('prediction', predictionId);
  
  // Disable caching for online predictions (files are continuously updated)
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');

  // For online mode, allow reading files even while running (they're continuously updated)
  // For offline mode, wait until completion
  const isOnlineMode = session?.mode === 'online';
  if (session && session.isRunning && !isOnlineMode) {
    return res.status(202).send('Prediction is still in progress, attacks file not yet available');
  }

  const predictionFilePath = `${PREDICTION_PATH}${predictionId}/attacks.csv`;
  isFileExist(predictionFilePath, (ret) => {
    if (!ret) {
      // Prediction completed but no attacks file (all flows were normal)
      // Return empty CSV with just a header so client can parse it
      res.setHeader('Content-Type', 'text/csv');
      res.status(200).send(''); // Empty CSV = no attacks
    } else {
      res.sendFile(predictionFilePath);
    }
  });
});

/**
 * Get attacks as JSON (plural - for backward compatibility)
 * For online mode: returns current attacks (continuously updated)
 * For offline mode: returns 202 if still running, final results when complete
 */
router.get('/:predictionId/attacks', (req, res, next) => {
  const { predictionId } = req.params;
  const sessionManager = require('../utils/sessionManager');
  const session = sessionManager.getSession('prediction', predictionId);
  const isOnlineMode = session?.mode === 'online';

  // Disable caching for online predictions
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');

  // For offline mode, wait until completion
  if (session && session.isRunning && !isOnlineMode) {
    return res.status(202).json({
      status: 'processing',
      message: 'Prediction is still in progress, attacks file not yet available'
    });
  }

  const predictionFilePath = `${PREDICTION_PATH}${predictionId}/attacks.csv`;
  isFileExist(predictionFilePath, (ret) => {
    if (!ret) {
      // No attacks file - return empty with appropriate status
      return res.status(200).json({
        attacks: null,
        status: isOnlineMode && session?.isRunning ? 'running' : 'completed',
        message: 'No attacks detected or file not generated yet'
      });
    } else {
      readTextFile(predictionFilePath, (err, content) => {
        if (err) {
          return res.status(500).json({
            error: 'Failed to read attacks file',
            attacks: null
          });
        }
        // Parse attacks count from CSV
        const lines = content.trim().split('\n').filter(line => line.trim() && !line.startsWith('#'));
        const attackCount = lines.length > 1 ? lines.length - 1 : 0; // Subtract header

        res.json({
          attacks: content,
          attackCount,
          status: isOnlineMode && session?.isRunning ? 'running' : 'completed'
        });
      });
    }
  });
});

/**
 * Get normal traffic predictions (CSV file)
 * For online mode: returns current normals (continuously updated)
 * For offline mode: returns 202 if still running, file when complete
 */
router.get('/:predictionId/normal', (req, res, next) => {
  const { predictionId } = req.params;
  const sessionManager = require('../utils/sessionManager');
  const session = sessionManager.getSession('prediction', predictionId);
  const isOnlineMode = session?.mode === 'online';

  // Disable caching for online predictions
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');

  // For offline mode, wait until completion
  if (session && session.isRunning && !isOnlineMode) {
    return res.status(202).json({
      status: 'processing',
      message: 'Prediction is still in progress, normal traffic file not yet available'
    });
  }

  const predictionFilePath = `${PREDICTION_PATH}${predictionId}/normals.csv`;
  isFileExist(predictionFilePath, (ret) => {
    if (!ret) {
      if (isOnlineMode) {
        res.setHeader('Content-Type', 'text/csv');
        return res.status(200).send(''); // Empty CSV = no normal traffic yet
      }
      res.status(404).json({
        error: 'File not found',
        message: `The prediction file for normal traffic of ${predictionId} does not exist`
      });
    } else {
      res.sendFile(predictionFilePath);
    }
  });
});

/**
 * Get normal traffic as JSON (plural - consistent with /attacks)
 * For online mode: returns current normals (continuously updated)
 * For offline mode: returns 202 if still running, final results when complete
 */
router.get('/:predictionId/normals', (req, res, next) => {
  const { predictionId } = req.params;
  const sessionManager = require('../utils/sessionManager');
  const session = sessionManager.getSession('prediction', predictionId);
  const isOnlineMode = session?.mode === 'online';

  // Disable caching for online predictions
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');

  // For offline mode, wait until completion
  if (session && session.isRunning && !isOnlineMode) {
    return res.status(202).json({
      status: 'processing',
      message: 'Prediction is still in progress, normals file not yet available'
    });
  }

  const predictionFilePath = `${PREDICTION_PATH}${predictionId}/normals.csv`;
  isFileExist(predictionFilePath, (ret) => {
    if (!ret) {
      return res.status(200).json({
        normals: null,
        normalCount: 0,
        status: isOnlineMode && session?.isRunning ? 'running' : 'completed',
        message: 'No normal traffic detected or file not generated yet'
      });
    } else {
      readTextFile(predictionFilePath, (err, content) => {
        if (err) {
          return res.status(500).json({
            error: 'Failed to read normals file',
            normals: null
          });
        }
        // Parse normal count from CSV
        const lines = content.trim().split('\n').filter(line => line.trim() && !line.startsWith('#'));
        const normalCount = lines.length > 1 ? lines.length - 1 : 0; // Subtract header

        res.json({
          normals: content,
          normalCount,
          status: isOnlineMode && session?.isRunning ? 'running' : 'completed'
        });
      });
    }
  });
});

// /**
//  * Get a prediction result content
//  */
// router.get('/:predictionId/all', (req, res, next) => {
//   const { predictionId } = req.params;
//   readTextFile(`${PREDICTION_PATH}${predictionId}/predictions.csv`, (err, content) => {
//     if (err) {
//       res.status(401).send({ error: 'Something went wrong!' });
//     } else {
//       res.send({ content });
//     }
//   });
// });

/**
 * Get a prediction result content
 */
router.get('/:predictionId', (req, res, next) => {
  const { predictionId } = req.params;
  const sessionManager = require('../utils/sessionManager');
  
  // Disable caching for online predictions (stats.csv is continuously updated)
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, proxy-revalidate');
  res.setHeader('Pragma', 'no-cache');
  res.setHeader('Expires', '0');

  // Check if prediction is still running
  const session = sessionManager.getSession('prediction', predictionId);
  
  // For online mode, allow reading stats even while running (they're continuously updated)
  // For offline mode, wait until completion
  const isOnlineMode = session?.mode === 'online';
  if (session && session.isRunning && !isOnlineMode) {
    return res.status(202).json({
      status: 'processing',
      message: 'Prediction is still in progress',
      predictionId: predictionId,
      startedAt: session.createdAt
    });
  }

  // Use fs.readFile directly to avoid logging ENOENT errors for online mode
  const fs = require('fs');
  const path = require('path');
  const statsPath = path.join(PREDICTION_PATH, predictionId, 'stats.csv');
  
  fs.readFile(statsPath, 'utf8', (err, prediction) => {
    if (err) {
      // Check if the prediction directory exists
      const predictionDir = path.join(PREDICTION_PATH, predictionId);

      if (!fs.existsSync(predictionDir)) {
        return res.status(404).json({
          error: 'Prediction not found',
          message: `No prediction found with ID: ${predictionId}`
        });
      }

      // Directory exists but stats.csv doesn't
      // For online mode, return empty stats (no predictions completed yet) - don't log error
      // For offline mode, this is an error - log it
      if (isOnlineMode) {
        return res.json({
          totalFlows: 0,
          attackFlows: 0,
          normalFlows: 0,
          status: 'running',
          message: 'No predictions completed yet'
        });
      }

      // Log error for offline mode only
      console.error(`[Prediction Stats] Error reading stats.csv for ${predictionId}:`, err.message);

      if (session && !session.isRunning) {
        return res.status(500).json({
          error: 'Prediction failed',
          message: 'The prediction process completed but did not generate results. Check the prediction logs.',
          predictionId: predictionId
        });
      }

      return res.status(404).json({
        error: 'Results not available',
        message: 'Prediction results file (stats.csv) not found. The prediction may still be processing or may have failed.',
        predictionId: predictionId
      });
    } else {
      // Parse stats.csv and return aggregated results
      // Each line is: total,attacks,normal (cumulative)
      // The last line contains the final totals
      const lines = prediction.trim().split('\n').filter(line => line.trim());

      if (lines.length === 0) {
        return res.json({
          totalFlows: 0,
          attackFlows: 0,
          normalFlows: 0,
          status: isOnlineMode ? 'running' : 'completed'
        });
      }

      // Get the last line for final totals
      const lastLine = lines[lines.length - 1];
      const parts = lastLine.split(',').map(p => parseInt(p.trim(), 10) || 0);

      const normalFlows = parts[0] || 0;
      const attackFlows = parts[1] || 0;
      const totalFlows = parts[2] || 0;

      res.json({
        totalFlows,
        attackFlows,
        normalFlows,
        attackPercentage: totalFlows > 0 ? ((attackFlows / totalFlows) * 100).toFixed(2) : '0.00',
        normalPercentage: totalFlows > 0 ? ((normalFlows / totalFlows) * 100).toFixed(2) : '0.00',
        status: isOnlineMode && session?.isRunning ? 'running' : 'completed',
        intervals: lines.length,
        prediction: prediction
      });
    }
  });
});

/**
 * Get all prediction result list
 */
router.get('/', (req, res, next) => {
  listFiles(PREDICTION_PATH, '*', (files) => {
    res.send({
      predictions: files,
    });
  });
});


module.exports = router;
