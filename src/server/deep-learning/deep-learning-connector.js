/* eslint-disable no-plusplus */
const fs = require('fs');
const {
  REPORT_PATH,
  LOG_PATH,
  MODEL_PATH,
  PREDICTION_PATH,
  TRAINING_PATH,
  DEEP_LEARNING_PATH,
  DATASETS_PATH,
  ATTACKS_PATH,
  PYTHON_CMD,
} = require('../constants');
const { startMMTOnline, stopMMT } = require('../mmt/mmt-connector');
const probeService = require('../mmt/probeService');
const {
  isFileExist,
  isFileExistSync,
  listFilesAsync,
  createFolderSync,
  writeTextFile,
  listFilesByTypeAsync,
} = require('../utils/file-utils');
const {
  spawnCommand,
  getUniqueId,
  spawnCommandAsync,
} = require('../utils/utils');

const path = require('path');
const sessionManager = require('../utils/sessionManager');

/**
 * The building status
 */
const buildingStatus = {
  isRunning: false, // indicates if the building process is on going
  lastBuildAt: null, // indicates the started time of the last build
  lastBuildId: null, // indicates the id of the last build
  config: null, // the configuration of the last build
};

/**
 * DEPRECATED: The prediction status - kept for backward compatibility
 * Use predictionSessionManager instead for multi-user support
 */
const predictingStatus = {
  isRunning: false, // indicate if ANY online prediction is ongoing (summary for /status)
  lastPredictedAt: null, // indicate the time started time of the last prediction
  lastPredictedId: null, // indicate the last prediction id -> to get the result
  config: null, // the configuration of the last prediction
};

/**
 * Active online predictions, keyed by capture sessionId, so multiple can run
 * concurrently (e.g. several edges, or local + edge). Each watcher loop checks
 * its OWN running flag; stop targets a single session. The global
 * predictingStatus above is kept as a summary for the legacy /status endpoint.
 *   sessionId -> { running: boolean, remote: { hostId, sessionId } | null }
 */
const onlinePredictions = new Map();
const isOnlineRunning = (sessionId) => (
  sessionId != null ? Boolean(onlinePredictions.get(sessionId)?.running) : predictingStatus.isRunning
);

/**
 * The retrain status
 */
const retrainStatus = {
  isRunning: false, // indicate if the retraining process is ongoing
  lastRetrainAt: null, // indicate the time started time of the last retraining model
  config: null, // the configuration of the last retraining model
};

// Get the building status
const getBuildingStatus = () => buildingStatus;

const prepareTraining = (buildConfig, callback) => {
  const {
    datasets,
  } = buildConfig;
  if (!datasets || datasets.length < 2) {
    return callback({
      error: 'Invalid dataset. Please check the document and the example',
    });
  }

  const buildDatasets = [];
  for (let index = 0; index < datasets.length; index++) {
    const {
      isAttack,
      datasetId,
    } = datasets[index];
    const datasetDataPath = `${REPORT_PATH}${datasetId}/`;
    if (!isFileExistSync(datasetDataPath)) {
      return callback({
        error: `Invalid dataset. Dataset ${datasetId} does not exist`,
      });
    }
    const csvFiles = listFilesAsync(datasetDataPath, '.csv');
    for (let index2 = 0; index2 < csvFiles.length; index2++) {
      if (csvFiles[index2] !== 'security-reports.csv') {
        buildDatasets.push({
          csvPath: `${datasetDataPath}${csvFiles[index2]}`,
          isAttack,
        });
      }
    }
  }

  const newBuildConfig = {
    ...buildConfig,
    datasets: buildDatasets,
  };
  const buildId = getUniqueId();
  const trainingPath = `${TRAINING_PATH}${buildId}/`;
  createFolderSync(trainingPath);
  const buildConfigPath = `${trainingPath}build-config.json`;
  return writeTextFile(buildConfigPath, JSON.stringify(newBuildConfig), (error) => {
    if (error) {
      console.log('Failed to create buildConfig file');
      return callback({
        error: 'Failed to prepare the training location',
      });
    }
    return callback({
      buildConfig: buildConfigPath,
      buildId,
    });
  });
};

/**
 *  Start building a model
 * - Verify if all the dataset exist
 * -
 * @param {Object} buildConfig information needed for building a model
 * Example of buildConfig
 * {
 *  datasets: [
 *    {
 *      datasetId: 'my-dataset-01',
 *      isAttack: true
 *    },
 *    {
 *      datasetId: 'my-dataset-02',
 *      isAttack: false
 *    },
 *  ],
 *  total_sample: '4567',
 *  train_ratio: 0.7,
 *  train_parameters: {
 *    nb_epoch_cnn: 5,
 *    nb_epoch_sae: 2,
 *    batch_size_cnn: 16,
 *    batch_size_sae: 32
 *  }
 * }
 * @param {Function} callback callback function after setting up the building process
 */
const startBuildingModel = (buildConfig, callback) => {
  // Generate an Id for the build
  // Create the location of the build
  // Generate the build-configuration file. build-config.json
  // - convert the datasetId to dataPath
  // - Create the build-config.json file
  // - execute the command to build the model

  prepareTraining(buildConfig, (ret) => {
    if (ret.error) {
      callback({
        error: 'Failed to prepare the training location',
      });
    } else {
      console.log(ret);
      console.log('Start building the model');
      buildingStatus.isRunning = true;
      buildingStatus.config = buildConfig;
      buildingStatus.lastBuildAt = Date.now();
      buildingStatus.lastBuildId = ret.buildId;
      const logFilePath = `${LOG_PATH}training_${ret.buildId}.log`;
      spawnCommand(PYTHON_CMD, [`${DEEP_LEARNING_PATH}/deep_learning.py`, ret.buildId, ret.buildConfig], logFilePath, () => {
        buildingStatus.isRunning = false;
      });
      callback(buildingStatus);
    }
  });
};

const getRetrainStatus = () => retrainStatus;

const retrainModel = (retrainConfig, callback) => {
  const {
    modelId,
    trainingDataset,
    testingDataset,
    training_parameters,
  } = retrainConfig;
  console.log(retrainConfig);
  const {
    nb_epoch_cnn,
    nb_epoch_sae,
    batch_size_cnn,
    batch_size_sae,
  } = training_parameters;

  if (retrainStatus.isRunning) {
    console.warn('An building process is on going. Only one process can be run at a time');
    return callback({
      error: 'An building process is on going',
    });
  }

  const retrainId = getUniqueId();
  const retrainPath = `${TRAINING_PATH}${retrainId}/`;
  createFolderSync(retrainPath);
  console.log(retrainPath);
  const retrainConfigPath = `${retrainPath}retrain-config.json`;
  writeTextFile(retrainConfigPath, JSON.stringify(retrainConfig), (error) => {
    if (error) {
      console.log('Failed to create retrainConfig file');
      return callback({
        error: 'Failed to create retrainConfig file',
      });
    }
  });

  const attacksPath = `${ATTACKS_PATH}${modelId.replace('.h5', '')}/`;
  const trainingPath = `${TRAINING_PATH}${modelId.replace('.h5', '')}/datasets/`;

  let trainingDatasetFile = null;
  let testingDatasetFile = null;

  if (isFileExistSync(path.join(attacksPath, trainingDataset))) {
    trainingDatasetFile = path.join(attacksPath, trainingDataset);
  } else if (isFileExistSync(path.join(trainingPath, trainingDataset))) {
    trainingDatasetFile = path.join(trainingPath, trainingDataset);
  } else {
    return callback({
      error: `Invalid training dataset`,
    });
  }
  console.log(path.join(trainingPath, testingDataset));
  if (isFileExistSync(path.join(trainingPath, testingDataset))) {
    testingDatasetFile = path.join(trainingPath, testingDataset);
  } else {
    return callback({
      error: `Invalid testing dataset`,
    });
  }

  console.log(trainingDatasetFile);
  console.log(testingDatasetFile);

  const datasetsPath = `${TRAINING_PATH}${retrainId.replace('.h5', '')}/datasets`;
  createFolderSync(datasetsPath);
  fs.copyFile(trainingDatasetFile, path.join(datasetsPath, 'Train_samples.csv'), (err) => {
    if (err) {
      callback(err);
    }
  });
  fs.copyFile(testingDatasetFile, path.join(datasetsPath, 'Test_samples.csv'), (err) => {
    if (err) {
      callback(err);
    }
  });

  const inputModelFilePath = `${MODEL_PATH}${modelId}`;
  if (!fs.existsSync(inputModelFilePath)) {
    return callback({
      error: `The given model file ${modelId} does not exist`,
    });
  }

  retrainStatus.isRunning = true;
  retrainStatus.config = retrainConfig;
  retrainStatus.lastRetrainId = retrainId;
  retrainStatus.lastRetrainAt = Date.now();

  const logFile = `${LOG_PATH}retraining_${retrainId.replace('.h5', '')}.log`;
  const resultsPath = `${TRAINING_PATH}${retrainId.replace('.h5', '')}/results`;
  createFolderSync(resultsPath);
  spawnCommand(PYTHON_CMD, [`${DEEP_LEARNING_PATH}/retrain.py`, trainingDatasetFile, testingDatasetFile, resultsPath, nb_epoch_cnn, nb_epoch_sae, batch_size_cnn, batch_size_sae,], logFile, () => {
    retrainStatus.isRunning = false;
    console.log('Finish retraining the model');
  });

  return callback({
    retrainConfig: retrainConfigPath,
    retrainId,
  });
};

/**
 * Stop an online prediction
 */
/**
 * Stop online prediction. With a sessionId, stops just that session; without
 * one, stops all active online sessions (backward-compatible single-session
 * behaviour). Each watcher exits on its own once its running flag is cleared.
 */
const stopOnlinePrediction = (callback, sessionId = null) => {
  console.log('Going to stop online prediction', sessionId || '(all)');
  const ids = sessionId ? [sessionId] : [...onlinePredictions.keys()];
  let hadLocal = false;
  const remoteStops = [];
  for (const id of ids) {
    const entry = onlinePredictions.get(id);
    if (!entry) continue;
    entry.running = false;            // the session's watcher loop will exit
    onlinePredictions.delete(id);
    if (entry.remote) remoteStops.push(probeService.stop(entry.remote).catch(() => {}));
    else hadLocal = true;             // local capture → stop the local mmt-probe
  }
  // Summary status reflects whether anything is still running.
  predictingStatus.isRunning = onlinePredictions.size > 0;
  const finish = () => Promise.all(remoteStops).then(() => callback(predictingStatus));
  if (hadLocal) stopMMT(finish);
  else finish();
};

/**
 * Execute prediction on a single CSV file (common logic for both online and offline)
 * @param {String} csvPath - Full path to the CSV file
 * @param {String} modelPath - Path to the model
 * @param {String} predictionPath - Output directory for predictions
 * @param {String} logPath - Log file path
 * @param {Function} onComplete - Callback when prediction completes (exitCode)
 * @returns {Boolean} - True if prediction started, false if skipped
 */
const executePrediction = (csvPath, modelPath, predictionPath, logPath, onComplete) => {
  try {
    // Check if CSV has actual flow data
    const stats = fs.statSync(csvPath);
    const fileSizeKB = stats.size / 1024;
    const fileContent = fs.readFileSync(csvPath, 'utf8');
    const lines = fileContent.trim().split('\n');
    const dataLines = lines.filter(line => line.trim().length > 0 && !line.startsWith('#'));
    
    // Skip if no flow data (< 3 lines = header/metadata only)
    if (dataLines.length < 3) {
      const fileName = path.basename(csvPath);
      console.log(`⏭️  Skipping CSV with no flow data (${dataLines.length} data lines, ${fileSizeKB.toFixed(2)} KB): ${fileName}`);
      if (onComplete) onComplete(0); // Call completion with success code
      return false;
    }
    
    const fileName = path.basename(csvPath);
    console.log(`🔍 Running prediction: ${fileName} (${dataLines.length} data lines, ${fileSizeKB.toFixed(2)} KB)`);
    
    spawnCommand(
      PYTHON_CMD,
      [`${DEEP_LEARNING_PATH}/prediction.py`, csvPath, modelPath, predictionPath],
      logPath,
      (error) => {
        // spawnCommand passes Error object on failure, null on success
        if (!error) {
          console.log(`✅ Prediction completed: ${fileName}`);
          if (onComplete) onComplete(0);
        } else {
          console.error(`❌ Prediction failed: ${fileName}`);
          console.error(`   Error: ${error.message}`);
          console.error(`   Log: ${logPath}`);
          if (onComplete) onComplete(1);
        }
      },
      { suppressOutput: true }
    );
    return true;
  } catch (error) {
    console.error(`Error executing prediction: ${error.message}`);
    if (onComplete) onComplete(1); // Call with error code
    return false;
  }
};

/**
 * Start online prediction process
 * - Read the list of completed report file (which has both .csv and .sem files)
 * - Execute the prediction process for the one that has not been processed (not in the list of processedReports)
 * - Recursive until there is no report to process.
 * @param {String} reportPath Path to the folder where reports are being generated
 * @param {String} modelPath Path to the model which is being used for prediction
 * @param {String} predictionPath Path to the location where the prediction output will be stored
 * @param {String} logPath Path to the log of what's going on with the prediction process
 * @param {Number} currentIndex The index of the report is going to be processed
 */
const startOnlinePrediction = (reportPath, modelPath, predictionPath, logPath, currentIndex, sessionId, waitCount = 0) => {
  if (!isOnlineRunning(sessionId)) {
    console.log('The online prediction process has been terminated');
    return;
  }

  // First call for this index
  if (waitCount === 0) {
    console.log(`startOnlinePrediction: ${currentIndex}`);
  }

  // Only mmt-probe feature reports ("*_data.csv") are model inputs. Exclude
  // other reports the probe config emits (e.g. security-reports.csv), which
  // would otherwise stall the watcher waiting for a .sem that never comes.
  let allCSVFiles = listFilesByTypeAsync(reportPath, '.csv').filter((f) => f.endsWith('_data.csv'));
  let currentReport = allCSVFiles[currentIndex];

  // If CSV file doesn't exist yet, wait and retry
  if (!currentReport) {
    waitCount++;
    // Use setTimeout to avoid blocking the event loop (no logging during wait)
    setTimeout(() => {
      startOnlinePrediction(reportPath, modelPath, predictionPath, logPath, currentIndex, sessionId, waitCount);
    }, 1000);
    return;
  }

  // CSV exists, now check for .sem file
  checkForSemFile(reportPath, modelPath, predictionPath, logPath, currentIndex, currentReport, sessionId, 0);
};

const checkForSemFile = (reportPath, modelPath, predictionPath, logPath, currentIndex, currentReport, sessionId, semWaitCount) => {
  if (!isOnlineRunning(sessionId)) {
    console.log('The online prediction process has been terminated');
    return;
  }

  const allSEMFiles = listFilesByTypeAsync(reportPath, '.sem');
  const currentSemFile = `${currentReport}.sem`;

  // If .sem file doesn't exist yet, decide whether to wait or skip immediately.
  if (allSEMFiles.indexOf(currentSemFile) === -1) {
    // Fast path: an already-empty CSV (header/metadata only) will never produce
    // useful predictions, and mmt-probe doesn't emit a .sem for empty windows.
    // Skip it now instead of stalling the whole sequential pipeline.
    try {
      const lines = fs.readFileSync(`${reportPath}/${currentReport}`, 'utf8').trim().split('\n')
        .filter(l => l.trim().length > 0 && !l.startsWith('#'));
      if (lines.length < 3) {
        startOnlinePrediction(reportPath, modelPath, predictionPath, logPath, currentIndex + 1, sessionId);
        return;
      }
    } catch (e) { /* file not readable yet — fall through to wait */ }

    semWaitCount++;
    // Timeout: the report puller polls every ~3s, so a real .sem arrives within
    // ~6-9s. 30 attempts x 500ms = 15s gives margin without stalling 60s/file.
    if (semWaitCount >= 30) {
      console.log(`⚠️  Timeout waiting for .sem file: ${currentSemFile} - skipping`);
      startOnlinePrediction(reportPath, modelPath, predictionPath, logPath, currentIndex + 1, sessionId);
      return;
    }

    // Use setTimeout to avoid blocking the event loop (no logging during wait)
    setTimeout(() => {
      checkForSemFile(reportPath, modelPath, predictionPath, logPath, currentIndex, currentReport, sessionId, semWaitCount);
    }, 500);
    return;
  }

  // Both CSV and .sem exist, execute prediction using common function
  const csvPath = `${reportPath}/${currentReport}`;

  executePrediction(csvPath, modelPath, predictionPath, logPath, (exitCode) => {
    // Process next report regardless of success/failure/skip
    startOnlinePrediction(reportPath, modelPath, predictionPath, logPath, currentIndex + 1, sessionId);
  });
};

/**
 * Start a predicting process
 * - verify if the model exist
 * - verify if the input traffic exist
 * - Execute the MMT to analyze the traffic
 * - on the callback, execute the prediction to get the result
 *
 * @param {Object} predictConfig configuration relates to the predicting process
 * Example of a predictConfig
 * - Analyze a pcap file
 * {
 *  modelId: 'model-001',
 *  inputTraffic: {
 *    type: 'pcapFile',
 *    value: 'my-pcap-file.pcap'
 *  }
 * }
 * - Analyze a dataset
 * {
 *  modelId: 'model-001',
 *  inputTraffic: {
 *    type: 'dataset',
 *    value: 'my-dataset-01'
 *  }
 * }
 * - Analyze a live traffic
 * {
 *  modelId: 'model-001',
 *  inputTraffic: {
 *    type: 'net',
 *    value: 'eth0'
 *  }
 * }
 *
 * - Analyze a report
 * {
 *  modelId: 'model-01',
 *  inputTraffic:{
 *    type: 'report',
 *    value: {
 *      reportId: 'report-tcp_segmented_fpm.pcap-40f5e3ce-d0a7-4cd7-8f98-d9016bbbfd79',
 *      reportFileName: '1675545880.212144_0_tcp_segmented_fpm.pcap.csv'
 *    }
 *  }
 * }
 * @param {Function} callback callback function after setting up the predicting process
 */
const startPredicting = async (predictConfig, callback) => {
  const {
    modelId,
    inputTraffic,
  } = predictConfig;
  const modelPath = `${MODEL_PATH}${modelId}`;
  const logFilePath = `${LOG_PATH}predict_`;
  const predictionId = getUniqueId();
  const predictionPath = `${PREDICTION_PATH}${predictionId}/`;
  const logFile = `${logFilePath}${predictionId}.log`;
  isFileExist(modelPath, async (exist) => {
    if (!exist) {
      callback({
        error: `Model ${modelId} doest not exist`,
      });
    } else {
      const {
        type,
        value,
      } = inputTraffic;
      switch (type) {
        case 'report':
          // eslint-disable-next-line no-case-declarations
          const csvPath = `${REPORT_PATH}/${value.reportId}/${value.reportFileName}`;
          isFileExist(csvPath, async (report) => {
            if (!report) {
              callback({
                error: `The report file does not exist: ${value.reportFileName}`,
                details: `Report ID: ${value.reportId}. You must first run feature extraction (POST /features/extract) to generate a report before running prediction.`,
                missingFile: csvPath
              });
            } else {
              // Create session in session manager
              const session = sessionManager.createSession('prediction', predictionId, 'offline', { config: predictConfig });

              // Use common prediction execution function
              executePrediction(csvPath, modelPath, predictionPath, logFile, (exitCode) => {
                // Mark session as completed
                sessionManager.completeSession('prediction', predictionId);
              });

              // Return session data in legacy format
              callback({
                isRunning: session.isRunning,
                lastPredictedAt: session.createdAt,
                lastPredictedId: session.sessionId,
                config: session.config
              });
            }
          });
          break;
        case 'online':
          // Start MMTOnline (local) or capture on a remote edge (hostId).
          // eslint-disable-next-line no-case-declarations
          const netInf = value.netInf || value;
          // eslint-disable-next-line no-case-declarations
          const onlineHostId = value.hostId;
          predictingStatus.remote = null;
          if (onlineHostId && onlineHostId !== 'local') {
            // Remote edge: capture features on the edge via the agent, pull them
            // to report-<sessionId>/, then run the SAME prediction watcher on ACAS.
            probeService.startOnline({ hostId: onlineHostId, netInf, engine: 'probe' })
              .then((r) => {
                if (r.error) {
                  callback({ error: r.error, details: 'Edge mmt-probe failed to start for online capture' });
                  return;
                }
                value.sessionId = r.sessionId; // expose the capture session id in status/response
                const session = sessionManager.createSession('prediction', predictionId, 'online', { config: predictConfig });
                const csvRootPath = `${REPORT_PATH}/report-${r.sessionId}`;
                // Register this online session (keyed by capture sessionId) so its
                // watcher runs independently of any other concurrent session.
                onlinePredictions.set(r.sessionId, { running: true, remote: { hostId: onlineHostId, sessionId: r.sessionId } });
                predictingStatus.isRunning = true;
                predictingStatus.lastPredictedAt = session.createdAt;
                predictingStatus.lastPredictedId = session.sessionId;
                predictingStatus.config = session.config;
                predictingStatus.remote = { hostId: onlineHostId, sessionId: r.sessionId };
                callback({
                  isRunning: session.isRunning,
                  lastPredictedAt: session.createdAt,
                  lastPredictedId: session.sessionId,
                  config: session.config,
                });
                startOnlinePrediction(csvRootPath, modelPath, predictionPath, logFile, 0, r.sessionId);
              })
              .catch((e) => callback({ error: e.message || 'Failed to start remote online prediction' }));
            break;
          }
          startMMTOnline(netInf, (mmtStatus) => {
            console.log('MMTStatus:');
            // isRunning: true,
            // sessionId,
            // isOnlineMode: true,
            // startedAt: Date.now(),
            console.log(mmtStatus);

            // Check for MMT errors (e.g., sudo access issues)
            if (mmtStatus && mmtStatus.error) {
              console.error('[Online Prediction] MMT failed to start:', mmtStatus.error);
              callback({
                error: mmtStatus.error,
                details: mmtStatus.details || 'MMT-probe failed to start for online capture'
              });
              return;
            }

            if (mmtStatus && mmtStatus.isRunning) {
              // MMT has been started, start processing the report
              // Create session in session manager
              const session = sessionManager.createSession('prediction', predictionId, 'online', { config: predictConfig });

              const { sessionId } = mmtStatus;
              value.sessionId = sessionId; // expose the capture session id in status/response
              const csvRootPath = `${REPORT_PATH}/report-${sessionId}`;

              // Register this online session (local → remote: null). Keyed by the
              // capture sessionId so its watcher runs independently.
              onlinePredictions.set(sessionId, { running: true, remote: null });
              // Update the summary status for the legacy /status endpoint.
              predictingStatus.isRunning = true;
              predictingStatus.lastPredictedAt = session.createdAt;
              predictingStatus.lastPredictedId = session.sessionId;
              predictingStatus.config = session.config;

              // Return session data in legacy format
              callback({
                isRunning: session.isRunning,
                lastPredictedAt: session.createdAt,
                lastPredictedId: session.sessionId,
                config: session.config
              });

              startOnlinePrediction(csvRootPath, modelPath, predictionPath, logFile, 0, sessionId);
            } else {
              // MMT didn't start and no explicit error - generic failure
              callback({
                error: 'MMT-probe failed to start for online capture',
                details: 'Unknown error - check server logs'
              });
            }
          });
          break;
        case 'pcap':
          // get the pcap file
          // move the pcap file to the pcaps/ folder
          // execute startMMTOffline
          // analysing based on the result of startMMTOffline
          break;
        default:
          callback({
            error: `Unsupported input traffic type: ${type}`,
          });
          break;
      }
    }
  });

  // callback(predictingStatus);
};
/**
 * Get status of the current prediction
 * Returns legacy format for backward compatibility
 */
const getPredictingStatus = () => {
  // Return legacy status format from session manager
  return sessionManager.getLegacyStatus('prediction');
};


module.exports = {
  getBuildingStatus,
  startBuildingModel,
  getRetrainStatus,
  retrainModel,
  getPredictingStatus,
  startPredicting,
  stopOnlinePrediction,
};
