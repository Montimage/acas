/**
 * Job Queue Manager for NDR Training Environment
 *
 * Handles 30+ concurrent users by queuing resource-intensive tasks
 * Minimal changes to existing code - just wrap existing functions
 * 
 * IMPORTANT: Redis/Valkey is REQUIRED for queue functionality
 * - Client assumes Redis is always available
 * - Server will automatically fallback to sync mode if Redis is unavailable
 * - Install Redis: sudo apt-get install redis-server
 * - Configure: Set REDIS_URL in .env (default: redis://localhost:6379/2)
 */

const Queue = require('bull');
const path = require('path');

// Redis connection (default: localhost:6379)
// REQUIRED: Redis must be running for queue-based processing
const REDIS_URL = process.env.REDIS_URL || 'redis://127.0.0.1:6379';

// Redis connection options with retry logic
const redisOptions = {
  enableOfflineQueue: true,   // Allow buffering during initial connection
  connectTimeout: 5000,       // Timeout after 5 seconds
  maxRetriesPerRequest: 1,    // Fail fast on request
  retryStrategy: (times) => {
    if (times > 3) {
      // Stop retrying after 3 attempts
      return null;
    }
    const delay = Math.min(times * 500, 1500);
    return delay;
  },
  reconnectOnError: (err) => {
    const targetErrors = ['READONLY', 'ECONNRESET', 'ETIMEDOUT'];
    if (targetErrors.some(e => err.message.includes(e))) {
      return true; // Reconnect on specific errors
    }
    return false;
  }
};

// Create shared queue options
const baseQueueOptions = {
  redis: redisOptions,
  defaultJobOptions: {
    attempts: 3,
    backoff: {
      type: 'exponential',
      delay: 2000
    },
    removeOnComplete: 100,
    removeOnFail: 50
  }
};

// Try to create queues with proper error handling
let featureQueue, trainingQueue, predictionQueue, ruleBasedQueue, xaiQueue, attackQueue, retrainQueue;
let REDIS_AVAILABLE = false;

try {
  // Create queues for different job types
  featureQueue = new Queue('feature-extraction', REDIS_URL, baseQueueOptions);

  trainingQueue = new Queue('model-training', REDIS_URL, {
    ...baseQueueOptions,
    defaultJobOptions: {
      ...baseQueueOptions.defaultJobOptions,
      attempts: 2,
      backoff: {
        type: 'exponential',
        delay: 5000
      },
      removeOnComplete: 50,
      removeOnFail: 25
    }
  });

  predictionQueue = new Queue('prediction', REDIS_URL, baseQueueOptions);

  ruleBasedQueue = new Queue('rule-based-detection', REDIS_URL, {
    ...baseQueueOptions,
    defaultJobOptions: {
      ...baseQueueOptions.defaultJobOptions,
      attempts: 2,
      backoff: {
        type: 'exponential',
        delay: 2000
      }
    }
  });

  xaiQueue = new Queue('xai-explanations', REDIS_URL, {
    ...baseQueueOptions,
    defaultJobOptions: {
      ...baseQueueOptions.defaultJobOptions,
      attempts: 2,
      backoff: {
        type: 'exponential',
        delay: 3000
      },
      timeout: 10 * 60 * 1000 // 10 minutes timeout for XAI (can be slow)
    }
  });

  attackQueue = new Queue('adversarial-attacks', REDIS_URL, {
    ...baseQueueOptions,
    defaultJobOptions: {
      ...baseQueueOptions.defaultJobOptions,
      attempts: 2,
      backoff: {
        type: 'exponential',
        delay: 5000
      },
      removeOnComplete: 50,
      removeOnFail: 25,
      timeout: 15 * 60 * 1000 // 15 minutes timeout for attacks
    }
  });

  retrainQueue = new Queue('model-retraining', REDIS_URL, {
    ...baseQueueOptions,
    defaultJobOptions: {
      ...baseQueueOptions.defaultJobOptions,
      attempts: 2,
      backoff: {
        type: 'exponential',
        delay: 5000
      },
      removeOnComplete: 50,
      removeOnFail: 25,
      timeout: 30 * 60 * 1000 // 30 minutes timeout for retraining
    },
    settings: {
      lockDuration: 30 * 60 * 1000,
      stalledInterval: 5 * 60 * 1000,
      maxStalledCount: 2
    }
  });

  // Add error handlers to prevent crashes
  const queues = [featureQueue, trainingQueue, predictionQueue, ruleBasedQueue, xaiQueue, attackQueue, retrainQueue];
  queues.forEach(queue => {
    queue.on('error', (error) => {
      // Suppress connection errors, only log unexpected errors
      if (!error.message.includes('ECONNREFUSED') && !error.message.includes('Connection refused')) {
        console.error(`[Queue ${queue.name}] Error:`, error.message);
      }
      REDIS_AVAILABLE = false;
    });

    queue.on('failed', (job, err) => {
      console.error(`[Queue ${queue.name}] Job ${job.id} failed:`, err.message);
    });

    queue.on('ready', () => {
      REDIS_AVAILABLE = true;
    });
  });

  // Test connection with timeout - use a more reliable method
  const testConnection = async () => {
    try {
      // Get the Redis client from the queue and ping it
      const client = await featureQueue.client;
      await Promise.race([
        client.ping(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('Connection timeout')), 3000))
      ]);
      REDIS_AVAILABLE = true;
      console.log('[Queue] Successfully connected to Redis/Valkey');
    } catch (error) {
      REDIS_AVAILABLE = false;
      console.log('[Queue] Redis/Valkey not available - queue-based processing disabled');
      console.log('[Queue] Server will run in synchronous mode. Set useQueue=false in API requests.');
    }
  };

  // Call async but don't wait - status will update when connection completes
  testConnection();

} catch (error) {
  REDIS_AVAILABLE = false;
  console.log('[Queue] Failed to initialize queues - Redis/Valkey not available');
  console.log('[Queue] Server will run in synchronous mode. Set useQueue=false in API requests.');

  // Throw error to be caught by workers.js
  throw error;
}

// Configure concurrency (number of parallel workers)
const CONCURRENCY = {
  featureExtraction: parseInt(process.env.FEATURE_WORKERS) || 3,
  modelTraining: parseInt(process.env.TRAINING_WORKERS) || 2,
  prediction: parseInt(process.env.PREDICTION_WORKERS) || 3,
  ruleBasedDetection: parseInt(process.env.RULEBASED_WORKERS) || 2,
  xaiExplanations: parseInt(process.env.XAI_WORKERS) || 1,
  adversarialAttacks: parseInt(process.env.ATTACK_WORKERS) || 2,
  modelRetraining: parseInt(process.env.RETRAIN_WORKERS) || 2
};

/**
 * Add job to feature extraction queue
 */
const queueFeatureExtraction = async (data) => {
  const job = await featureQueue.add('extract', data, {
    priority: data.priority || 5,
    timeout: 5 * 60 * 1000 // 5 minutes timeout
  });

  let position = 0;
  try {
    position = typeof job.getPosition === 'function' ? await job.getPosition() : 0;
  } catch (e) {
    position = 0;
  }

  return {
    jobId: job.id,
    queueName: 'feature-extraction',
    position: position,
    estimatedWait: await estimateWaitTime(featureQueue, job)
  };
};

/**
 * Add job to model training queue
 */
const queueModelTraining = async (data) => {
  const job = await trainingQueue.add('train', data, {
    priority: data.priority || 5,
    timeout: 15 * 60 * 1000 // 15 minutes timeout
  });

  let position = 0;
  try {
    position = typeof job.getPosition === 'function' ? await job.getPosition() : 0;
  } catch (e) {
    position = 0;
  }

  return {
    jobId: job.id,
    queueName: 'model-training',
    position: position,
    estimatedWait: await estimateWaitTime(trainingQueue, job)
  };
};

/**
 * Add job to prediction queue
 */
const queuePrediction = async (data) => {
  const job = await predictionQueue.add('predict', data, {
    priority: data.priority || 5,
    timeout: 5 * 60 * 1000 // 5 minutes timeout
  });

  let position = 0;
  try {
    position = typeof job.getPosition === 'function' ? await job.getPosition() : 0;
  } catch (e) {
    position = 0;
  }

  return {
    jobId: job.id,
    queueName: 'prediction',
    position: position,
    estimatedWait: await estimateWaitTime(predictionQueue, job)
  };
};

/**
 * Add job to rule-based detection queue
 */
const queueRuleBasedDetection = async (data) => {
  const job = await ruleBasedQueue.add('detect', data, {
    priority: data.priority || 5,
    timeout: 5 * 60 * 1000 // 5 minutes timeout
  });

  let position = 0;
  try {
    position = typeof job.getPosition === 'function' ? await job.getPosition() : 0;
  } catch (e) {
    position = 0;
  }

  return {
    jobId: job.id,
    queueName: 'rule-based-detection',
    position: position,
    estimatedWait: await estimateWaitTime(ruleBasedQueue, job)
  };
};

/**
 * Add job to XAI explanations queue (SHAP/LIME)
 */
const queueXAI = async (data) => {
  const { xaiType, modelId } = data;
  const jobName = `${xaiType}-${modelId}`;

  const job = await xaiQueue.add(jobName, data, {
    priority: data.priority || 5,
    timeout: 10 * 60 * 1000 // 10 minutes timeout for XAI
  });

  let position = 0;
  try {
    position = typeof job.getPosition === 'function' ? await job.getPosition() : 0;
  } catch (e) {
    position = 0;
  }

  return {
    jobId: job.id,
    queueName: 'xai-explanations',
    position: position,
    estimatedWait: await estimateWaitTime(xaiQueue, job)
  };
};

/**
 * Add job to adversarial attack queue
 */
const queueAttack = async (data) => {
  const { selectedAttack, modelId } = data;
  const jobName = `${selectedAttack}-${modelId}`;

  const job = await attackQueue.add(jobName, data, {
    priority: data.priority || 5,
    timeout: 15 * 60 * 1000 // 15 minutes timeout for attacks
  });

  let position = 0;
  try {
    position = typeof job.getPosition === 'function' ? await job.getPosition() : 0;
  } catch (e) {
    position = 0;
  }

  return {
    jobId: job.id,
    queueName: 'adversarial-attacks',
    position: position,
    estimatedWait: await estimateWaitTime(attackQueue, job)
  };
};

/**
 * Add job to model retraining queue
 */
const queueRetrain = async (data) => {
  const { modelId, trainingDataset } = data;
  const jobName = `retrain-${modelId}-${trainingDataset}-${Date.now()}`;

  const job = await retrainQueue.add('retrain', data, {
    jobId: jobName,
    priority: data.priority || 5,
    timeout: 20 * 60 * 1000
  });

  let position = 0;
  try {
    position = typeof job.getPosition === 'function' ? await job.getPosition() : 0;
  } catch (e) {
    position = 0;
  }

  return {
    jobId: job.id,
    queueName: 'model-retraining',
    position: position,
    estimatedWait: await estimateWaitTime(retrainQueue, job)
  };
};

/**
 * Estimate wait time
 */
const estimateWaitTime = async (queue, job) => {
  try {
    let position = 0;
    try {
      position = typeof job.getPosition === 'function' ? await job.getPosition() : 0;
    } catch (e) {
      position = 0;
    }

    const metrics = await queue.getJobCounts();
    const avgDuration = {
      'feature-extraction': 60,
      'model-training': 300,
      'prediction': 30,
      'rule-based-detection': 45
    }[queue.name] || 60;

    const waitingJobs = position || metrics.waiting || 0;
    const workers = CONCURRENCY[queue.name.replace('-', '')] || 1;

    const estimatedSeconds = Math.ceil((waitingJobs / workers) * avgDuration);

    return {
      seconds: estimatedSeconds,
      minutes: Math.ceil(estimatedSeconds / 60),
      formatted: formatDuration(estimatedSeconds)
    };
  } catch (error) {
    return { seconds: 0, minutes: 0, formatted: 'Unknown' };
  }
};

/**
 * Format duration
 */
const formatDuration = (seconds) => {
  if (seconds < 60) return `${seconds} seconds`;
  const minutes = Math.ceil(seconds / 60);
  if (minutes === 1) return '1 minute';
  if (minutes < 60) return `${minutes} minutes`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return `${hours}h ${remainingMinutes}m`;
};

/**
 * Get job status
 */
const getJobStatus = async (jobId, queueName) => {
  const queue = {
    'feature-extraction': featureQueue,
    'model-training': trainingQueue,
    'prediction': predictionQueue,
    'rule-based-detection': ruleBasedQueue,
    'xai-explanations': xaiQueue,
    'adversarial-attacks': attackQueue,
    'model-retraining': retrainQueue
  }[queueName];

  if (!queue) {
    throw new Error(`Unknown queue: ${queueName}`);
  }

  const job = await queue.getJob(jobId);

  if (!job) {
    return { status: 'not-found', message: 'Job not found' };
  }

  const state = await job.getState();
  const progress = job.progress();

  let position = null;
  if (state === 'waiting') {
    try {
      position = typeof job.getPosition === 'function' ? await job.getPosition() : null;
    } catch (e) {
      position = null;
    }
  }

  return {
    jobId: job.id,
    status: state,
    progress: progress,
    position: position,
    data: job.data,
    result: job.returnvalue,
    failedReason: job.failedReason,
    processedOn: job.processedOn,
    finishedOn: job.finishedOn,
    estimatedWait: position !== null ? await estimateWaitTime(queue, job) : null
  };
};

/**
 * Get queue statistics
 */
const getQueueStats = async () => {
  const [featureStats, trainingStats, predictionStats, ruleBasedStats, xaiStats, attackStats, retrainStats] = await Promise.all([
    featureQueue.getJobCounts(),
    trainingQueue.getJobCounts(),
    predictionQueue.getJobCounts(),
    ruleBasedQueue.getJobCounts(),
    xaiQueue.getJobCounts(),
    attackQueue.getJobCounts(),
    retrainQueue.getJobCounts()
  ]);

  return {
    featureExtraction: { ...featureStats, workers: CONCURRENCY.featureExtraction },
    modelTraining: { ...trainingStats, workers: CONCURRENCY.modelTraining },
    prediction: { ...predictionStats, workers: CONCURRENCY.prediction },
    ruleBasedDetection: { ...ruleBasedStats, workers: CONCURRENCY.ruleBasedDetection },
    xaiExplanations: { ...xaiStats, workers: CONCURRENCY.xaiExplanations },
    adversarialAttacks: { ...attackStats, workers: CONCURRENCY.adversarialAttacks },
    modelRetraining: { ...retrainStats, workers: CONCURRENCY.modelRetraining },
    total: {
      waiting: (featureStats.waiting || 0) + (trainingStats.waiting || 0) + (predictionStats.waiting || 0) + (ruleBasedStats.waiting || 0) + (xaiStats.waiting || 0) + (attackStats.waiting || 0) + (retrainStats.waiting || 0),
      active: (featureStats.active || 0) + (trainingStats.active || 0) + (predictionStats.active || 0) + (ruleBasedStats.active || 0) + (xaiStats.active || 0) + (attackStats.active || 0) + (retrainStats.active || 0),
      completed: (featureStats.completed || 0) + (trainingStats.completed || 0) + (predictionStats.completed || 0) + (ruleBasedStats.completed || 0) + (xaiStats.completed || 0) + (attackStats.completed || 0) + (retrainStats.completed || 0),
      failed: (featureStats.failed || 0) + (trainingStats.failed || 0) + (predictionStats.failed || 0) + (ruleBasedStats.failed || 0) + (xaiStats.failed || 0) + (attackStats.failed || 0) + (retrainStats.failed || 0)
    }
  };
};

/**
 * Cancel a job
 */
const cancelJob = async (jobId, queueName) => {
  const queue = {
    'feature-extraction': featureQueue,
    'model-training': trainingQueue,
    'prediction': predictionQueue,
    'rule-based-detection': ruleBasedQueue,
    'xai-explanations': xaiQueue,
    'adversarial-attacks': attackQueue,
    'model-retraining': retrainQueue
  }[queueName];

  if (!queue) {
    throw new Error(`Unknown queue: ${queueName}`);
  }

  const job = await queue.getJob(jobId);
  if (job) {
    await job.remove();
    return { success: true, message: 'Job cancelled' };
  }

  return { success: false, message: 'Job not found' };
};

/**
 * Clean up old jobs
 */
const cleanupOldJobs = async (olderThanHours = 24) => {
  const ms = olderThanHours * 60 * 60 * 1000;
  await Promise.all([
    featureQueue.clean(ms, 'completed'),
    featureQueue.clean(ms, 'failed'),
    trainingQueue.clean(ms, 'completed'),
    trainingQueue.clean(ms, 'failed'),
    predictionQueue.clean(ms, 'completed'),
    predictionQueue.clean(ms, 'failed'),
    ruleBasedQueue.clean(ms, 'completed'),
    ruleBasedQueue.clean(ms, 'failed'),
    xaiQueue.clean(ms, 'completed'),
    xaiQueue.clean(ms, 'failed'),
    attackQueue.clean(ms, 'completed'),
    attackQueue.clean(ms, 'failed'),
    retrainQueue.clean(ms, 'completed'),
    retrainQueue.clean(ms, 'failed')
  ]);
};

// Auto-cleanup every 6 hours
setInterval(() => cleanupOldJobs(24), 6 * 60 * 60 * 1000);

module.exports = {
  featureQueue,
  trainingQueue,
  predictionQueue,
  ruleBasedQueue,
  xaiQueue,
  attackQueue,
  retrainQueue,
  queueFeatureExtraction,
  queueModelTraining,
  queuePrediction,
  queueRuleBasedDetection,
  queueXAI,
  queueAttack,
  queueRetrain,
  getJobStatus,
  cancelJob,
  getQueueStats,
  cleanupOldJobs,
  isRedisAvailable: () => REDIS_AVAILABLE
};
