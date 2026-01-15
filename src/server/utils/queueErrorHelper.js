/**
 * Helper to handle queue-related errors consistently across routes
 */

/**
 * Check if an error is related to Redis/Valkey being unavailable
 * @param {Error} error - The error to check
 * @returns {boolean} - True if the error is a Redis connection error
 */
function isRedisError(error) {
  if (!error || !error.message) return false;
  
  return error.message.includes('Redis') ||
    error.message.includes('ECONNREFUSED') ||
    error.message.includes('Connection refused') ||
    error.message.includes('Stream isn\'t writeable') ||
    error.message.includes('enableOfflineQueue');
}

function handleQueueError(res, error, operation = 'Background job processing') {
  console.error(`[Queue Error] ${operation}:`, error.message);

  if (isRedisError(error)) {
    return res.status(503).json({
      error: 'Queue service (Redis) unavailable',
      message: `${operation} is currently unavailable because the Redis/Valkey service is not running.`,
      suggestion: 'Please ensure Redis/Valkey is running on your system, or set "useQueue": false in your request body for direct (non-queued) processing.',
      details: error.message
    });
  }

  // For other errors, return a standard 500 error
  return res.status(500).json({
    error: `${operation} failed`,
    message: error.message || 'An internal error occurred while queuing the job'
  });
}

module.exports = {
  handleQueueError,
  isRedisError
};
