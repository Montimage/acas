const { expressCspHeader, INLINE, NONE, SELF } = require('express-csp-header');
var express = require('express');
const https = require('https');
const fs = require('fs');
var path = require('path');
var cookieParser = require('cookie-parser');
const logger = require('morgan');
const fileUpload = require('express-fileupload');
const swaggerUi = require('swagger-ui-express');
const swaggerDocument = require('./swagger/swagger.json');
var bodyParser = require('body-parser');
const dotenv = require('dotenv');
const cors = require('cors');
const { URL } = require('url');

// Load environment variables from .env (if present)
dotenv.config();
// Derive server configuration, preferring REACT_APP_API_URL when present
const API_URL_STR = process.env.REACT_APP_API_URL;
let derivedProtocol = undefined;
let derivedPort = undefined;
if (API_URL_STR) {
  try {
    const parsed = new URL(API_URL_STR);
    derivedProtocol = parsed.protocol === 'https:' ? 'HTTPS' : 'HTTP';
    // If no explicit port in the URL, fall back to standard ports
    const portFromUrl = parsed.port ? parseInt(parsed.port, 10) : (derivedProtocol === 'HTTPS' ? 443 : 80);
    if (!Number.isNaN(portFromUrl)) {
      derivedPort = portFromUrl;
    }
  } catch (e) {
    console.warn(`[CONFIG] Failed to parse REACT_APP_API_URL='${API_URL_STR}': ${e.message}`);
  }
}

// Server configuration with sensible defaults and overrides
const PROTOCOL = process.env.PROTOCOL || derivedProtocol || 'HTTP';
const SERVER_HOST = process.env.SERVER_HOST || '0.0.0.0';
const SERVER_PORT = (process.env.SERVER_PORT && parseInt(process.env.SERVER_PORT, 10)) || derivedPort || 31057;
const MODE = process.env.MODE || 'SERVER';

const mmtRouter = require('./routes/mmt');
const pcapRouter = require('./routes/pcap');
const reportRouter = require('./routes/report');
const logRouter = require('./routes/log');
const modelRouter = require('./routes/model');
const buildRouter = require('./routes/build');
const retrainRouter = require('./routes/retrain');
const predictionRouter = require('./routes/prediction');
const predictRouter = require('./routes/predict');
const xaiRouter = require('./routes/xai');
const attacksRouter = require('./routes/attacks');
const metricsRouter = require('./routes/metrics');
const securityRouter = require('./routes/security');
const assistantRouter = require('./routes/assistant');
const featuresRouter = require('./routes/features');
const queueRouter = require('./routes/queue');
const dpiRouter = require('./routes/dpi');
const networkRouter = require('./routes/network');

// Handle unhandled promise rejections (e.g., Redis connection errors)
process.on('unhandledRejection', (reason, promise) => {
  // Suppress Redis connection errors - server continues in sync mode
  if (reason && reason.message &&
      (reason.message.includes('ECONNREFUSED') ||
       reason.message.includes('Connection refused') ||
       reason.message.includes("Stream isn't writeable") ||
       reason.message.includes('enableOfflineQueue') ||
       reason.message.includes('Redis'))) {
    // Redis unavailable - already logged by queue initialization
    return;
  }
  // Log other unhandled rejections
  console.error('[Server] Unhandled Promise Rejection:', reason);
});

// Initialize queue workers (this starts all background workers)
require('./queue/workers');

const app = express();
var compression = require('compression');
var helmet = require('helmet');

app.use(compression()); //Compress all routes
app.use(helmet());
app.set("port", SERVER_PORT);
app.use(logger('dev'));
// Set generous limits for JSON and urlencoded bodies to accommodate large payloads
app.use(express.json({ limit: '200mb' }));
app.use(express.urlencoded({ extended: true, limit: '200mb' }));
app.use(cookieParser());
app.use(fileUpload());
// Set up CORS
// Build allowed origins from environment variables
const allowedOrigins = [
  'http://localhost:3000',
  'https://localhost:3000',
  'http://0.0.0.0:3000',
];

// Add custom origins from environment variable (comma-separated)
if (process.env.CORS_ALLOWED_ORIGINS) {
  const customOrigins = process.env.CORS_ALLOWED_ORIGINS.split(',').map(o => o.trim());
  allowedOrigins.push(...customOrigins);
}

// Automatically derive origins from REACT_APP_API_URL if present
if (API_URL_STR) {
  try {
    const parsed = new URL(API_URL_STR);
    const baseOrigin = `${parsed.protocol}//${parsed.hostname}`;

    // Add the base domain with common ports
    allowedOrigins.push(baseOrigin);
    allowedOrigins.push(`http://${parsed.hostname}:3000`);
    allowedOrigins.push(`https://${parsed.hostname}:3000`);

    // If there's a port in the URL, add that too
    if (parsed.port) {
      allowedOrigins.push(`${parsed.protocol}//${parsed.hostname}:${parsed.port}`);
    }
  } catch (e) {
    console.warn(`[CORS] Failed to derive origins from REACT_APP_API_URL: ${e.message}`);
  }
}

// Remove duplicates
const uniqueOrigins = [...new Set(allowedOrigins)];

// Configure CORS with allowed origins
app.use(cors({
  origin: uniqueOrigins,
  methods: ['GET', 'POST', 'DELETE', 'PUT'],
}));
/* // Add headers
app.use((req, res, next) => {
  // Website you wish to allow to connect
  res.setHeader('Access-Control-Allow-Origin', '*');

  // Request methods you wish to allow
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, PUT');

  // Request headers you wish to allow
  res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With, content-type, authorization');

  // Set to true if you need the website to include cookies in the requests sent
  // to the API (e.g. in case you use sessions)
  res.setHeader('Access-Control-Allow-Credentials', true);

  // Log the request
  // logInfo(`${req.method} ${req.protocol}://${req.hostname}${req.path} ${res.statusCode}`);
  // Pass to next layer of middleware
  next();
}); */

// Apply CSP headers but exclude Swagger UI routes
app.use((req, res, next) => {
  // Skip CSP for Swagger UI to allow it to load properly
  if (req.path.startsWith('/docs')) {
    return next();
  }
  expressCspHeader({
    policies: {
      'default-src': [expressCspHeader.NONE],
      'img-src': [expressCspHeader.SELF],
    }
  })(req, res, next);
});

app.use('/api/mmt', mmtRouter);
app.use('/api/pcaps', pcapRouter);
app.use('/api/reports', reportRouter);
app.use('/api/logs', logRouter);
app.use('/api/models', modelRouter);
app.use('/api/build', buildRouter);
app.use('/api/retrain', retrainRouter);
app.use('/api/predictions', predictionRouter);
app.use('/api/predict', predictRouter);
app.use('/api/xai', xaiRouter);
app.use('/api/attacks', attacksRouter);
app.use('/api/metrics', metricsRouter);
app.use('/api/security', securityRouter);
app.use('/api/assistant', assistantRouter);
app.use('/api/features', featuresRouter);
app.use('/api/queue', queueRouter);
app.use('/api/dpi', dpiRouter);
app.use('/api/network', networkRouter);

// Swagger API documentation - available in all modes at /docs
app.use('/docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    mode: MODE
  });
});

if (MODE === 'SERVER') {
  // Check if public/index.html exists and is a real client app
  const publicIndexPath = path.join(__dirname, '../public', 'index.html');
  const hasClientApp = fs.existsSync(publicIndexPath) &&
    fs.statSync(publicIndexPath).size > 1000; // Real React app is larger

  if (hasClientApp) {
    app.use(express.static(path.join(__dirname, '../public')));
    app.get('/*', function (req, res) {
      res.sendFile(publicIndexPath);
    });
  } else {
    // No client app - redirect root to API docs
    app.get('/', (req, res) => {
      res.redirect('/docs');
    });
  }
} else if (MODE === 'API') {
  // start Swagger API server
  app.use('/', swaggerUi.serve, swaggerUi.setup(swaggerDocument));
  app.use(express.static(path.join(__dirname, 'swagger')));
  module.exports = app;
}

module.exports = app;

let server;
if (PROTOCOL === 'HTTP') {
  server = app.listen(SERVER_PORT, SERVER_HOST, () => {
    console.log(`[HTTP SERVER] NDR server started on http://${SERVER_HOST}:${SERVER_PORT}`);

    // Start periodic cleanup of all sessions (every 15 minutes)
    const sessionManager = require('./utils/sessionManager');

    setInterval(() => {
      sessionManager.cleanupOldSessions();
    }, 15 * 60 * 1000); // 15 minutes
  });
}

module.exports = server;
