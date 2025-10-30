<?php
/**
 * FTSC RAG Search Tool - Configuration
 *
 * IMPORTANT: Configure these settings before deployment
 */

// =============================================================================
// CLOUD RUN CONFIGURATION
// =============================================================================

/**
 * Your Google Cloud Run service URL
 * Replace 'YOUR_CLOUD_RUN_URL' with your actual Cloud Run URL
 * Example: 'https://ftsc-rag-search-abc123.a.run.app'
 *
 * DO NOT include a trailing slash
 * define('CLOUD_RUN_URL', 'https://rag-service-55qjbsthda-uc.a.run.app');
 */
define('CLOUD_RUN_URL', 'http://127.0.0.1:8080');

// =============================================================================
// RATE LIMITING CONFIGURATION
// =============================================================================

/**
 * Maximum queries per IP address within the time window
 */
define('RATE_LIMIT_MAX_QUERIES', 20);

/**
 * Time window for rate limiting in seconds (60 = 1 minute)
 */
define('RATE_LIMIT_WINDOW', 60);

/**
 * Maximum conversation history length to send to Cloud Run
 * Helps prevent excessive payload sizes
 */
define('MAX_CONVERSATION_HISTORY', 10);

// =============================================================================
// SECURITY CONFIGURATION
// =============================================================================

/**
 * Session timeout in seconds (3600 = 1 hour)
 */
define('SESSION_TIMEOUT', 3600);

/**
 * Enable/disable CSRF protection
 */
define('CSRF_PROTECTION', true);

/**
 * Maximum query length (characters)
 */
define('MAX_QUERY_LENGTH', 2000);

// =============================================================================
// DEBUGGING (Set to false in production)
// =============================================================================

/**
 * Enable detailed error messages
 * WARNING: Set to false in production to avoid exposing sensitive info
 */
define('DEBUG_MODE', true);

/**
 * Log file path (optional)
 * Set to null to disable file logging
 */
define('LOG_FILE', null); // Example: '/var/log/ftsc-rag/app.log'

// =============================================================================
// VALIDATION
// =============================================================================

// Validate Cloud Run URL is configured
if (CLOUD_RUN_URL === 'YOUR_CLOUD_RUN_URL') {
    if (DEBUG_MODE) {
        die('ERROR: Please configure CLOUD_RUN_URL in config.php');
    } else {
        die('Configuration error. Please contact the administrator.');
    }
}

// Validate Cloud Run URL format
if (!filter_var(CLOUD_RUN_URL, FILTER_VALIDATE_URL)) {
    if (DEBUG_MODE) {
        die('ERROR: Invalid CLOUD_RUN_URL format in config.php');
    } else {
        die('Configuration error. Please contact the administrator.');
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Log a message (if logging is enabled)
 */
function log_message($message, $level = 'INFO') {
    if (LOG_FILE !== null) {
        $timestamp = date('Y-m-d H:i:s');
        $log_entry = "[$timestamp] [$level] $message\n";
        error_log($log_entry, 3, LOG_FILE);
    }

    if (DEBUG_MODE) {
        error_log("[$level] $message");
    }
}

/**
 * Get client IP address
 */
function get_client_ip() {
    $ip = $_SERVER['REMOTE_ADDR'];

    // Check for proxy headers (be careful with these in production)
    if (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) {
        $ip = $_SERVER['HTTP_X_FORWARDED_FOR'];
    } elseif (!empty($_SERVER['HTTP_CLIENT_IP'])) {
        $ip = $_SERVER['HTTP_CLIENT_IP'];
    }

    return $ip;
}

/**
 * Initialize session if not already started
 */
function init_session() {
    if (session_status() === PHP_SESSION_NONE) {
        session_start();

        // Regenerate session ID periodically for security
        if (!isset($_SESSION['created'])) {
            $_SESSION['created'] = time();
        } elseif (time() - $_SESSION['created'] > SESSION_TIMEOUT) {
            session_regenerate_id(true);
            $_SESSION['created'] = time();
        }
    }
}
