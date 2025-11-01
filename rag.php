<?php
/**
 * FTSC RAG Search Tool - Combined Application (Two-Column Layout)
 *
 * This single file contains both configuration and application logic.
 * Configure the settings in the CONFIGURATION SECTION below before deployment.
 */

// =============================================================================
// ⚙️ CONFIGURATION SECTION - CUSTOMIZE THESE SETTINGS
// =============================================================================

// Load .env file for local development
function load_env_file($file_path) {
    if (!file_exists($file_path)) {
        return;
    }

    $lines = file($file_path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
    foreach ($lines as $line) {
        // Skip comments
        if (strpos(trim($line), '#') === 0) {
            continue;
        }

        // Parse KEY=VALUE
        if (strpos($line, '=') !== false) {
            list($key, $value) = explode('=', $line, 2);
            $key = trim($key);
            $value = trim($value);

            // Set as environment variable if not already set
            if (!getenv($key)) {
                putenv("$key=$value");
            }
        }
    }
}

// Load .env from the same directory as this file
load_env_file(__DIR__ . '/.env');

// -----------------------------------------------------------------------------
// CLOUD RUN CONFIGURATION
// -----------------------------------------------------------------------------

define('CLOUD_RUN_URL', getenv('CLOUD_RUN_URL') ?:'no url found');
//define('CLOUD_RUN_URL', 'http://99.45.36.114:5001');

define('INTERNAL_API_KEY', getenv('INTERNAL_API_KEY') ?: 'no key found');

// -----------------------------------------------------------------------------
// RATE LIMITING CONFIGURATION
// -----------------------------------------------------------------------------

/** * Maximum queries per IP address within the time window */
define('RATE_LIMIT_MAX_QUERIES', 20);

/** * Time window for rate limiting in seconds (60 = 1 minute) */
define('RATE_LIMIT_WINDOW', 60);

/** * Maximum conversation history length to send to Cloud Run. Helps prevent excessive payload sizes */
define('MAX_CONVERSATION_HISTORY', 10);

// -----------------------------------------------------------------------------
// SECURITY CONFIGURATION
// -----------------------------------------------------------------------------

/** * Session timeout in seconds */
define('SESSION_TIMEOUT', 3600);

/** * Enable/disable CSRF protection */
define('CSRF_PROTECTION', true);

/** * Maximum query length (characters) */
define('MAX_QUERY_LENGTH', 2000);

// -----------------------------------------------------------------------------
// DEBUGGING (Set to false in production)
// -----------------------------------------------------------------------------

/** * Enable detailed error messages, WARNING: Set to false in production to avoid exposing sensitive info */
define('DEBUG_MODE', true);

/** * Log file path (optional). Set to null to disable file logging */
define('LOG_FILE', null); // Example: '/var/log/ftsc-rag/app.log'

// =============================================================================
// 🔧 CONFIGURATION VALIDATION
// =============================================================================

// Validate Cloud Run URL is configured
if (CLOUD_RUN_URL === 'YOUR_CLOUD_RUN_URL') {
    if (DEBUG_MODE) {
        die('ERROR: Please configure CLOUD_RUN_URL in this file (around line 24)');
    } else {
        die('Configuration error. Please contact the administrator.');
    }
}

// Validate Cloud Run URL format
if (!filter_var(CLOUD_RUN_URL, FILTER_VALIDATE_URL)) {
    if (DEBUG_MODE) {
        die('ERROR: Invalid CLOUD_RUN_URL format in configuration section');
    } else {
        die('Configuration error. Please contact the administrator.');
    }
}

// =============================================================================
// 🛠️ HELPER FUNCTIONS
// =============================================================================

/** * Log a message (if logging is enabled) */
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

/** * Get client IP address */
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

/** * Initialize session if not already started */
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

// =============================================================================
// 📱 APPLICATION LOGIC
// =============================================================================

// Initialize session
init_session();

// =============================================================================
// RATE LIMITING
// =============================================================================

function check_rate_limit() {
    $ip = get_client_ip();
    $current_time = time();

    // Initialize rate limit tracking in session
    if (!isset($_SESSION['rate_limit'])) {
        $_SESSION['rate_limit'] = [];
    }

    // Clean up old entries
    $_SESSION['rate_limit'] = array_filter($_SESSION['rate_limit'], function($timestamp) use ($current_time) {
        return ($current_time - $timestamp) < RATE_LIMIT_WINDOW;
    });

    // Check if rate limit exceeded
    if (count($_SESSION['rate_limit']) >= RATE_LIMIT_MAX_QUERIES) {
        return false;
    }

    // Add current request
    $_SESSION['rate_limit'][] = $current_time;
    return true;
}

// =============================================================================
// CSRF PROTECTION
// =============================================================================

function get_csrf_token() {
    if (!isset($_SESSION['csrf_token'])) {
        $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
    }
    return $_SESSION['csrf_token'];
}

function verify_csrf_token($token) {
    if (!CSRF_PROTECTION) {
        return true;
    }
    return isset($_SESSION['csrf_token']) && hash_equals($_SESSION['csrf_token'], $token);
}

// =============================================================================
// API PROXY FUNCTIONS
// =============================================================================

function proxy_to_cloud_run($endpoint, $method = 'GET', $data = null) {
    $url = CLOUD_RUN_URL . $endpoint;

    $ch = curl_init($url);

    // Set curl options
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_FOLLOWLOCATION, true);
    curl_setopt($ch, CURLOPT_TIMEOUT, 60);

    // Debug: Log the API key being used
    log_message("Using API key: " . substr(INTERNAL_API_KEY, 0, 10) . "...", "DEBUG");

    // Prepare headers - always include API key
    $headers = ['X-API-Key: ' . INTERNAL_API_KEY];

    // Set method and data
    if ($method === 'POST') {
        curl_setopt($ch, CURLOPT_POST, true);
        if ($data !== null) {
            curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
            $headers[] = 'Content-Type: application/json';
            $headers[] = 'Content-Length: ' . strlen(json_encode($data));
        }
    }

    // Set headers for all requests
    curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);

    // Execute request
    $response = curl_exec($ch);
    $http_code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $error = curl_error($ch);

    curl_close($ch);

    // Handle errors
    if ($response === false) {
        log_message("cURL error: $error", 'ERROR');
        return [
            'success' => false,
            'error' => DEBUG_MODE ? "Connection error: $error" : 'Service temporarily unavailable',
            'http_code' => 500
        ];
    }

    // Parse JSON response
    $json_response = json_decode($response, true);
    if ($json_response === null) {
        log_message("JSON decode error: $response", 'ERROR');
        return [
            'success' => false,
            'error' => 'Invalid response from service',
            'http_code' => 500
        ];
    }

    return [
        'success' => true,
        'data' => $json_response,
        'http_code' => $http_code
    ];
}

// =============================================================================
// API ENDPOINTS
// =============================================================================

// Handle API requests
if (isset($_GET['api'])) {
    header('Content-Type: application/json');

    $endpoint = $_GET['api'];

    // Rate limiting for query endpoint
    if ($endpoint === 'query' && !check_rate_limit()) {
        http_response_code(429);
        echo json_encode([
            'error' => 'Rate limit exceeded. Please wait a moment before trying again.'
        ]);
        exit;
    }

    // CSRF check for POST requests
    if ($_SERVER['REQUEST_METHOD'] === 'POST' && CSRF_PROTECTION) {
        $input = json_decode(file_get_contents('php://input'), true);
        if (!isset($input['csrf_token']) || !verify_csrf_token($input['csrf_token'])) {
            http_response_code(403);
            echo json_encode(['error' => 'Invalid security token']);
            exit;
        }
        unset($input['csrf_token']); // Remove token before forwarding
    } else {
        $input = json_decode(file_get_contents('php://input'), true);
    }

    // Route to appropriate endpoint
    switch ($endpoint) {
        case 'status':
            $result = proxy_to_cloud_run('/status', 'GET');
            break;

        case 'initialize':
            $result = proxy_to_cloud_run('/initialize', 'POST');
            break;

        case 'query':
            // Validate input
            if (!isset($input['query']) || empty(trim($input['query']))) {
                http_response_code(400);
                echo json_encode(['error' => 'Query cannot be empty']);
                exit;
            }

            // Validate query length
            if (strlen($input['query']) > MAX_QUERY_LENGTH) {
                http_response_code(400);
                echo json_encode(['error' => 'Query too long (max ' . MAX_QUERY_LENGTH . ' characters)']);
                exit;
            }

            // Limit conversation history
            if (isset($input['conversation_history']) && is_array($input['conversation_history'])) {
                $input['conversation_history'] = array_slice($input['conversation_history'], -MAX_CONVERSATION_HISTORY);
            }

            $result = proxy_to_cloud_run('/query', 'POST', $input);
            break;

        case 'refresh-files':
            $result = proxy_to_cloud_run('/refresh-files', 'POST');
            break;

        default:
            http_response_code(404);
            echo json_encode(['error' => 'Unknown endpoint']);
            exit;
    }

    // Return result
    if ($result['success']) {
        http_response_code($result['http_code']);
        echo json_encode($result['data']);
    } else {
        http_response_code($result['http_code']);
        echo json_encode(['error' => $result['error']]);
    }

    exit;
}

// =============================================================================
// HTML FRONTEND
// =============================================================================
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FTSC RAG Search Tool</title>

    <!-- Add markdown-it library -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/markdown-it/13.0.1/markdown-it.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #374151;
            height: 100vh;
            overflow: hidden;
        }

        /* Guidelines Modal - Full Screen Overlay */
        .guidelines-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }

        .guidelines-modal {
            background: white;
            border-radius: 20px;
            padding: 40px;
            max-width: 600px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }

        .guidelines-modal h2 {
            color: #1e293b;
            margin-bottom: 20px;
            font-size: 1.8em;
        }

        .guidelines-content {
            margin-bottom: 25px;
            line-height: 1.8;
            color: #334155;
        }

        .agreement-checkbox {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
            padding: 15px;
            background: #f8fafc;
            border-radius: 10px;
        }

        .agreement-checkbox input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }

        .agreement-checkbox label {
            font-weight: 600;
            color: #1e293b;
            cursor: pointer;
            user-select: none;
        }

        .proceed-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }

        .proceed-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .proceed-btn:not(:disabled):hover {
            transform: translateY(-2px);
        }

        /* Two-Column Grid Layout */
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 3fr;
            height: 100vh;
            gap: 0;
        }

        /* Left Column - Info Side */
        .left-column {
            background: white;
            overflow-y: auto;
            border-right: 2px solid #e2e8f0;
        }

        /* Right Column - Conversation Side */
        .right-column {
            background: #f8fafc;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* Header in Left Column */
        .header {
            background: #13AFFC;
            color: white;
            padding: 30px;
        }

        .header-logo {
            height: 100px;
            width: auto;
            margin-bottom: 15px;
        }

        .header-content h1 {
            font-size: 2em;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .header-content p {
            font-size: 1em;
            opacity: 0.9;
            line-height: 1.4;
        }

        /* Instruction Block */
        .instruction-block {
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: 15px;
            padding: 25px;
            margin: 20px;
            color: #0c4a6e;
            line-height: 1.6;
        }

        .instruction-block p {
            margin-bottom: 12px;
        }

        .instruction-block ol {
            margin: 12px 0;
            padding-left: 24px;
        }

        .instruction-block li {
            margin: 8px 0;
        }

        .instruction-block strong {
            font-weight: 600;
            color: #1e293b;
        }

        /* Info Block in Left Column */
        .info-block {
            background: #f1f5f9;
            padding: 25px;
            margin: 20px;
            border-radius: 15px;
            font-size: 0.95em;
            color: #334155;
            line-height: 1.6;
        }

        .info-block h2 {
            font-size: 1.4em;
            margin-bottom: 15px;
            color: #1e293b;
        }

        .info-block p {
            margin-bottom: 1em;
        }

        /* Conversation Area in Right Column */
        .conversation-wrapper {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }

        .conversation-section {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .conversation-header {
            background: #4f46e5;
            color: white;
            padding: 15px 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }

        .conversation-title {
            font-size: 1.2em;
            font-weight: 600;
        }

        .clear-chat-btn {
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: background 0.2s;
        }

        .clear-chat-btn:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        .conversation-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }

        .message {
            margin-bottom: 20px;
            display: flex;
            gap: 15px;
        }

        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: 600;
            color: white;
            flex-shrink: 0;
        }

        .user-avatar {
            background: #3b82f6;
        }

        .assistant-avatar {
            background: #10b981;
        }

        .message-content {
            flex: 1;
            background: white;
            border-radius: 15px;
            padding: 15px 20px;
            border: 1px solid #e2e8f0;
            line-height: 1.6;
        }

        .message-content h1, .message-content h2, .message-content h3, .message-content h4, .message-content h5, .message-content h6 {
            margin-top: 15px;
            margin-bottom: 8px;
            color: #1e293b;
        }
        .message-content h1 {
            border-bottom: 2px solid #4f46e5;
            padding-bottom: 5px;
            font-size: 1.6em;
        }
        .message-content h2 {
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 3px;
            font-size: 1.3em;
        }
        .message-content h3 {
            font-size: 1.1em;
        }
        .message-content p {
            margin: 10px 0;
        }
        .message-content ul, .message-content ol {
            margin: 10px 0;
            padding-left: 20px;
        }
        .message-content li {
            margin: 5px 0;
        }
        .message-content blockquote {
            margin: 12px 0;
            padding: 10px 12px;
            background: #f1f5f9;
            border-left: 3px solid #64748b;
            font-style: italic;
            border-radius: 3px;
        }
        .message-content code {
            background: #f1f5f9;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.9em;
        }
        .message-content pre {
            background: #1e293b;
            color: #e2e8f0;
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 12px 0;
        }
        .message-content pre code {
            background: none;
            padding: 0;
            color: inherit;
        }
        .message-content strong {
            font-weight: 600;
            color: #1e293b;
        }
        .message-content em {
            font-style: italic;
        }
        .message-content table {
            border-collapse: collapse;
            width: 100%;
            margin: 12px 0;
        }
        .message-content th, .message-content td {
            border: 1px solid #e2e8f0;
            padding: 8px 12px;
            text-align: left;
        }
        .message-content th {
            background: #f8fafc;
            font-weight: 600;
        }
        .message-content a {
            color: #4f46e5;
            text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: border-color 0.2s;
        }
        .message-content a:hover {
            border-bottom-color: #4f46e5;
        }

        /* Query Section - Fixed at Bottom Right */
        .query-section {
            background: white;
            border-top: 2px solid #e2e8f0;
            padding: 10px;
            flex-shrink: 0;
            display: flex;
            flex-direction: row;
            gap: 15px;
            align-items: flex-start;
        }

        .query-form {
            display: flex;
            flex-direction: row;
            gap: 10px;
            align-items: flex-end;
            flex: 1;
        }

        .query-label {
            font-size: 1.1em;
            font-weight: 600;
            color: #1e293b;
        }

        .query-input {
            padding: 15px;
            border: 2px solid #e5e7eb;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 40px;
            max-height: 120px;
            width: 100%;
            max-width: 100%;
            transition: border-color 0.3s;
            overflow-y: auto;
            flex: 1;
        }

        .query-input:focus {
            outline: none;
            border-color: #4f46e5;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }

        .submit-btn {
            padding: 15px 30px;
            background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            flex-shrink: 0;
            height: 70px;
        }

        .submit-btn:hover:not(:disabled) {
            transform: translateY(-2px);
        }

        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .example-queries {
            background: #f0f9ff;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #bae6fd;
            flex: 0 0 450px;
            min-width: 300px;
        }

        .example-title {
            font-size: 0.95em;
            font-weight: 600;
            color: #0c4a6e;
            margin-bottom: 10px;
        }

        .example-item {
            background: white;
            padding: 10px 12px;
            margin: 6px 0;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #e0f2fe;
            font-size: 13px;
        }

        .example-item:hover {
            background: #f0f9ff;
            transform: translateX(4px);
        }

        /* Sources section styling within messages */
        .sources-section {
            background: #f0f9ff;
            border-radius: 15px;
            padding: 10px;
            margin-top: 5px;
            border: 1px solid #bae6fd;
        }

        .sources-title {
            font-size: 1.1em;
            font-weight: 600;
            color: #0c4a6e;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .sources-list {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        .source-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: white;
            border: 1px solid #e0f2fe;
            border-radius: 10px;
            transition: all 0.2s ease;
        }

        .source-item:hover {
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.15);
            transform: translateY(-1px);
        }

        .source-info {
            display: flex;
            align-items: center;
            flex: 1;
            min-width: 0;
        }

        .source-icon {
            margin-right: 12px;
            font-size: 18px;
        }

        .source-name {
            font-weight: 500;
            color: #1e293b;
            word-break: break-word;
            font-size: 14px;
        }

        .source-actions {
            display: flex;
            gap: 10px;
            flex-shrink: 0;
        }

        .download-btn, .view-btn {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            text-decoration: none;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s ease;
            background: white;
            border: 2px solid #4f46e5;
        }

        .download-btn {
            color: #4f46e5;
        }

        .download-btn:hover {
            background: white;
            text-decoration: none;
            color: #3730a3;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(79, 70, 229, 0.2);
        }

        .view-btn {
            color: #4f46e5;
        }

        .view-btn:hover {
            background: white;
            text-decoration: none;
            color: #3730a3;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(79, 70, 229, 0.2);
        }

        .no-download {
            color: #6b7280;
            font-style: italic;
            font-size: 13px;
        }

        .sources-count {
            background: #4f46e5;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }

        .loading-spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #4f46e5;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Mobile Responsiveness - Keep Side-by-Side */
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr 2fr;
            }

            .header {
                padding: 20px;
            }

            .header-logo {
                height: 60px;
            }

            .header-content h1 {
                font-size: 1.5em;
            }

            .header-content p {
                font-size: 0.9em;
            }

            .instruction-block, .info-block {
                margin: 15px;
                padding: 20px;
                font-size: 0.85em;
            }

            .query-section {
                padding: 15px;
                flex-direction: column;
                gap: 12px;
            }

            .example-queries {
                flex: 1 1 auto;
                min-width: 100%;
                padding: 12px;
            }

            .query-form {
                flex: 1 1 auto;
            }

            .query-input {
                min-height: 40px;
            }

            .submit-btn {
                height: 60px;
                padding: 12px 24px;
            }

            .example-item {
                font-size: 12px;
                padding: 8px 10px;
            }

            .message {
                gap: 10px;
            }

            .message-avatar {
                width: 32px;
                height: 32px;
                font-size: 14px;
            }

            .source-item {
                flex-direction: column;
                align-items: stretch;
                gap: 12px;
            }

            .source-actions {
                justify-content: center;
            }

            .download-btn, .view-btn {
                flex: 1;
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <!-- Guidelines Modal Overlay - Full Screen -->
    <div id="guidelinesOverlay" class="guidelines-overlay">
        <div class="guidelines-modal">
            <h2>Guidelines for Use</h2>
            <div class="guidelines-content">
                <p><strong>Before using this tool, please understand:</strong></p>
                <ol>
                    <li>This tool uses AI that may create fake information (hallucinate) or fabricate sources. <strong>Always verify responses carefully.</strong></li>
                    <li>You can ask questions in complete sentences or short phrases.</li>
                    <li>If you get "no relevant information found", try a more detailed query with additional keywords.</li>
                    <li>Clear the conversation history when switching topics or when conversations get long to maintain performance.</li>
                </ol>
            </div>
            <div class="agreement-checkbox">
                <input type="checkbox" id="guidelinesAgree">
                <label for="guidelinesAgree">I have read and agree to the guidelines</label>
            </div>
            <button id="proceedBtn" class="proceed-btn" disabled>Proceed to Tool</button>
        </div>
    </div>

    <!-- Two-Column Layout -->
    <div class="main-grid">
        <!-- Left Column: Info Side -->
        <div class="left-column">
            <div class="header">
                <img src="/static/ftsc_logo.png" alt="FTSC Logo" class="header-logo">
                <div class="header-content">
                    <h1>FTSC Paper Database Search Tool</h1>
                    <p>Search the FTSC paper database using retrieval-augmented generation</p>
                </div>
            </div>

            <div class="instruction-block">
                <p><strong>Welcome to the Flight Test Safety Committee's paper database RAG search tool!</strong> This tool uses <strong>Retrieval Augmented Generation</strong>, a form of large language model automation.</p>
                <p>You can use this tool to search the FTSC paper database for sources about a specific topic.</p>
                <p><strong>Guidelines for using the tool:</strong></p>
                <ol>
                    <li>You can ask in complete sentences ("What do I need to know about high-altitude flight test?") or short phrases ("high-altitude flight testing")</li>
                    <li>The LLM may create fake information (hallucinate), or even fabricate entire sources. <strong>Carefully check all responses.</strong></li>
                    <li>If the model returns "no relevant information found in the available sources", try submitting a slightly longer, more detailed prompt with more keywords.</li>
                    <li>LLM performance tends to degrade as a conversation gets longer. <strong> We strongly recommend clearing the conversation history using the "Clear Chat" button when you switch topics, or when the conversation starts to get long.</strong> </li>
                </ol>
            </div>

            <div class="info-block">
                <h2>How does this tool work?</h2>

                <p>
                    This tool uses a technique called <strong>Retrieval-Augmented Generation</strong>,
                    which uses a large language model (LLM) connected to a database of information.
                </p>

                <p>
                    Retrieval-Augmented Generation (RAG) is a technique to fine-tune an LLM to a specific database or
                    use case without the need for retraining, which would be expensive and infeasible for small-scale use.
                </p>

                <p>
                    When you ask a question, the RAG tool first generates a general answer and then backs it up with information retrieved directly from the database.
                </p>

                <p><strong>RAG does the following:</strong></p>

                <p>
                    <strong>1. Database Creation:</strong> The files in the database are broken into
                    text "chunks" which are converted into numerical vectors ("embeddings") that encode their semantic meaning.
                    From this point on, the RAG tool only uses this vector database of chunk embeddings, and does not have access to the
                    raw files (e.g. PDFs) in the original database.
                </p>

                <p>
                    <strong>2. Document Retrieval:</strong> When you ask a query, it is converted into a numerical vector embedding
                    in the same way as the database files. The retrieval system then searches through the embeddings of all
                    corpus documents to find the document chunks that are most similar to the query (using vector euclidean distance or another metric).
                    These correspond to the most relevant documents.
                </p>

                <p>
                    <strong>3. Context Assembly:</strong> The most relevant document chunks are retrieved and combined
                    with your original question and conversation history to create a comprehensive context.
                </p>

                <p>
                    <strong>4. Response Generation:</strong> A large language model (in our case, a lightweight variant
                    of Gemini) uses the combined context from step 3 to generate a response to your query.
                </p>

                <p>
                    Because the context from step 3 only contains the most relevant content from the database,
                    the LLM's response is tailored to your query and is less likely to be distracted by irrelevant content.
                </p>

                <p>
                    <strong>Multi-message conversations:</strong> This tool now maintains conversation history. Each new message
                    includes the previous conversation context, allowing for follow-up questions and coherent multi-turn discussions.
                </p>

                <p>
                    Like other LLM-based chat tools (e.g. ChatGPT, Claude, Gemini), this tool uses a system prompt
                    which your query is appended to. This prompt shapes the model's behavior, tone, and things it is
                    allowed and not allowed to say in response to your query.
                </p>
            </div>
        </div>

        <!-- Right Column: Conversation Side -->
        <div class="right-column">
            <div class="conversation-wrapper">
                <div id="conversationSection" class="conversation-section" style="display: none;">
                    <div class="conversation-header">
                        <div class="conversation-title">💬 Conversation</div>
                        <button onclick="clearConversation()" class="clear-chat-btn">Clear Chat</button>
                    </div>
                    <div id="conversationMessages" class="conversation-messages">
                        <!-- Messages will be added here dynamically -->
                    </div>
                </div>
            </div>

            <div class="query-section">
                <div class="example-queries">
                    <div class="example-title">💡 Example Queries:</div>
                    <div class="example-item" onclick="setQuery('What do I need to know about high-altitude flight test?')">
                        What do I need to know about high-altitude flight test?
                    </div>
                    <div class="example-item" onclick="setQuery('What does the database say about safety mishap accountability?')">
                        What does the database say about safety mishap accountability?
                    </div>
                    <div class="example-item" onclick="setQuery('Aircraft certification requirements')">
                        Aircraft certification requirements
                    </div>
                    <div class="example-item" onclick="setQuery('What are the safety considerations for this?')">
                        (Follow up) What are the safety considerations for this?
                    </div>
                </div>

                <form id="queryForm" class="query-form">
                    <textarea id="query" class="query-input" placeholder="Enter your question here, then press enter"></textarea>
                    <button type="submit" id="submitButton" class="submit-btn">Send</button>
                </form>
            </div>
        </div>
    </div>

    <script>
        // CSRF Token
        const CSRF_TOKEN = '<?php echo get_csrf_token(); ?>';

        // Initialize markdown parser
        const md = window.markdownit({
            html: false,        // Disable HTML tags for security
            breaks: true,       // Convert '\n' to <br>
            linkify: true,      // Autoconvert URL-like text to links
            typographer: true   // Enable smart quotes and other typographic replacements
        });

        // Conversation history storage
        let conversationHistory = [];

        // Guidelines agreement handling
        document.addEventListener('DOMContentLoaded', function() {
            const overlay = document.getElementById('guidelinesOverlay');
            const checkbox = document.getElementById('guidelinesAgree');
            const proceedBtn = document.getElementById('proceedBtn');

            // Check if user has already agreed
            const hasAgreed = localStorage.getItem('ftsc_guidelines_agreed');
            if (hasAgreed === 'true') {
                overlay.style.display = 'none';
            }

            // Enable/disable proceed button based on checkbox
            checkbox.addEventListener('change', function() {
                proceedBtn.disabled = !this.checked;
            });

            // Handle proceed button
            proceedBtn.addEventListener('click', function() {
                if (checkbox.checked) {
                    localStorage.setItem('ftsc_guidelines_agreed', 'true');
                    overlay.style.display = 'none';
                    checkStatus();
                }
            });
        });

        // Check status on page load
        window.onload = function() {
            // Only check status if guidelines have been agreed to
            const hasAgreed = localStorage.getItem('ftsc_guidelines_agreed');
            if (hasAgreed === 'true') {
                checkStatus();
            } else {
                // If guidelines not agreed, ensure submit button is disabled
                document.getElementById('submitButton').disabled = true;
            }
            loadConversationHistory();
        };

        // Load conversation history from localStorage
        function loadConversationHistory() {
            try {
                const saved = localStorage.getItem('ftsc_conversation_history');
                if (saved) {
                    conversationHistory = JSON.parse(saved);
                    displayConversationHistory();
                }
            } catch (e) {
                console.log('No previous conversation history found');
                conversationHistory = [];
            }
        }

        // Save conversation history to localStorage
        function saveConversationHistory() {
            try {
                localStorage.setItem('ftsc_conversation_history', JSON.stringify(conversationHistory));
            } catch (e) {
                console.log('Could not save conversation history');
            }
        }

        // Display conversation history
        function displayConversationHistory() {
            const conversationSection = document.getElementById('conversationSection');
            const messagesContainer = document.getElementById('conversationMessages');

            if (conversationHistory.length === 0) {
                conversationSection.style.display = 'none';
                return;
            }

            conversationSection.style.display = 'flex';
            messagesContainer.innerHTML = '';

            conversationHistory.forEach((item, index) => {
                // Add user message
                const userMessage = document.createElement('div');
                userMessage.className = 'message';
                userMessage.innerHTML = `
                    <div class="message-avatar user-avatar">U</div>
                    <div class="message-content">${escapeHtml(item.query)}</div>
                `;
                messagesContainer.appendChild(userMessage);

                // Add assistant message with sources
                const assistantMessage = document.createElement('div');
                assistantMessage.className = 'message';

                let assistantContent = `
                    <div class="message-avatar assistant-avatar">A</div>
                    <div class="message-content">${md.render(item.response)}`;

                // Add sources inline if available
                if (item.sources && item.sources.length > 0) {
                    assistantContent += displaySources(item.sources);
                }

                assistantContent += `</div>`;
                assistantMessage.innerHTML = assistantContent;
                messagesContainer.appendChild(assistantMessage);
            });

            // Scroll to bottom
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        // Clear conversation history
        function clearConversation() {
            if (confirm('Are you sure you want to clear the conversation history?')) {
                conversationHistory = [];
                localStorage.removeItem('ftsc_conversation_history');
                saveConversationHistory();

                // Clear the messages container completely
                const messagesContainer = document.getElementById('conversationMessages');
                messagesContainer.innerHTML = '';

                displayConversationHistory();
            }
        }

        // Add message to conversation history
        function addToConversationHistory(query, response, sources = []) {
            conversationHistory.push({
                query: query,
                response: response,
                sources: sources,
                timestamp: new Date().toISOString()
            });
            saveConversationHistory();
            displayConversationHistory();
        }

        // Get conversation context for API
        function getConversationContext() {
            return conversationHistory.map(item => ({
                query: item.query,
                response: item.response
            }));
        }

        // Escape HTML for safety
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // API call helper
        function apiCall(endpoint, method = 'GET', data = null) {
            const url = window.location.pathname + '?api=' + endpoint;

            const options = {
                method: method,
                headers: {
                    'Content-Type': 'application/json'
                }
            };

            if (method === 'POST' && data !== null) {
                // Add CSRF token
                data.csrf_token = CSRF_TOKEN;
                options.body = JSON.stringify(data);
            }

            return fetch(url, options);
        }

        function checkStatus() {
            apiCall('status', 'GET')
                .then(response => response.json())
                .then(data => {
                    var submitButton = document.getElementById('submitButton');

                    if (data.initialized) {
                        submitButton.disabled = false;
                    } else {
                        submitButton.disabled = true;
                    }
                })
                .catch(error => {
                    console.error('Error checking status');
                });
        }

        // Function to display sources
        function displaySources(sources) {
            if (!sources || sources.length === 0) {
                return '';
            }

            var sourcesHtml = '<div class="sources-section">';
            sourcesHtml += '<div class="sources-title">🔗 Sources <span class="sources-count">' + sources.length + '</span></div>';
            sourcesHtml += '<div class="sources-list">';

            sources.forEach(function(source) {
                var downloadLink = source.download_link || source.view_link;
                var fileName = source.name;

                sourcesHtml += '<div class="source-item">';
                sourcesHtml += '<div class="source-info">';
                sourcesHtml += '<span class="source-icon">📄</span>';
                sourcesHtml += '<span class="source-name">' + fileName + '</span>';
                sourcesHtml += '</div>';
                sourcesHtml += '<div class="source-actions">';

                if (downloadLink) {
                    sourcesHtml += '<a href="' + downloadLink + '" class="download-btn" title="Download ' + fileName + '" target="_blank" rel="noopener noreferrer">';
                    sourcesHtml += '<span>⬇️</span> Download</a>';
                } else {
                    sourcesHtml += '<span class="no-download">Download not available</span>';
                }

                if (source.view_link) {
                    sourcesHtml += '<a href="' + source.view_link + '" class="view-btn" title="View ' + fileName + '" target="_blank" rel="noopener noreferrer">';
                    sourcesHtml += '<span>👁️</span> View</a>';
                }

                sourcesHtml += '</div>';
                sourcesHtml += '</div>';
            });

            sourcesHtml += '</div></div>';
            return sourcesHtml;
        }

        function setQuery(query) {
            document.getElementById('query').value = query;
            document.getElementById('query').focus();
        }

        // Query form submission with conversation history support
        document.getElementById('queryForm').onsubmit = function(e) {
            e.preventDefault();

            var query = document.getElementById('query').value;
            if (!query.trim()) {
                alert('Please enter a question');
                return;
            }

            // Show loading state
            var submitButton = document.getElementById('submitButton');
            submitButton.disabled = true;
            submitButton.innerHTML = '<div class="loading-spinner"></div>Processing...';

            // Add user message to conversation immediately
            addMessageToConversation('user', query);

            // Add loading message for assistant
            var loadingId = addMessageToConversation('assistant', '<div class="loading-spinner"></div> Processing your query...');

            apiCall('query', 'POST', {
                query: query,
                conversation_history: getConversationContext()
            })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        updateMessageInConversation(loadingId, 'Error: ' + data.error, []);
                    } else if (data.response) {
                        // Update the loading message with the actual response
                        updateMessageInConversation(loadingId, data.response, data.sources || []);

                        // Add to persistent conversation history
                        addToConversationHistory(query, data.response, data.sources || []);

                        // Clear the input
                        document.getElementById('query').value = '';
                    }

                    // Reset button
                    submitButton.disabled = false;
                    submitButton.innerHTML = 'Send';
                })
                .catch(error => {
                    updateMessageInConversation(loadingId, 'Network error occurred', []);

                    // Reset button
                    submitButton.disabled = false;
                    submitButton.innerHTML = 'Send';
                });
        };

        // Function to add a message to the conversation display
        function addMessageToConversation(type, content) {
            const conversationSection = document.getElementById('conversationSection');
            const messagesContainer = document.getElementById('conversationMessages');

            // Show conversation section if hidden
            conversationSection.style.display = 'flex';

            const message = document.createElement('div');
            message.className = 'message';
            const messageId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
            message.id = messageId;

            if (type === 'user') {
                message.innerHTML = `
                    <div class="message-avatar user-avatar">U</div>
                    <div class="message-content">${escapeHtml(content)}</div>
                `;
            } else {
                message.innerHTML = `
                    <div class="message-avatar assistant-avatar">A</div>
                    <div class="message-content">${content}</div>
                `;
            }

            messagesContainer.appendChild(message);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;

            return messageId;
        }

        // Function to update a message in the conversation display
        function updateMessageInConversation(messageId, content, sources = []) {
            const message = document.getElementById(messageId);
            if (message) {
                const messageContent = message.querySelector('.message-content');
                let htmlContent = md.render(content);

                // Add sources if available
                if (sources && sources.length > 0) {
                    htmlContent += displaySources(sources);
                }

                messageContent.innerHTML = htmlContent;

                // Scroll to bottom
                const messagesContainer = document.getElementById('conversationMessages');
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
        }

        // Add keyboard shortcut for form submission
        document.getElementById('query').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                document.getElementById('queryForm').dispatchEvent(new Event('submit'));
            }
        });
    </script>
</body>
</html>
