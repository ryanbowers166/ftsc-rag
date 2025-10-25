# Local Testing Guide

This guide will help you test the PHP integration locally before giving it to your webmaster.

---

## Prerequisites

You need PHP installed on your local machine. Here's how to check and install:

### Check if PHP is installed

```bash
php -v
```

If you see a version number (7.4 or higher), you're good to go!

### Install PHP (if needed)

**Windows:**
1. Download PHP from: https://windows.php.net/download/
2. Or install XAMPP: https://www.apachefriends.org/

**macOS:**
```bash
# PHP comes pre-installed on macOS, but you can upgrade with Homebrew
brew install php
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install php php-curl
```

---

## Quick Start: Test with Built-in PHP Server

### Step 1: Configure Your Cloud Run URL

First, you need to get your Cloud Run URL. If you've already deployed:

```bash
gcloud run services describe ftsc-rag-search --region us-central1 --format 'value(status.url)'
```

Copy the URL (it will look like: `https://ftsc-rag-search-abc123xyz.a.run.app`)

### Step 2: Edit config.php

1. Open `config.php` in a text editor
2. Find line 17:
   ```php
   define('CLOUD_RUN_URL', 'YOUR_CLOUD_RUN_URL');
   ```
3. Replace with your actual URL:
   ```php
   define('CLOUD_RUN_URL', 'https://ftsc-rag-search-abc123xyz.a.run.app');
   ```
4. Enable debug mode for testing (line 48):
   ```php
   define('DEBUG_MODE', true);  // Change false to true
   ```
5. Save the file

### Step 3: Create the Static Directory

The logo needs to be in a `static/` directory:

```bash
# In your project directory
mkdir static

# Copy your logo (if you have it)
# If not, we'll handle the missing logo gracefully
```

### Step 4: Start the PHP Development Server

```bash
# Navigate to your project directory
cd C:\Users\Test\PycharmProjects\ftsc-rag-demo

# Start PHP's built-in web server
php -S localhost:8000
```

You should see:
```
PHP 8.x Development Server (http://localhost:8000) started
```

### Step 5: Test in Your Browser

1. Open your browser
2. Go to: http://localhost:8000/index.php
3. You should see the guidelines modal
4. Accept the guidelines
5. Try a test query!

---

## Testing Checklist

### Basic Functionality
- [ ] Page loads without errors
- [ ] Guidelines modal appears
- [ ] Can accept guidelines and proceed
- [ ] Query input box is visible
- [ ] Can type a query
- [ ] Submit button is enabled after accepting guidelines

### API Connection
- [ ] Submit a test query (e.g., "flight test procedures")
- [ ] Loading spinner appears
- [ ] Response appears in chat area
- [ ] Sources are displayed (if available)
- [ ] No error messages

### Rate Limiting
- [ ] Submit 21 queries rapidly (more than the 20/minute limit)
- [ ] Should get "Rate limit exceeded" error on 21st query
- [ ] Wait 60 seconds, should work again

### Conversation History
- [ ] Submit a query
- [ ] Submit a follow-up query
- [ ] Both messages appear in conversation area
- [ ] Click "Clear Chat" button
- [ ] Conversation clears

### Security (CSRF)
- [ ] Open browser console (F12)
- [ ] Try to manually make API call without CSRF token
- [ ] Should get "Invalid security token" error

---

## Common Issues & Solutions

### "Configuration error. Please contact the administrator."

**Problem:** Cloud Run URL not configured or DEBUG_MODE is false

**Solution:**
1. Edit `config.php`
2. Set `DEBUG_MODE` to `true` (line 48)
3. Configure `CLOUD_RUN_URL` with your actual URL (line 17)
4. Restart PHP server

### "Call to undefined function curl_init"

**Problem:** cURL extension not enabled

**Solution:**
```bash
# Check installed extensions
php -m | grep curl

# If not listed, you need to enable it
# Edit php.ini and uncomment:
extension=curl

# Or on Linux:
sudo apt install php-curl
```

### Logo not showing (broken image)

**Problem:** Logo file missing or wrong path

**Solution:**
1. Create `static/` directory if it doesn't exist
2. Add any PNG file as `ftsc_logo.png` for testing
3. Or ignore for now - the app will still work

### "Service temporarily unavailable"

**Problem:** Cannot connect to Cloud Run

**Possible causes:**
1. Cloud Run URL is wrong or service not deployed
2. Cloud Run service is down
3. No internet connection

**Solution:**
1. Verify Cloud Run URL is correct
2. Test Cloud Run directly:
   ```bash
   curl https://your-cloud-run-url.a.run.app/health
   ```
3. Check your internet connection

### Port 8000 already in use

**Problem:** Another service is using port 8000

**Solution:** Use a different port:
```bash
php -S localhost:8080  # Try port 8080 instead
# Then visit: http://localhost:8080/index.php
```

### Session errors

**Problem:** Cannot write session files

**Solution:** PHP needs a writable temp directory
```bash
# Check where sessions are stored
php -i | grep session.save_path

# Make sure that directory is writable
```

---

## Advanced Testing

### Test API Endpoints Directly

You can test individual endpoints using curl:

```bash
# Test status endpoint
curl "http://localhost:8000/index.php?api=status"

# Test query endpoint (with CSRF token - get from browser console)
curl -X POST "http://localhost:8000/index.php?api=query" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "csrf_token": "your-token-here"}'
```

### Test Rate Limiting

Create a simple test script `test-rate-limit.php`:

```php
<?php
// Test rate limiting by making rapid requests

for ($i = 1; $i <= 25; $i++) {
    $ch = curl_init('http://localhost:8000/index.php?api=status');
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

    $response = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    echo "Request $i: HTTP $httpCode\n";

    if ($httpCode == 429) {
        echo "Rate limit triggered at request $i\n";
        break;
    }
}
?>
```

Run it:
```bash
php test-rate-limit.php
```

### View Detailed Errors

With `DEBUG_MODE` set to `true`, you'll see detailed error messages.

To see PHP errors in real-time:
```bash
# Start server with error display
php -S localhost:8000 -d display_errors=1 -d error_reporting=E_ALL
```

---

## Testing Without Cloud Run

If you don't have Cloud Run deployed yet, you can test the frontend:

1. The page will load fine
2. You'll get connection errors when trying to submit queries
3. This is expected - it proves the PHP proxy is working
4. You can still test:
   - Page layout
   - Guidelines modal
   - Input validation
   - Rate limiting (sort of)
   - Chat interface

---

## When Testing is Complete

### Before giving to webmaster:

1. **Disable debug mode:**
   ```php
   // In config.php, line 48:
   define('DEBUG_MODE', false);  // Change back to false
   ```

2. **Reset test data:**
   - Clear browser localStorage
   - Delete test rate limit scripts
   - Remove any test files

3. **Verify files:**
   ```
   ✓ index.php
   ✓ config.php (with DEBUG_MODE = false)
   ✓ WEBMASTER-GUIDE.md
   ✓ .htaccess.example (optional)
   ```

4. **Provide to webmaster:**
   - The files above
   - Your Cloud Run URL
   - Any specific customization requirements

---

## Quick Reference

| Item | Value |
|------|-------|
| **Test URL** | http://localhost:8000/index.php |
| **Start server** | `php -S localhost:8000` |
| **Stop server** | Press Ctrl+C |
| **Enable debug** | Set `DEBUG_MODE` to `true` in config.php |
| **View logs** | Check terminal where server is running |
| **Rate limit** | 20 requests/minute (configurable in config.php) |

---

## Troubleshooting Resources

- **PHP Manual:** https://www.php.net/manual/en/
- **cURL PHP:** https://www.php.net/manual/en/book.curl.php
- **Sessions:** https://www.php.net/manual/en/book.session.php

---

**Happy Testing!** If everything works locally, it should work on your production server too.
