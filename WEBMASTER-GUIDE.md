# FTSC RAG Search Tool - Webmaster Integration Guide

## Quick Start for Webmasters

This guide will help you integrate the FTSC RAG Search Tool into your existing website.

---

## What You're Integrating

A search tool that uses AI to search through FTSC's paper database. It includes:
- Full chat interface with conversation history
- Source citations with download links
- Mobile-responsive design
- Rate limiting and security features

---

## Files You Need

You'll receive these files:

1. **index.php** - Main application file (standalone page)
2. **config.php** - Configuration file (you'll edit this)
3. **static/ftsc_logo.png** - FTSC logo (you may already have this)

---

## Installation Steps

### Step 1: Upload Files

Upload to your web server:
```
yourwebsite.com/
├── rag-search/              (create this directory)
│   ├── index.php
│   ├── config.php
│   └── static/
│       └── ftsc_logo.png
```

**Note:** You can name the directory anything you want (e.g., `search`, `rag`, `database-search`)

### Step 2: Configure Cloud Run URL

1. Open `config.php` in a text editor

2. Find line 17 that says:
   ```php
   define('CLOUD_RUN_URL', 'YOUR_CLOUD_RUN_URL');
   ```

3. Replace `YOUR_CLOUD_RUN_URL` with the actual URL provided by your developer

   Example:
   ```php
   define('CLOUD_RUN_URL', 'https://ftsc-rag-search-abc123xyz.a.run.app');
   ```

   **Important:**
   - Include `https://`
   - Do NOT include a trailing slash
   - The URL will end in `.run.app` or `.a.run.app`

4. Save the file

### Step 3: Verify PHP Requirements

Your server needs:
- **PHP 7.4 or higher** (most modern hosting has this)
- **cURL extension** (usually enabled by default)
- **Session support** (usually enabled by default)

To check, create a file called `phpinfo.php`:
```php
<?php phpinfo(); ?>
```

Upload it to your server and visit it in a browser. Look for:
- PHP Version 7.4+
- Search for "curl" - should show as enabled
- Search for "session" - should show as enabled

**Delete `phpinfo.php` after checking** (security risk)

### Step 4: Set File Permissions

If you have SSH access:
```bash
chmod 644 index.php config.php
chmod 755 static/
chmod 644 static/ftsc_logo.png
```

If using FTP, set permissions to:
- Files (index.php, config.php): 644
- Directories (static/): 755

### Step 5: Test

1. Navigate to: `https://yourwebsite.com/rag-search/index.php`
2. You should see the guidelines modal
3. Accept the guidelines
4. Try a test query: "What are flight test procedures?"
5. Verify you get a response

---

## Configuration Options

All configuration is in `config.php`. Here are the main settings:

### Cloud Run URL (Required)
```php
define('CLOUD_RUN_URL', 'https://your-service.a.run.app');
```

### Rate Limiting
```php
define('RATE_LIMIT_MAX_QUERIES', 20);  // Max queries per IP per minute
define('RATE_LIMIT_WINDOW', 60);       // Time window in seconds
```

### Query Limits
```php
define('MAX_QUERY_LENGTH', 2000);      // Max characters in a query
define('MAX_CONVERSATION_HISTORY', 10); // Max messages to remember
```

### Security
```php
define('CSRF_PROTECTION', true);       // CSRF token protection
define('SESSION_TIMEOUT', 3600);       // Session timeout (1 hour)
```

### Debugging
```php
define('DEBUG_MODE', false);           // Set to true for testing ONLY
```

**Warning:** Never leave `DEBUG_MODE` set to `true` in production - it exposes error details.

---

## Troubleshooting

### "Configuration error" message

**Problem:** Cloud Run URL not configured

**Solution:** Edit `config.php` and set `CLOUD_RUN_URL` to the correct value

### "Service temporarily unavailable"

**Problem:** Cannot connect to Cloud Run service

**Possible causes:**
1. Wrong Cloud Run URL in config.php
2. Cloud Run service is down
3. cURL extension not enabled
4. Firewall blocking outbound connections

**Solution:**
1. Verify Cloud Run URL is correct
2. Ask your developer to check Cloud Run status
3. Contact your hosting provider to verify cURL is enabled

### Rate limit errors

**Problem:** Users getting "Rate limit exceeded" message

**Solution:** Increase rate limits in `config.php`:
```php
define('RATE_LIMIT_MAX_QUERIES', 50);  // Increase from 20 to 50
```

### Sessions not working

**Problem:** Rate limiting not working or users getting logged out

**Solution:**
1. Verify session support is enabled in PHP
2. Check that your server's session directory is writable
3. Contact hosting provider if needed

### Logo not showing

**Problem:** Broken image icon where logo should be

**Solution:**
1. Verify logo file exists at: `static/ftsc_logo.png`
2. Check file permissions (should be 644)
3. Verify image file is valid PNG format
4. Check the path in index.php matches your directory structure

---

## Customization

### Changing Colors

The tool uses these main colors (search for these in `index.php`):
- Header background: `#13AFFC` (blue)
- Dark gray background: `#374151`
- Primary button: `#1e40af` (dark blue)

To change colors, search in `index.php` for these hex codes and replace them.

### Changing Text

All text is in `index.php`. Search for the text you want to change and edit directly.

For example, to change the title:
```html
<!-- Find this line: -->
<h1>FTSC Paper Database Search Tool</h1>

<!-- Change to: -->
<h1>Your Custom Title Here</h1>
```

### Adding to Site Navigation

Add a link to your main navigation:
```html
<a href="/rag-search/index.php">Database Search</a>
```

---

## Security Best Practices

1. **Always use HTTPS** - Never run this on HTTP-only sites
2. **Keep DEBUG_MODE off** in production
3. **Monitor rate limits** - Adjust if needed
4. **Regular backups** - Include these files in your backups
5. **Keep PHP updated** - Use the latest stable version your host supports

---

## Support

If you encounter issues:

1. **Check Cloud Run status** - Contact your developer
2. **Verify PHP requirements** - Use phpinfo.php
3. **Check file permissions** - See Step 4 above
4. **Enable DEBUG_MODE temporarily** - See detailed errors (remember to disable after)
5. **Contact hosting provider** - For server-related issues

---

## Production Checklist

Before going live:

- [ ] Cloud Run URL configured in config.php
- [ ] DEBUG_MODE set to false
- [ ] Logo uploaded to /static/ftsc_logo.png
- [ ] File permissions set correctly (644 for files, 755 for directories)
- [ ] PHP version 7.4+ verified
- [ ] cURL extension enabled
- [ ] Session support enabled
- [ ] HTTPS enabled on website
- [ ] Test query completed successfully
- [ ] Rate limiting tested
- [ ] Mobile responsiveness verified
- [ ] Added to site navigation (optional)

---

## Performance Notes

- **First query** may take 5-8 seconds (Cloud Run "cold start")
- **Subsequent queries** are instant
- **Conversation history** stored in browser (localStorage)
- **Rate limiting** prevents abuse and protects backend costs
- **No database required** on your server - everything proxies to Cloud Run

---

## Quick Reference

| Item | Value |
|------|-------|
| **Main file** | index.php |
| **Config file** | config.php |
| **Logo path** | static/ftsc_logo.png |
| **Default rate limit** | 20 queries/minute |
| **Session timeout** | 1 hour |
| **Max query length** | 2000 characters |
| **PHP requirement** | 7.4+ |

---

**Questions?** Contact your developer or refer to the full documentation in README-DEPLOY.md
