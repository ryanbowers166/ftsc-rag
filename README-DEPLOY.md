# Deployment Guide - FTSC RAG Search

## Understanding Your Setup

**IMPORTANT:** You do NOT need to keep your server running constantly!

- Your **RAG corpus/database is stored on Ragie's servers** (not yours)
- Your Flask app just makes API calls to Ragie
- When Cloud Run "scales to zero" (shuts down), your corpus stays persistent on Ragie
- Next request just re-authenticates (takes milliseconds)
- **You only pay for actual request processing time**

---

## Cost-Optimized Cloud Run Deployment

### What You'll Pay

With the configuration in this repo:

- **When idle (no users):** $0.00/month
- **Per request:** ~$0.0001-0.0005 (fractions of a cent)
- **1,000 queries/month:** ~$0.10-$2.00
- **10,000 queries/month:** ~$1-$10
- **First visit after idle:** 5-8 second cold start (then instant)

### Prerequisites

1. **Google Cloud Account**
   - Create at: https://cloud.google.com
   - Enable billing (but you'll stay in free tier with low usage)

2. **Install Google Cloud CLI**
   ```bash
   # macOS
   brew install google-cloud-sdk

   # Windows
   # Download from: https://cloud.google.com/sdk/docs/install

   # Linux
   curl https://sdk.cloud.google.com | bash
   ```

3. **Ragie API Key**
   - Get from your Ragie dashboard
   - This is where your RAG corpus actually lives

---

## Quick Deploy (Recommended)

### Step 1: Authenticate with Google Cloud

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### Step 2: Set Your Ragie API Key

```bash
export RAGIE_API_KEY="your_ragie_api_key_here"
```

### Step 3: Deploy

```bash
./deploy.sh
```

That's it! The script will:
- Build your Docker container
- Deploy to Cloud Run with cost-optimized settings
- Configure scale-to-zero (min-instances=0)
- Set up 512MB memory, 60s timeout
- Enable CPU throttling for lower costs

---

## Manual Deployment

If you prefer to deploy manually:

```bash
gcloud run deploy ftsc-rag-search \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 3 \
  --memory 512Mi \
  --cpu 1 \
  --timeout 60s \
  --concurrency 80 \
  --cpu-throttling \
  --no-cpu-boost \
  --execution-environment gen2 \
  --set-env-vars RAGIE_API_KEY=your_key_here
```

---

## Setting Environment Variables After Deployment

If you didn't set RAGIE_API_KEY during deployment:

```bash
gcloud run services update ftsc-rag-search \
  --update-env-vars RAGIE_API_KEY=your_actual_key \
  --region us-central1
```

---

## Testing Your Deployment

### 1. Test Basic Connectivity

```bash
curl https://your-service-url.run.app/test
```

Expected response:
```json
{
  "message": "Flask server is working!",
  "initialized": true,
  "healthy": true
}
```

### 2. Test RAG Query

```bash
curl -X POST https://your-service-url.run.app/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are flight test procedures?"}'
```

### 3. View Logs

```bash
gcloud run services logs read ftsc-rag-search --region us-central1 --limit 50
```

---

## Integrating with Your PHP Website

Since your website is PHP and hosted elsewhere, you'll embed this as an external API:

### Option 1: Simple AJAX Call

Add to your PHP page:

```html
<script>
async function searchRAG(query) {
  const response = await fetch('https://your-service-url.run.app/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: query })
  });
  return await response.json();
}

// Usage
searchRAG('flight test safety').then(result => {
  console.log(result.response);
  console.log(result.sources);
});
</script>
```

### Option 2: Server-Side PHP Call

```php
<?php
function queryRAG($query) {
    $url = 'https://your-service-url.run.app/query';
    $data = array('query' => $query);

    $options = array(
        'http' => array(
            'method'  => 'POST',
            'content' => json_encode($data),
            'header'  => 'Content-Type: application/json'
        )
    );

    $context = stream_context_create($options);
    $result = file_get_contents($url, false, $context);
    return json_decode($result, true);
}

// Usage
$result = queryRAG('high altitude flight test');
echo $result['response'];
?>
```

### Option 3: Embed in iFrame

```html
<iframe
  src="https://your-service-url.run.app"
  width="100%"
  height="600px"
  frameborder="0">
</iframe>
```

---

## Cost Optimization Tips

### Current Configuration (Already Optimized)

 **min-instances=0** - Scales to zero when idle
 **memory=512Mi** - Minimal for your app
 **cpu-throttling** - Slower but cheaper billing
 **concurrency=80** - Serves multiple users per instance
 **timeout=60s** - Adequate for RAG queries

### Monitor Your Costs

```bash
# View current month's costs
gcloud billing accounts list

# View service-specific usage
gcloud run services describe ftsc-rag-search --region us-central1 --format json
```

### If You Need Faster Response (Costs More)

To eliminate cold starts, keep 1 instance warm:

```bash
gcloud run services update ftsc-rag-search \
  --min-instances 1 \
  --region us-central1
```

**Warning:** This will cost ~$10-30/month even with zero traffic.

---

## Automated CI/CD Deployment (Optional)

### Using Cloud Build with GitHub

1. **Connect Repository to Cloud Build:**
   - Go to: https://console.cloud.google.com/cloud-build/triggers
   - Click "Connect Repository"
   - Select GitHub, authorize, choose your repo

2. **Create Build Trigger:**
   - Click "Create Trigger"
   - Name: "Deploy on Push to Main"
   - Event: Push to branch
   - Branch: `^main$`
   - Configuration: Cloud Build configuration file
   - Location: `/cloudbuild.yaml`

3. **Set RAGIE_API_KEY in Secret Manager:**
   ```bash
   echo -n "your_ragie_api_key" | gcloud secrets create RAGIE_API_KEY --data-file=-
   ```

4. **Update cloudbuild.yaml to use secret:**
   Add to the deploy step:
   ```yaml
   - '--set-secrets=RAGIE_API_KEY=RAGIE_API_KEY:latest'
   ```

Now every push to `main` automatically deploys!

---

## Troubleshooting

### Cold Starts Too Slow?

If 5-8 seconds is too slow, you have options:

1. **Keep 1 instance warm** (costs ~$10-30/month):
   ```bash
   gcloud run services update ftsc-rag-search --min-instances 1 --region us-central1
   ```

2. **Use Cloud Scheduler to ping every 5 minutes** (cheaper workaround):
   ```bash
   gcloud scheduler jobs create http keep-warm \
     --schedule="*/5 * * * *" \
     --uri="https://your-service-url.run.app/health" \
     --http-method=GET
   ```
   Note: This costs ~$0.10/month for the scheduler

### Service Won't Start?

Check logs:
```bash
gcloud run services logs read ftsc-rag-search --region us-central1 --limit 100
```

Common issues:
- Missing RAGIE_API_KEY
- Google Drive permissions not configured
- Port mismatch (ensure your app uses PORT env var)

### High Costs Unexpectedly?

Check if you have:
- Multiple instances running (`min-instances` > 0)
- Health checks pinging too frequently
- Excessive logging
- Memory leaks keeping instances alive

View current configuration:
```bash
gcloud run services describe ftsc-rag-search --region us-central1
```

---

## Updating Your Deployment

### Quick Update

Just run deploy.sh again:
```bash
./deploy.sh
```

### Update Only Environment Variables

```bash
gcloud run services update ftsc-rag-search \
  --update-env-vars NEW_VAR=value \
  --region us-central1
```

### Rollback to Previous Version

```bash
gcloud run services update-traffic ftsc-rag-search \
  --to-revisions=PREVIOUS_REVISION=100 \
  --region us-central1
```

---

## Security Considerations

### Current Status: Public Access

Your service is deployed with `--allow-unauthenticated`, meaning anyone can access it.

### To Restrict Access

#### Option 1: Require Authentication

```bash
gcloud run services update ftsc-rag-search \
  --no-allow-unauthenticated \
  --region us-central1
```

Then call with auth token:
```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  https://your-service-url.run.app/query
```

#### Option 2: Use API Keys

Add API key validation in your Flask app:

```python
@app.before_request
def validate_api_key():
    api_key = request.headers.get('X-API-Key')
    if api_key != os.getenv('EXPECTED_API_KEY'):
        return jsonify({'error': 'Unauthorized'}), 401
```

#### Option 3: IP Allowlisting (via Load Balancer)

For advanced setups, use Cloud Armor with a load balancer.

---

## Support & Resources

- **Cloud Run Docs:** https://cloud.google.com/run/docs
- **Pricing Calculator:** https://cloud.google.com/products/calculator
- **Ragie Docs:** https://docs.ragie.ai
- **This Repo Issues:** [Your GitHub repo URL]

---

## Summary: Why This is Cost-Effective

| Component | Cost When Idle | Why |
|-----------|----------------|-----|
| **Ragie (RAG Database)** | Ragie's pricing | Your corpus lives here, not on your server |
| **Cloud Run** | $0.00 | Scales to zero (min-instances=0) |
| **Container Registry** | ~$0.01/month | Just stores your Docker image |
| **Per Request** | ~$0.0001 | Only charged for actual processing time |

**Total for 1,000 queries/month:** ~$0.10-$2.00 + Ragie costs

Compare to:
- **Always-on VM:** $10-50/month minimum
- **App Engine Standard (min 1 instance):** $20-40/month
- **Your previous setup:** (Whatever you were paying)

---

**You're all set! Deploy with `./deploy.sh` and enjoy near-zero idle costs.**
