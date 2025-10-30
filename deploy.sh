#!/bin/bash

# ============================================
# Cloud Run Deployment Script
# Ultra-Low-Cost Configuration (Scale to Zero)
# ============================================

# Configuration
SERVICE_NAME="ftsc-rag-search"
REGION="us-central1"  # Change to your preferred region
PROJECT_ID=""  # Leave empty to use current project, or specify: "your-project-id"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  FTSC RAG Search - Cloud Run Deploy${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed${NC}"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID if not set
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}Error: No GCP project configured${NC}"
        echo "Run: gcloud config set project YOUR_PROJECT_ID"
        exit 1
    fi
fi

echo -e "${YELLOW}Project:${NC} $PROJECT_ID"
echo -e "${YELLOW}Region:${NC} $REGION"
echo -e "${YELLOW}Service:${NC} $SERVICE_NAME"
echo ""

# Check for RAGIE_API_KEY
if [ -z "$RAGIE_API_KEY" ]; then
    echo -e "${YELLOW}Warning: RAGIE_API_KEY not found in environment${NC}"
    echo "You'll need to set it after deployment with:"
    echo "  gcloud run services update $SERVICE_NAME --update-env-vars RAGIE_API_KEY=your_key --region $REGION"
    echo ""
    read -p "Continue without setting API key? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for GOOGLE_DRIVE_FOLDER_ID (optional, has default)
if [ -z "$GOOGLE_DRIVE_FOLDER_ID" ]; then
    echo -e "${YELLOW}Note: GOOGLE_DRIVE_FOLDER_ID not set, will use default from code${NC}"
    echo ""
fi

echo -e "${GREEN}Deploying to Cloud Run...${NC}"
echo ""

# Deploy with cost-optimized settings
# Build environment variables string
ENV_VARS=""
if [ -n "$RAGIE_API_KEY" ]; then
    ENV_VARS="RAGIE_API_KEY=$RAGIE_API_KEY"
fi
if [ -n "$GOOGLE_DRIVE_FOLDER_ID" ]; then
    if [ -n "$ENV_VARS" ]; then
        ENV_VARS="$ENV_VARS,GOOGLE_DRIVE_FOLDER_ID=$GOOGLE_DRIVE_FOLDER_ID"
    else
        ENV_VARS="GOOGLE_DRIVE_FOLDER_ID=$GOOGLE_DRIVE_FOLDER_ID"
    fi
fi

gcloud run deploy "$SERVICE_NAME" \
    --source . \
    --region "$REGION" \
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
    --port 8080 \
    ${ENV_VARS:+--set-env-vars $ENV_VARS} \
    --project "$PROJECT_ID"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Deployment Successful!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    # Get the service URL
    SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format 'value(status.url)' 2>/dev/null)

    if [ -n "$SERVICE_URL" ]; then
        echo -e "${YELLOW}Service URL:${NC} $SERVICE_URL"
        echo ""
    fi

    echo -e "${YELLOW}Cost-Saving Features Enabled:${NC}"
    echo "   Scales to zero when idle (no charges)"
    echo "   Only charged for actual request time"
    echo "   512MB memory (minimal overhead)"
    echo "   CPU throttling (cheaper billing)"
    echo ""

    if [ -z "$RAGIE_API_KEY" ]; then
        echo -e "${YELLOW}Next Step:${NC} Set your RAGIE_API_KEY"
        echo "  gcloud run services update $SERVICE_NAME \\"
        echo "    --update-env-vars RAGIE_API_KEY=your_actual_key \\"
        echo "    --region $REGION"
        echo ""
    fi

    echo -e "${YELLOW}Test your deployment:${NC}"
    echo "  curl $SERVICE_URL/test"
    echo ""

    echo -e "${YELLOW}View logs:${NC}"
    echo "  gcloud run services logs read $SERVICE_NAME --region $REGION"
    echo ""

else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Deployment Failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    exit 1
fi
