#!/bin/bash

# Deploy Fixed Risk-Aware PPO Trading AI
# This script deploys the corrected PPO implementation

echo "🚀 DEPLOYING FIXED RISK-AWARE PPO TRADING AI"
echo "================================================"

# Set project and region
PROJECT_ID="YOUR_GCP_PROJECT"
REGION="us-central1"
JOB_NAME="PPO-RISK-AWARE-FIXED-$(date +%Y%m%d-%H%M%S)"

echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Job Name: $JOB_NAME"
echo ""

# Build container
echo "🔨 Building fixed PPO container..."
gcloud builds submit --config cloudbuild_fixed.yaml .

if [ $? -eq 0 ]; then
    echo "✅ Container build successful"
else
    echo "❌ Container build failed"
    exit 1
fi

echo ""

# Update job config with new image
echo "📝 Updating job configuration..."
sed -i.bak "s|gcr.io/deeplearning-platform-release/pytorch-gpu.1-13.py310|gcr.io/$PROJECT_ID/ppo-trading-ai-fixed:latest|g" ppo_training_job.yaml

# Deploy training job
echo "🚀 Deploying training job..."
gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name="$JOB_NAME" \
    --config=ppo_training_job.yaml

if [ $? -eq 0 ]; then
    echo "✅ Training job deployed successfully"
    echo ""
    echo "📊 Monitor job with:"
    echo "gcloud ai custom-jobs list --region=$REGION --filter=\"displayName:PPO-RISK-AWARE-FIXED\""
    echo ""
    echo "📋 View logs with:"
    echo "gcloud ai custom-jobs stream-logs <JOB_ID> --region=$REGION"
else
    echo "❌ Training job deployment failed"
    exit 1
fi

echo ""
echo "🎯 DEPLOYMENT COMPLETE!"
echo "Key improvements implemented:"
echo "✅ Continuous action space [-1, 1]"
echo "✅ Risk-aware log-equity rewards"
echo "✅ Proper data validation"
echo "✅ Compact recurrent architecture"
echo "✅ GPU optimization with AMP"
echo "✅ Kill-switches and risk controls"
echo ""
echo "Expected performance: 80%+ GPU utilization, stable learning, risk-controlled trading"