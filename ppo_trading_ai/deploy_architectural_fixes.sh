#!/bin/bash

# Deploy Architectural-Fix PPO Trading AI
# This script deploys the PPO implementation with senior dev architectural fixes

echo "🚀 DEPLOYING ARCHITECTURAL-FIX PPO TRADING AI"
echo "================================================"

# Set project and region
PROJECT_ID="YOUR_GCP_PROJECT"
REGION="us-central1"
JOB_NAME="PPO-ARCHITECTURAL-FIXES-$(date +%Y%m%d-%H%M%S)"

echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Job Name: $JOB_NAME"
echo ""

# Build container
echo "🔨 Building architectural-fixes PPO container..."
gcloud builds submit --config cloudbuild_architectural.yaml .

if [ $? -eq 0 ]; then
    echo "✅ Container build successful"
else
    echo "❌ Container build failed"
    exit 1
fi

echo ""

# Deploy training job with new architectural fixes
echo "🚀 Deploying training job..."
gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name="$JOB_NAME" \
    --config=ppo_training_job_architectural.yaml

if [ $? -eq 0 ]; then
    echo "✅ Training job deployed successfully"
    echo ""
    echo "📊 Monitor job with:"
    echo "gcloud ai custom-jobs list --region=$REGION --filter=\"displayName:PPO-ARCHITECTURAL-FIXES\""
    echo ""
    echo "📋 View logs with:"
    echo "gcloud ai custom-jobs stream-logs <JOB_ID> --region=$REGION"
else
    echo "❌ Training job deployment failed"
    exit 1
fi

echo ""
echo "🎯 DEPLOYMENT COMPLETE!"
echo "Key architectural improvements implemented:"
echo "✅ Squashed Gaussian policy with log-det-jacobian correction"
echo "✅ Full minibatch coverage (all samples processed each epoch)"
echo "✅ Pure MLP architecture (GRU removed)"
echo "✅ Dropout removed from policy/value paths"
echo "✅ Proper weight initialization with gains"
echo "✅ Fixed GAE bootstrapping with next state values"
echo "✅ Numerical stability improvements"
echo ""
echo "Expected performance: Stable learning, no NaN issues, better sample efficiency"