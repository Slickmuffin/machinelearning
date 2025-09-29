#!/bin/bash

# Submit custom training job using gcloud CLI
JOB_NAME="btc-trading-custom-$(date +%Y%m%d-%H%M%S)"

echo "🚀 Submitting custom BTC trading AI training job: $JOB_NAME"
echo "📊 Using our specialized code with 57,820 samples and 86 features"

gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name="$JOB_NAME" \
  --python-package-uris="gs://YOUR_GCP_PROJECT-ml-trading/code/trading_ai_code.tar.gz" \
  --python-module="scripts.gcp.vertex_train_csv" \
  --container-image-uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest" \
  --machine-type="n1-standard-4" \
  --boot-disk-size="100GB" \
  --boot-disk-type="pd-ssd" \
  --env-vars="TRAINING_DATA_URI=gs://YOUR_GCP_PROJECT-ml-trading/btc_comprehensive_dataset_20250812_125933_features.csv,AIP_MODEL_DIR=gs://YOUR_GCP_PROJECT-ml-trading/models/$JOB_NAME"

echo "✅ Training job submitted!"
echo "🌐 Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=YOUR_GCP_PROJECT"