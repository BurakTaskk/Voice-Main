#!/bin/bash

# Proxy servisi için Cloud Run deployment scripti

# Proje ID'si
PROJECT_ID="voice-asistant-459013"
SERVICE_NAME="vertex-proxy-service"
REGION="us-central1"

echo "Proxy servisi deploy ediliyor..."

# Docker image build et
cd proxy
docker build -t gcr.io/$PROJECT_ID/$SERVICE_NAME .

# Image'ı Google Container Registry'ye push et
docker push gcr.io/$PROJECT_ID/$SERVICE_NAME

# Cloud Run'da servisi deploy et
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10 \
  --timeout 3600 \
  --concurrency 80

echo "Proxy servisi deploy edildi!"
echo "URL: $(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')" 