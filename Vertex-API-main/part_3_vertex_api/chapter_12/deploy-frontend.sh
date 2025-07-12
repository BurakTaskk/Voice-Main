#!/bin/bash

# Frontend servisi için Cloud Run deployment scripti

# Proje ID'si
PROJECT_ID="voice-asistant-459013"
SERVICE_NAME="vertex-frontend-service"
REGION="us-central1"

echo "Frontend servisi deploy ediliyor..."

# Docker image build et
cd frontend
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
  --memory 512Mi \
  --cpu 1 \
  --max-instances 5 \
  --timeout 300 \
  --concurrency 1000

echo "Frontend servisi deploy edildi!"
echo "URL: $(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')" 