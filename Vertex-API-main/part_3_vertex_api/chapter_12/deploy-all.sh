#!/bin/bash

# Tüm servisleri Cloud Run'a deploy etmek için ana script

echo "=== Vertex API Cloud Run Deployment ==="
echo ""

# Proje ID'si
PROJECT_ID="voice-asistant-459013"
REGION="us-central1"

# Google Cloud projesini ayarla
echo "Google Cloud projesi ayarlanıyor..."
gcloud config set project $PROJECT_ID

# Container Registry'yi etkinleştir
echo "Container Registry etkinleştiriliyor..."
gcloud services enable containerregistry.googleapis.com

# Cloud Run API'yi etkinleştir
echo "Cloud Run API etkinleştiriliyor..."
gcloud services enable run.googleapis.com

echo ""
echo "=== Proxy Servisi Deploy Ediliyor ==="
# Proxy servisini deploy et
cd proxy
docker build -t gcr.io/$PROJECT_ID/vertex-proxy-service .
docker push gcr.io/$PROJECT_ID/vertex-proxy-service

gcloud run deploy vertex-proxy-service \
  --image gcr.io/$PROJECT_ID/vertex-proxy-service \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10 \
  --timeout 3600 \
  --concurrency 80

PROXY_URL=$(gcloud run services describe vertex-proxy-service --region=$REGION --format='value(status.url)')
echo "Proxy URL: $PROXY_URL"

cd ..

echo ""
echo "=== Frontend Servisi Deploy Ediliyor ==="
# Frontend'deki proxy URL'yi güncelle
cd frontend
# index.html'deki proxy URL'yi güncelle
sed -i "s|proxy-service-url|${PROXY_URL#https://}|g" index.html

# Frontend servisini deploy et
docker build -t gcr.io/$PROJECT_ID/vertex-frontend-service .
docker push gcr.io/$PROJECT_ID/vertex-frontend-service

gcloud run deploy vertex-frontend-service \
  --image gcr.io/$PROJECT_ID/vertex-frontend-service \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --cpu 1 \
  --max-instances 5 \
  --timeout 300 \
  --concurrency 1000

FRONTEND_URL=$(gcloud run services describe vertex-frontend-service --region=$REGION --format='value(status.url)')
echo "Frontend URL: $FRONTEND_URL"

cd ..

echo ""
echo "=== Deployment Tamamlandı ==="
echo "Proxy Servisi: $PROXY_URL"
echo "Frontend Servisi: $FRONTEND_URL"
echo ""
echo "Frontend URL'sini tarayıcınızda açarak uygulamayı kullanabilirsiniz!" 