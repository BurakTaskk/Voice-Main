# Tüm servisleri Cloud Run'a deploy etmek için PowerShell scripti

Write-Host "=== Vertex API Cloud Run Deployment ===" -ForegroundColor Green
Write-Host ""

# Proje ID'si
$PROJECT_ID = "voice-asistant-459013"
$REGION = "us-central1"

# Google Cloud projesini ayarla
Write-Host "Google Cloud projesi ayarlanıyor..." -ForegroundColor Yellow
gcloud config set project $PROJECT_ID

# Container Registry'yi etkinleştir
Write-Host "Container Registry etkinleştiriliyor..." -ForegroundColor Yellow
gcloud services enable containerregistry.googleapis.com

# Cloud Run API'yi etkinleştir
Write-Host "Cloud Run API etkinleştiriliyor..." -ForegroundColor Yellow
gcloud services enable run.googleapis.com

Write-Host ""
Write-Host "=== Proxy Servisi Deploy Ediliyor ===" -ForegroundColor Cyan

# Proxy servisini deploy et
Set-Location proxy
docker build -t gcr.io/$PROJECT_ID/vertex-proxy-service .
docker push gcr.io/$PROJECT_ID/vertex-proxy-service

gcloud run deploy vertex-proxy-service `
  --image gcr.io/$PROJECT_ID/vertex-proxy-service `
  --platform managed `
  --region $REGION `
  --allow-unauthenticated `
  --port 8080 `
  --memory 1Gi `
  --cpu 1 `
  --max-instances 10 `
  --timeout 3600 `
  --concurrency 80

$PROXY_URL = (gcloud run services describe vertex-proxy-service --region=$REGION --format='value(status.url)')
Write-Host "Proxy URL: $PROXY_URL" -ForegroundColor Green

Set-Location ..

Write-Host ""
Write-Host "=== Frontend Servisi Deploy Ediliyor ===" -ForegroundColor Cyan

# Frontend'deki proxy URL'yi güncelle
Set-Location frontend

# index.html'deki proxy URL'yi güncelle (Windows için)
$proxyHost = $PROXY_URL -replace "https://", ""
(Get-Content index.html) -replace "proxy-service-url", $proxyHost | Set-Content index.html

# Frontend servisini deploy et
docker build -t gcr.io/$PROJECT_ID/vertex-frontend-service .
docker push gcr.io/$PROJECT_ID/vertex-frontend-service

gcloud run deploy vertex-frontend-service `
  --image gcr.io/$PROJECT_ID/vertex-frontend-service `
  --platform managed `
  --region $REGION `
  --allow-unauthenticated `
  --port 8080 `
  --memory 512Mi `
  --cpu 1 `
  --max-instances 5 `
  --timeout 300 `
  --concurrency 1000

$FRONTEND_URL = (gcloud run services describe vertex-frontend-service --region=$REGION --format='value(status.url)')
Write-Host "Frontend URL: $FRONTEND_URL" -ForegroundColor Green

Set-Location ..

Write-Host ""
Write-Host "=== Deployment Tamamlandı ===" -ForegroundColor Green
Write-Host "Proxy Servisi: $PROXY_URL" -ForegroundColor Yellow
Write-Host "Frontend Servisi: $FRONTEND_URL" -ForegroundColor Yellow
Write-Host ""
Write-Host "Frontend URL'sini tarayıcınızda açarak uygulamayı kullanabilirsiniz!" -ForegroundColor Green 