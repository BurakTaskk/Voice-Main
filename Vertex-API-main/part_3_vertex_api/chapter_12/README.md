# Vertex API Gemini Multimodal Live WebSocket Demo

Bu proje, Google Vertex AI Gemini 2.0 Flash Live Preview modelini kullanarak gerçek zamanlı ses ve video iletişimi sağlayan bir WebSocket proxy sunucusu ve frontend uygulamasıdır.

## Proje Yapısı

```
chapter_12/
├── proxy/                    # WebSocket Proxy Servisi
│   ├── proxy.py             # Ana proxy sunucu kodu
│   ├── requirements.txt     # Python bağımlılıkları
│   ├── Dockerfile          # Proxy için Docker konfigürasyonu
│   └── voice-asistant-459013-29c675d43902.json  # Service Account dosyası
├── frontend/                # Frontend Uygulaması
│   ├── index.html          # Ana HTML dosyası
│   ├── style.css           # CSS stilleri
│   ├── status-handler.js   # Durum yönetimi
│   ├── shared/             # Paylaşılan JavaScript modülleri
│   ├── Dockerfile          # Frontend için Docker konfigürasyonu
│   └── nginx.conf          # Nginx konfigürasyonu
├── shared/                  # Paylaşılan dosyalar (orijinal)
├── deploy-all.sh           # Linux/Mac deployment scripti
├── deploy-all.ps1          # Windows PowerShell deployment scripti
└── README.md               # Bu dosya
```

## Cloud Run Deployment

Bu proje iki ayrı Cloud Run servisi olarak deploy edilir:

1. **Proxy Servisi** (`vertex-proxy-service`) - WebSocket proxy sunucusu
2. **Frontend Servisi** (`vertex-frontend-service`) - Web arayüzü

### Ön Gereksinimler

1. **Google Cloud SDK** yüklü olmalı
2. **Docker** yüklü olmalı
3. **Google Cloud projesi** aktif olmalı
4. **Service Account** dosyası mevcut olmalı

### Deployment Adımları

#### Windows için:

```powershell
# PowerShell'i yönetici olarak çalıştırın
.\deploy-all.ps1
```

#### Linux/Mac için:

```bash
# Script'i çalıştırılabilir yapın
chmod +x deploy-all.sh

# Deployment'ı başlatın
./deploy-all.sh
```

### Manuel Deployment

Eğer script kullanmak istemiyorsanız, aşağıdaki adımları takip edebilirsiniz:

#### 1. Proxy Servisi Deploy Etme

```bash
cd proxy
docker build -t gcr.io/voice-asistant-459013/vertex-proxy-service .
docker push gcr.io/voice-asistant-459013/vertex-proxy-service

gcloud run deploy vertex-proxy-service \
  --image gcr.io/voice-asistant-459013/vertex-proxy-service \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10 \
  --timeout 3600 \
  --concurrency 80
```

#### 2. Frontend Servisi Deploy Etme

```bash
cd frontend
docker build -t gcr.io/voice-asistant-459013/vertex-frontend-service .
docker push gcr.io/voice-asistant-459013/vertex-frontend-service

gcloud run deploy vertex-frontend-service \
  --image gcr.io/voice-asistant-459013/vertex-frontend-service \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --cpu 1 \
  --max-instances 5 \
  --timeout 300 \
  --concurrency 1000
```

## Özellikler

### Proxy Servisi
- WebSocket bağlantı yönetimi
- Vertex AI API ile iletişim
- Service Account kimlik doğrulaması
- Gerçek zamanlı ses/video proxy
- Otomatik bağlantı temizleme
- **Hava durumu API endpoint'i** (`/weather`) - OpenWeatherMap API'si üzerinden hava durumu verileri sağlar

### Frontend Servisi
- Modern web arayüzü
- Ses kayıt ve oynatma
- Video kamera desteği
- Ekran paylaşımı
- Gerçek zamanlı iletişim
- Responsive tasarım
- **Hava durumu sorgulama** - Şehir bazlı hava durumu bilgileri

## API Endpoints

### Weather Endpoint

Proxy servisi aşağıdaki hava durumu endpoint'ini sağlar:

```
GET /weather?city={city_name}
```

**Parametreler:**
- `city` (required): Hava durumu bilgisi alınacak şehir adı

**Örnek Kullanım:**
```bash
curl "https://vertex-proxy-service-638345404110.us-central1.run.app/weather?city=Istanbul"
```

**Örnek Yanıt:**
```json
{
  "temperature": 18.5,
  "description": "scattered clouds",
  "humidity": 65,
  "windSpeed": 3.2,
  "city": "Istanbul",
  "country": "TR"
}
```

**Hata Yanıtı:**
```json
{
  "error": "Could not find location: InvalidCity"
}
```

### WebSocket Endpoint

```
WS /ws
```

Gemini AI ile gerçek zamanlı iletişim için WebSocket bağlantısı.

## Kullanım

1. Deployment tamamlandıktan sonra frontend URL'sini tarayıcınızda açın
2. Mikrofon ve kamera izinlerini verin
3. "Play" butonuna tıklayarak sesli iletişimi başlatın
4. Gemini AI ile gerçek zamanlı konuşma yapabilirsiniz

## Güvenlik

- Service Account dosyası container içinde güvenli şekilde saklanır
- HTTPS bağlantıları kullanılır
- CORS politikaları yapılandırılmıştır
- Kimlik doğrulama otomatik olarak yapılır

## Sorun Giderme

### Yaygın Sorunlar

1. **Port Hatası**: Cloud Run'da PORT environment variable'ı otomatik olarak ayarlanır
2. **Service Account Hatası**: Dosya yolu container içinde doğru olmalıdır
3. **CORS Hatası**: Frontend ve proxy servisleri arasında CORS ayarları kontrol edin

### Logları Görüntüleme

```bash
# Proxy servisi logları
gcloud logs read --service=vertex-proxy-service --limit=50

# Frontend servisi logları
gcloud logs read --service=vertex-frontend-service --limit=50
```

## Geliştirme

### Yerel Geliştirme

```bash
# Proxy servisini yerel olarak çalıştırma
cd proxy
python proxy.py

# Frontend'i yerel olarak çalıştırma
cd frontend
python -m http.server 8080
```

### Docker Compose (İsteğe Bağlı)

```yaml
version: '3.8'
services:
  proxy:
    build: ./proxy
    ports:
      - "8080:8080"
    environment:
      - PORT=8080
  
  frontend:
    build: ./frontend
    ports:
      - "8081:8080"
    depends_on:
      - proxy
```

## Lisans

Bu proje Apache 2.0 lisansı altında lisanslanmıştır.
