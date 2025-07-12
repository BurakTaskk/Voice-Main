# Vertex API Projesi

Bu proje, Google Cloud Vertex AI API'sini kullanarak ses ve görüntü işleme özellikleri sunan bir web uygulamasıdır.

## Özellikler

- Ses tanıma ve işleme
- Görüntü işleme
- Hava durumu bilgisi alma
- Google araması yapma
- YouTube video araması

## Kurulum

1. Projeyi klonlayın:
```bash
git clone https://github.com/[kullanıcı-adınız]/Vertex-API.git
cd Vertex-API
```

2. Gerekli bağımlılıkları yükleyin:
```bash
npm install
```

3. `.env` dosyası oluşturun ve gerekli API anahtarlarını ekleyin:
```
OPENWEATHER_API_KEY=your_api_key_here
```

4. Uygulamayı başlatın:
```bash
npm start
```

## Kullanım

1. Tarayıcınızda `http://localhost:3000` adresine gidin
2. Mikrofon düğmesine basın ve komutunuzu söyleyin
3. Desteklenen komutlar:
   - "İstanbul'da hava durumu nasıl?"
   - "YouTube'da [video adı] ara"
   - "Google'da [arama terimi] ara"

## Teknolojiler

- Node.js
- Google Cloud Vertex AI
- WebSocket
- HTML5 Audio/Video API

## Lisans

Bu proje Apache 2.0 lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın. 