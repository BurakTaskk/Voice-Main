# Live API Audio Processing - Wav2Lip Streaming Server

Bu dokümantasyon, `streaming_server_mp4_perviewer.py` dosyasının live API'den gelen ses dosyalarını işleyecek şekilde nasıl modifiye edildiğini açıklar.

## Yapılan Değişiklikler

### 1. Ses Formatı Desteği
- **Desteklenen Formatlar**: WAV, MP3, OGG, FLAC, M4A, AAC, AMR, Opus, WebM
- **Otomatik Format Algılama**: `python-magic` kütüphanesi kullanılarak
- **Manuel Format Belirtme**: WebSocket mesajlarında `audio_format` parametresi

### 2. Ses Dönüştürme
- **FFmpeg Entegrasyonu**: Tüm ses formatlarını WAV'a dönüştürme
- **Standart Format**: 16kHz, 16-bit PCM, Mono
- **Hata Yönetimi**: Dönüştürme başarısız olursa orijinal veriyi kullanma

### 3. WebSocket API Güncellemeleri
- **Format Bilgisi**: Mesajlarda opsiyonel `audio_format` alanı
- **Gelişmiş Yanıtlar**: Format bilgisi ile birlikte durum yanıtları

## Kurulum

### Gerekli Kütüphaneler
```bash
pip install -r requirements.txt
```

### Yeni Eklenen Bağımlılıklar
- `python-magic>=0.4.27` - Ses formatı algılama
- `fastapi>=0.68.0` - Web API framework
- `uvicorn>=0.15.0` - ASGI server

## Kullanım

### 1. Sunucuyu Başlatma
```bash
python streaming_server_mp4_perviewer.py
# veya
python -m uvicorn streaming_server_mp4_perviewer:app --host 0.0.0.0 --port 8000
```

### 2. WebSocket Bağlantısı
```javascript
// Bağlantı
const ws = new WebSocket('ws://localhost:8000/ws/command');

// Ses gönderme (format belirtme ile)
ws.send(JSON.stringify({
    "audio_b64": "base64_encoded_audio_data",
    "audio_format": "audio/mpeg"  // Opsiyonel
}));

// Ses gönderme (otomatik algılama ile)
ws.send(JSON.stringify({
    "audio_b64": "base64_encoded_audio_data"
}));
```

### 3. Test İstemcisi
```bash
# Tüm test dosyalarını dene
python test_live_api_client.py

# Belirli bir dosyayı test et
python test_live_api_client.py --audio-file test.mp3 --format audio/mpeg

# Farklı sunucu adresi
python test_live_api_client.py --server ws://192.168.1.100:8000/ws/command
```

## API Mesaj Formatları

### Gönderilen Mesaj
```json
{
    "audio_b64": "base64_encoded_audio_data",
    "audio_format": "audio/mpeg"  // Opsiyonel
}
```

### Alınan Yanıt
```json
{
    "status": "queued",
    "format": "audio/mpeg"  // veya "auto-detected"
}
```

## Desteklenen Ses Formatları

| Format | MIME Type | Dosya Uzantısı |
|--------|-----------|----------------|
| WAV | audio/wav | .wav |
| MP3 | audio/mpeg | .mp3 |
| OGG | audio/ogg | .ogg |
| FLAC | audio/flac | .flac |
| M4A | audio/mp4 | .m4a |
| AAC | audio/aac | .aac |
| AMR | audio/amr | .amr |
| Opus | audio/opus | .opus |
| WebM | audio/webm | .webm |

## Hata Yönetimi

### Ses Dönüştürme Hataları
- FFmpeg dönüştürme başarısız olursa orijinal ses verisi kullanılır
- 30 saniye timeout süresi
- Geçici dosyalar otomatik temizlenir

### Format Algılama
- `python-magic` başarısız olursa dosya başlığına göre manuel kontrol
- Hiçbir format algılanamazsa WAV olarak kabul edilir

## Performans Optimizasyonları

### 1. Bellek Yönetimi
- Geçici dosyalar otomatik temizlenir
- Büyük ses dosyaları parçalara bölünür

### 2. Hız Optimizasyonları
- WAV dosyaları doğrudan işlenir (dönüştürme yapılmaz)
- FFmpeg pipe modu kullanılır
- CUDA hızlandırma (varsa)

### 3. Ölçeklenebilirlik
- Asenkron işleme
- WebSocket bağlantı yönetimi
- Kuyruk sistemi

## Örnek Kullanım Senaryoları

### 1. Live API Entegrasyonu
```python
import asyncio
import websockets
import base64

async def send_live_audio(audio_bytes, format_type="audio/mpeg"):
    async with websockets.connect('ws://localhost:8000/ws/command') as ws:
        message = {
            "audio_b64": base64.b64encode(audio_bytes).decode(),
            "audio_format": format_type
        }
        await ws.send(json.dumps(message))
        response = await ws.recv()
        return json.loads(response)
```

### 2. Mikrofon Kaydı
```javascript
// Mikrofon kaydı ve gönderme
navigator.mediaDevices.getUserMedia({audio: true})
    .then(stream => {
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = event => {
            const reader = new FileReader();
            reader.onload = () => {
                const base64 = btoa(reader.result);
                ws.send(JSON.stringify({
                    "audio_b64": base64,
                    "audio_format": "audio/webm"
                }));
            };
            reader.readAsBinaryString(event.data);
        };
        mediaRecorder.start();
    });
```

## Sorun Giderme

### Yaygın Hatalar

1. **FFmpeg Bulunamadı**
   ```
   Error: FFmpeg dönüştürme hatası
   ```
   **Çözüm**: FFmpeg'in sistem PATH'inde olduğundan emin olun

2. **python-magic Hatası**
   ```
   ModuleNotFoundError: No module named 'magic'
   ```
   **Çözüm**: `pip install python-magic` komutunu çalıştırın

3. **Ses Formatı Desteklenmiyor**
   ```
   RuntimeError: Ses dönüştürme hatası
   ```
   **Çözüm**: Desteklenen formatlardan birini kullanın

### Debug Modu
```python
# Debug için logging ekleyin
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Gelecek Geliştirmeler

- [ ] Daha fazla ses formatı desteği
- [ ] Gerçek zamanlı ses akışı
- [ ] Ses kalitesi ayarları
- [ ] Batch işleme
- [ ] Cloud storage entegrasyonu 