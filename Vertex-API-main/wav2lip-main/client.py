# client.py

import asyncio
import websockets
import base64
import json
import os

# ==============================================================================
# --- AYARLAR ---
# ==============================================================================

# *** ÖNEMLİ: BURAYI KENDİ KURULUMUNUZA GÖRE DEĞİŞTİRİN ***
# Artık Uvicorn'un çalıştığı porta değil, Reverse Proxy'nin (NGINX) adresine bağlanıyoruz.

# --- Seçenek 1: NGINX ve Uvicorn aynı makinede çalışıyorsa (Lokal Test) ---
# NGINX varsayılan olarak 80 portunda çalışır, bu yüzden port belirtmeye gerek yoktur.
SERVER_URI = "ws://localhost:8000/ws"
# veya
# SERVER_URI = "ws://127.0.0.1/ws"


# --- Seçenek 2: NGINX uzak bir sunucuda (IP adresi ile) çalışıyorsa ---
# SERVER_URI = "ws://SUNUCUNUZUN_IP_ADRESI/ws"


# --- Seçenek 3: NGINX uzak bir sunucuda (Alan Adı ile) çalışıyorsa ve SSL (HTTPS) YOKSA ---
# SERVER_URI = "ws://siteniz.com/ws"


# --- Seçenek 4: NGINX uzak bir sunucuda (Alan Adı ile) çalışıyorsa ve SSL (HTTPS) VARSA (TAVSİYE EDİLEN) ---
# "ws://" yerine "wss://" (WebSocket Secure) kullanılır. NGINX port 443'te dinler.
# SERVER_URI = "wss://siteniz.com/ws"


# --- Dosya Yolları (Bu kısımları kendi bilgisayarınıza göre düzenleyin) ---
VIDEO_FILE_PATH = r"C:\Users\taski\OneDrive\Desktop\wav2lip-main\durgunkız.mp4"
AUDIO_FILE_PATH = r"C:\Users\taski\OneDrive\Desktop\wav2lip-main\sesss.WAV"
OUTPUT_FILE_PATH = r"sonuc_websocket_proxy.mp4"      # Sonucun kaydedileceği yer

# ==============================================================================

async def run_lip_sync():
    """Sunucuya bağlanır, dosyaları gönderir ve sonucu alır."""
    
    if not os.path.exists(VIDEO_FILE_PATH):
        print(f"Hata: Video dosyası bulunamadı -> {VIDEO_FILE_PATH}")
        return
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"Hata: Ses dosyası bulunamadı -> {AUDIO_FILE_PATH}")
        return

    print(f"Proxy sunucusuna bağlanılıyor: {SERVER_URI}")
    
    # max_size parametresini büyük dosyalar için yeterli bir değere ayarlayın (örn: 100 MB için 100 * 1024 * 1024)
    async with websockets.connect(SERVER_URI, max_size=100 * 1024 * 1024, ping_interval=None) as websocket:
        print("Bağlantı başarıyla kuruldu.")

        # Dosyaları oku ve Base64 formatına çevir
        print("Dosyalar okunuyor ve kodlanıyor...")
        try:
            with open(VIDEO_FILE_PATH, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            with open(AUDIO_FILE_PATH, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
        except Exception as e:
            print(f"Dosya okuma hatası: {e}")
            return
            
        # Sunucuya göndermek için JSON mesajı oluştur
        message = {
            "video_b64": video_b64,
            "audio_b64": audio_b64
        }

        print("Veriler sunucuya gönderiliyor...")
        await websocket.send(json.dumps(message))
        print("Veriler gönderildi. Sunucudan yanıt bekleniyor...")

        # Sunucudan gelen yanıtları dinle
        while True:
            try:
                # Sunucudan gelen mesajı bekle (timeout eklemek isteğe bağlıdır)
                response = await asyncio.wait_for(websocket.recv(), timeout=300.0) # 5 dakika bekle
                
                # Gelen yanıt binary (video dosyası) mi yoksa text (durum mesajı) mi?
                if isinstance(response, bytes):
                    print(f"Sonuç video verisi alındı! Boyut: {len(response) / (1024*1024):.2f} MB.")
                    with open(OUTPUT_FILE_PATH, "wb") as f:
                        f.write(response)
                    print(f"Video başarıyla '{OUTPUT_FILE_PATH}' olarak kaydedildi.")
                    break # İşlem bitti, döngüden çık
                
                elif isinstance(response, str):
                    status_update = json.loads(response)
                    if "status" in status_update:
                        print(f"[SUNUCU DURUMU]: {status_update['status']}")
                    elif "error" in status_update:
                        print(f"[SUNUCU HATASI]: {status_update['error']}")
                        break # Hata varsa döngüden çık
                
            except asyncio.TimeoutError:
                print("Hata: Sunucudan yanıt almak için bekleme süresi aşıldı.")
                break
            except websockets.ConnectionClosed as e:
                print(f"Sunucu bağlantıyı kapattı. Kod: {e.code}, Sebep: {e.reason}")
                break
            except Exception as e:
                print(f"Beklenmedik bir hata oluştu: {e}")
                break

if __name__ == "__main__":
    try:
        asyncio.run(run_lip_sync())
    except KeyboardInterrupt:
        print("\nİşlem kullanıcı tarafından iptal edildi.")