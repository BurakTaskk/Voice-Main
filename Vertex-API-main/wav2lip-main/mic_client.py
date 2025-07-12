# mic_client.py
import asyncio
import websockets
import sounddevice as sd

# --- AYARLAR ---
SERVER_URI = "ws://localhost:8000/ws/mic_input"
SAMPLE_RATE = 16000  # Wav2Lip için 16kHz olmalı
CHANNELS = 1         # Mono
BLOCK_SIZE = 8000    # Saniyede 2 kez veri gönder (16000 * 0.5)

async def audio_sender():
    # Ses akışını başlatmak için bir kuyruk
    q = asyncio.Queue()

    def callback(indata, frames, time, status):
        """Bu fonksiyon her ses bloğu için çağrılır."""
        asyncio.run_coroutine_threadsafe(q.put(indata.copy()), loop)

    print(f"Sunucuya bağlanılıyor: {SERVER_URI}")
    async with websockets.connect(SERVER_URI) as websocket:
        print("Bağlantı kuruldu. Mikrofon dinleniyor...")
        
        loop = asyncio.get_event_loop()
        
        # Ses cihazından akışı başlat
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, 
                             dtype='int16', blocksize=BLOCK_SIZE, callback=callback):
            while True:
                # Kuyruktan ses verisini al
                audio_data = await q.get()
                # Sunucuya byte olarak gönder
                await websocket.send(audio_data.tobytes())

if __name__ == "__main__":
    try:
        asyncio.run(audio_sender())
    except KeyboardInterrupt:
        print("\nİşlem durduruldu.")