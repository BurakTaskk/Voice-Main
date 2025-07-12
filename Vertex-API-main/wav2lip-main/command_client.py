"""
Kullanım:
    python command_client.py C:/tam/yol/ses.wav
"""

import asyncio, base64, json, pathlib, sys, websockets

wav_path = pathlib.Path(sys.argv[1])
if not wav_path.exists():
    sys.exit(f"WAV bulunamadı: {wav_path}")

b64 = base64.b64encode(wav_path.read_bytes()).decode()
msg = json.dumps({"audio_b64": b64})

async def main():
    async with websockets.connect("ws://localhost:8000/ws/command") as ws:
        await ws.send(msg)
        print("Gönderildi:", await ws.recv())

asyncio.run(main())
