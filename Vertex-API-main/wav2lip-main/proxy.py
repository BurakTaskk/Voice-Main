# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Vertex AI Gemini Multimodal Live WebSockets Proxy Server """
import asyncio
import json
import ssl
import traceback
import websockets
import certifi
import google.auth
import os
import aiohttp
from aiohttp import web
from google.auth.transport.requests import Request
from websockets.legacy.protocol import WebSocketCommonProtocol
from websockets.legacy.server import WebSocketServerProtocol
from google.oauth2 import service_account
import base64
import io
import wave

# --- Wav2Lip servis kodları ve FastAPI app entegrasyonu ---
import cv2, mediapipe as mp, numpy as np, torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import imageio_ffmpeg
import audio
from models import Wav2Lip
import pathlib, tempfile, shutil, io, wave, base64, subprocess, logging
from typing import List, Tuple, Generator

# --- Wav2Lip yardımcı fonksiyon ve sınıfları ---
PADS = (0, 10, 20, 10)
WAV2LIP_BATCH = 128
STATIC_FACE = True
FAST_BLEND = True
USE_FP16 = True
USE_PIPE = True
FFMPEG_CRF = 23
FFMPEG_PRESET = "ultrafast"
CHUNK_SEC = 1  # WAV dosyası 1 sn'lik parçalara bölünür

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "wav2lip-main"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "wav2lip_gan.pth"
FACE_VIDEO_PATH = BASE_DIR / "beklemevideosu5.mp4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert CHECKPOINT_PATH.exists(), f"Checkpoint bulunamadı: {CHECKPOINT_PATH}"
assert FACE_VIDEO_PATH.exists(), f"Yüz videosu bulunamadı: {FACE_VIDEO_PATH}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — [W2L] — %(levelname)s — %(message)s",
    force=False,
)

def stream_wav_chunks(wav_bytes: bytes, chunk_sec: float = CHUNK_SEC) -> Generator[bytes, None, None]:
    with io.BytesIO(wav_bytes) as bio, wave.open(bio, "rb") as wf:
        nch, sw, fr, total = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        frames_chunk = int(fr * chunk_sec)
        for start in range(0, total, frames_chunk):
            wf.setpos(start)
            data = wf.readframes(frames_chunk)
            if not data:
                break
            out = io.BytesIO()
            with wave.open(out, "wb") as wout:
                wout.setparams((nch, sw, fr, 0, "NONE", "NONE"))
                wout.writeframes(data)
            yield out.getvalue()

class WSManager:
    def __init__(self) -> None:
        self.clients: List[WebSocket] = []
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)
    def disconnect(self, ws: WebSocket):
        self.clients = [c for c in self.clients if c != ws]
    async def broadcast_video(self, payload: bytes):
        for ws in self.clients.copy():
            try:
                await ws.send_bytes(payload)
            except WebSocketDisconnect:
                self.disconnect(ws)
    async def broadcast_done(self, total_sub_segments: int):
        msg = json.dumps({"type": "done", "total": total_sub_segments})
        for ws in self.clients.copy():
            try:
                await ws.send_text(msg)
            except WebSocketDisconnect:
                self.disconnect(ws)

class Wav2LipService:
    def __init__(self, ckpt_path: pathlib.Path, pads: Tuple[int, int, int, int]):
        self.device     = DEVICE
        self.img_size   = 96
        self.pads       = pads
        self.batch      = WAV2LIP_BATCH
        self.static     = STATIC_FACE
        self.fast_blend = FAST_BLEND
        self.fp16       = USE_FP16 and torch.cuda.is_available()
        self.use_pipe   = USE_PIPE
        self.model      = self._load_model(str(ckpt_path))
        # Model ısındırma: sentetik forward-pass
        with torch.inference_mode():
            dummy_img = torch.randn(1, 6, self.img_size, self.img_size, device=self.device)
            dummy_mel = torch.randn(1, 1, 80, 16, device=self.device)
            _ = self.model(dummy_mel, dummy_img)
        logging.info("Model GPU’da ısındı (ilk çağrı gecikmesi ≈ 0).")
        self.frames, self.fps = self._load_frames(str(FACE_VIDEO_PATH))
        self.boxes_cache      = self._detect_faces(self.frames)
    def _load_model(self, ckpt: str):
        state = torch.load(ckpt, map_location=self.device)
        sd = state["state_dict"] if "state_dict" in state else state
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model = Wav2Lip().to(self.device).eval()
        model.load_state_dict(sd, strict=False)
        return model
    def _warm_model(self):
        with torch.inference_mode():
            dummy_img = torch.randn(1, 6, self.img_size, self.img_size, device=self.device)
            dummy_mel = torch.randn(1, 1, 80, 16, device=self.device)
            _ = self.model(dummy_mel, dummy_img)
        logging.info("Model GPU’da ısındı (ilk çağrı gecikmesi ≈ 0).")
    def _load_frames(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Video açılamadı: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            frames.append(fr)
        cap.release()
        if not frames:
            raise RuntimeError("Video boş")
        return frames, fps
    def _detect_faces(self, frames: List[np.ndarray]):
        detector = mp.solutions.face_detection.FaceDetection(1, 0.5)
        h, w = frames[0].shape[:2]
        pt, pr, pb, pl = self.pads
        if self.static:
            r = detector.process(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
            if not r.detections:
                raise RuntimeError("İlk karede yüz bulunamadı.")
            bx = r.detections[0].location_data.relative_bounding_box
            x, y, bw, bh = int(bx.xmin*w), int(bx.ymin*h), int(bx.width*w), int(bx.height*h)
            y1, y2 = max(0, y-pt), min(h, y+bh+pb)
            x1, x2 = max(0, x-pl), min(w, x+bw+pr)
            return [(x1, y1, x2, y2)] * len(frames)
        out = []
        for fr in frames:
            r = detector.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if r.detections:
                bx = r.detections[0].location_data.relative_bounding_box
                x, y, bw, bh = int(bx.xmin*w), int(bx.ymin*h), int(bx.width*w), int(bx.height*h)
                y1, y2 = max(0, y-pt), min(h, y+bh+pb)
                x1, x2 = max(0, x-pl), min(w, x+bw+pr)
                out.append((x1, y1, x2, y2))
            else:
                out.append(out[-1] if out else (0, 0, w, h))
        return out
    def make_mp4(self, wav_b64: str) -> bytes:
        """base64 WAV → dudak senkronlu MP4 baytları (_datagen yerine _pack_batch ile, boxes_cache doğrudan kullanılır)."""
        tmp_dir = pathlib.Path(tempfile.mkdtemp())
        wav_path = tmp_dir / "audio.wav"
        wav_path.write_bytes(base64.b64decode(wav_b64))

        mels = self._mels(str(wav_path), self.fps)
        if not mels:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return b""

        # Her zaman cache kullan: frame ve box uzunluklarını mels'e göre ayarla
        if len(self.frames) == 1:
            frames = self.frames * len(mels)
            boxes  = self.boxes_cache * len(mels)
        else:
            n = min(len(self.frames), len(mels))
            frames, boxes, mels = self.frames[:n], self.boxes_cache[:n], mels[:n]

        h, w = frames[0].shape[:2]
        raw_mp4 = tmp_dir / "video_raw.mp4"

        # ffmpeg sürecini hazırla
        if self.use_pipe:
            cmd = [
                imageio_ffmpeg.get_ffmpeg_exe(), "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
                "-r", str(self.fps), "-i", "-",
                "-c:v", "libx264", "-preset", FFMPEG_PRESET,
                "-crf", str(FFMPEG_CRF), "-pix_fmt", "yuv420p",
                "-an", str(raw_mp4)
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT)
        else:
            proc = cv2.VideoWriter(
                str(raw_mp4),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps, (w, h)
            )

        # --- _pack_batch ile tek seferde batch işle ---
        ib, mb, fb, cb = [], [], [], []
        single = len(frames) == 1
        if single:
            f0, b0 = frames[0], boxes[0]
            face0 = cv2.resize(f0[b0[1]:b0[3], b0[0]:b0[2]], (self.img_size, self.img_size))
        for i, m in enumerate(mels):
            idx  = 0 if single else i
            face = face0 if single else cv2.resize(
                frames[idx][boxes[idx][1]:boxes[idx][3], boxes[idx][0]:boxes[idx][2]],
                (self.img_size, self.img_size))
            ib.append(face)
            mb.append(m)
            fb.append(frames[idx].copy())
            cb.append(boxes[idx])
        t_img, t_mel, f_batch, c_batch = self._pack_batch(ib, mb, fb, cb)
        with torch.cuda.amp.autocast(enabled=self.fp16):
            with torch.inference_mode():
                pred = self.model(t_mel, t_img)
        pred = (pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
        for p, orig, coord in zip(pred, f_batch, c_batch):
            out_fr = self._blend(orig, p, coord)
            if self.use_pipe:
                proc.stdin.write(out_fr.tobytes())
            else:
                proc.write(out_fr)

        if self.use_pipe:
            proc.stdin.close(); proc.wait()
        else:
            proc.release()

        # ses + video mux (fragmented MP4)
        final_mp4 = tmp_dir / "final.mp4"
        cmd_mux = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y",
            "-i", str(raw_mp4), "-i", str(wav_path),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-movflags", "frag_keyframe+empty_moov+default_base_moof+faststart",
            "-shortest", str(final_mp4)
        ]
        subprocess.run(cmd_mux,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.STDOUT,
                       check=True)

        data = final_mp4.read_bytes()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return data
    # ... (diğer yardımcı fonksiyonlar ve make_mp4 aynen taşınacak) ...

def pcm_to_wav_b64(pcm_b64: str, sample_rate: int = 24000) -> str:
    pcm_bytes = base64.b64decode(pcm_b64)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcm") as f:
        f.write(pcm_bytes)
    wav_path = pathlib.Path(f.name).with_suffix(".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
         "-i", f.name, str(wav_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    out = base64.b64encode(wav_path.read_bytes()).decode()
    os.remove(f.name); os.remove(wav_path)
    return out

# --- FastAPI app ve endpointler ---
fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mgr   = WSManager()
queue: asyncio.Queue[str] = asyncio.Queue()
svc: "Wav2LipService | None" = None   # type: ignore

@fastapi_app.on_event("startup")
async def _startup():
    global svc
    svc = Wav2LipService(CHECKPOINT_PATH, PADS)
    asyncio.create_task(worker())
    logging.info("Wav2Lip servis kuyruğu dinleniyor.")

async def worker():
    segment_count = 0
    while True:
        wav_b64 = await queue.get()
        segment_count += 1
        total = 0
        try:
            logging.info(f"Processing segment #{segment_count}...")
            for seg in stream_wav_chunks(base64.b64decode(wav_b64)):
                total += 1
                seg_b64 = base64.b64encode(seg).decode()
                mp4 = await asyncio.to_thread(svc.make_mp4, seg_b64)
                if mp4:
                    await mgr.broadcast_video(mp4)
                    logging.info(f"   ↳ Segment #{segment_count}, {total}. sub-segment yollandı.")
        except Exception:
            logging.exception(f"Segment #{segment_count} işleme hatası")
        finally:
            await mgr.broadcast_done(total)
            logging.info(f"⏹  Segment #{segment_count} tamamlandı ({total} sub-segment).")

@fastapi_app.websocket("/wav2lip/ws/command")
async def command_ws(ws: WebSocket):
    if ws.client.host not in {"127.0.0.1", "localhost"}:
        await ws.close(); return
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if "audio_b64" not in msg: continue
            b64 = (pcm_to_wav_b64(msg["audio_b64"])
                   if msg.get("audio_format") == "audio/pcm"
                   else msg["audio_b64"])
            await queue.put(b64)
            await ws.send_text(json.dumps({"status": "queued"}))
    except WebSocketDisconnect:
        pass

@fastapi_app.websocket("/wav2lip/ws/video_stream")
async def video_ws(ws: WebSocket):
    await mgr.connect(ws)
    try:
        while True:
            await asyncio.sleep(3600)
    except WebSocketDisconnect:
        mgr.disconnect(ws)

@fastapi_app.post("/wav2lip/upload_wav")
async def upload_wav(request: Request):
    data = await request.json()
    if data.get("audio_format") != "audio/wav":
        return {"status": "error", "error": "audio_format audio/wav olmalı"}
    segment_name = data.get("segment_name", "unknown_segment")
    logging.info(f"Received WAV segment: {segment_name}, size: {len(data['audio_b64'])} chars")
    await queue.put(data["audio_b64"])
    return {"status": "queued", "segment": segment_name}

@fastapi_app.post("/wav2lip/upload_pcm")
async def upload_pcm(request: Request):
    data = await request.json()
    if data.get("audio_format") != "audio/pcm":
        return {"status": "error", "error": "audio_format audio/pcm olmalı"}
    segment_name = data.get("segment_name", "unknown_segment")
    sample_rate = data.get("sample_rate", 24000)
    logging.info(f"Received PCM segment: {segment_name}, size: {len(data['audio_b64'])} chars, sample_rate: {sample_rate}")
    wav_b64 = pcm_to_wav_b64(data["audio_b64"], sample_rate)
    await queue.put(wav_b64)
    return {"status": "queued", "segment": segment_name}

@fastapi_app.get("/wav2lip/download_pcm")
async def download_pcm():
    return FileResponse(
        "deneme_gelen_pcm.raw",
        media_type="application/octet-stream",
        filename="deneme_gelen_pcm.raw"
    )

# --- FastAPI app'i aiohttp'ya mount et ---
from aiohttp_asgi import ASGIApp


print("DEBUG: proxy.py - Starting script...")  # Add print here


HOST = "us-central1-aiplatform.googleapis.com"
SERVICE_URL = f"wss://{HOST}/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"

DEBUG = True

# Track active connections
active_connections = set()

# Weather API configuration
OPENWEATHER_API_KEY = '07e2ffbd63bb3cfbd8b0f27a4dd93104'

# Her bağlantı için ses buffer'ı
connection_audio_buffers = {}


async def get_access_token():
    """Retrieves the access token for the currently authenticated account."""
    try:
        # Cloud Run'da service account dosyası container içinde olacak
        SERVICE_ACCOUNT_FILE = "voice-asistant-459013-29c675d43902.json"
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(Request())
        print("Kullanılan service account:", creds.service_account_email)
        return creds.token
    except Exception as e:
        print(f"Error getting access token: {e}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        raise


async def send_audio_to_wav2lip(base64_audio, format_type="audio/pcm"):
    """
    wav2lip-main klasöründeki streaming_server_mp4_perviewer.py WebSocket sunucusuna ses verisini gönderir.
    """
    import websockets
    import json
    try:
        uri = "ws://localhost:9000/ws/command"  # wav2lip sunucusunun adresi
        async with websockets.connect(uri) as ws:
            message = {
                "audio_b64": base64_audio,
                "audio_format": format_type
            }
            await ws.send(json.dumps(message))
            response = await ws.recv()
            print("[Wav2Lip] Yanıt:", response)
    except Exception as e:
        print(f"[Wav2Lip] Gönderim hatası: {e}")


async def proxy_task(
    source_websocket: WebSocketCommonProtocol,
    target_websocket: WebSocketCommonProtocol,
    name: str = "",
) -> None:
    """
    Forwards messages from one WebSocket connection to another.
    """
    # Her bağlantı için buffer anahtarı olarak id kullan
    buffer_id = id(source_websocket)
    connection_audio_buffers[buffer_id] = []
    try:
        async for message in source_websocket:
            try:
                data = json.loads(message)

                # Log message type for debugging
                if "setup" in data:
                    print(f"{name} forwarding setup message")
                    print(f"Setup message content: {json.dumps(data, indent=2)}")
                elif "realtime_input" in data:
                    # Kullanıcıdan gelen chunk'lar Wav2Lip'e asla gönderilmesin, bu blok boş bırakıldı.
                    pass
                elif "serverContent" in data:
                    has_audio = False
                    # Model yanıtında inlineData varsa, sesi yakala
                    try:
                        parts = data["serverContent"].get("modelTurn", {}).get("parts", [])
                        if parts and "inlineData" in parts[0] and "data" in parts[0]["inlineData"]:
                            base64_audio = parts[0]["inlineData"]["data"]
                            has_audio = True
                            print(f"{name} - Model yanıtı sesi biriktiriliyor.")
                            connection_audio_buffers[buffer_id].append(base64_audio)
                    except Exception as e:
                        print(f"[Wav2Lip] Model yanıtı ses yakalama hatası: {e}")
                elif "client_content" in data:
                    turn_complete = False
                    if isinstance(data["client_content"], dict):
                        turn_complete = data["client_content"].get("turn_complete", False)
                    if turn_complete:
                        print(f"{name} - Konuşma tamamlandı, model sesi WAV'a dönüştürülüp wav2lip'e gönderiliyor.")
                        # Tüm chunk'ları birleştir
                        all_audio = b''
                        for b64 in connection_audio_buffers[buffer_id]:
                            all_audio += base64.b64decode(b64)
                        if all_audio:
                            print("[Proxy] turn_complete tetiklendi")
                            print(f"[Proxy] Biriken chunk sayısı: {len(connection_audio_buffers[buffer_id])}")
                            try:
                                with open("deneme_gelen_pcm.raw", "wb") as f:
                                    f.write(all_audio)
                                print("[Proxy] PCM dosyası kaydedildi: deneme_gelen_pcm.raw")
                                # Bildirimi async olarak gönder
                                try:
                                    base64_pcm = base64.b64encode(all_audio).decode()
                                    msg = json.dumps({
                                        "pcm_download": True,
                                        "filename": "deneme_gelen_pcm.raw",
                                        "pcm_base64": base64_pcm
                                    })
                                    if hasattr(source_websocket, 'send'):
                                        await source_websocket.send(msg)
                                    elif hasattr(source_websocket, 'send_str'):
                                        await source_websocket.send_str(msg)
                                    print("[Proxy] PCM indirme bildirimi gönderildi.")
                                except Exception as e:
                                    print(f"[Proxy] Bildirim gönderilemedi: {e}")
                            except Exception as e:
                                print(f"[Proxy] PCM kaydı veya bildirim hatası: {e}")
                        connection_audio_buffers[buffer_id] = []
                else:
                    print(f"{name} forwarding message type: {list(data.keys())}")
                    print(f"Message content: {json.dumps(data, indent=2)}")

                # Forward the message
                try:
                    await target_websocket.send(json.dumps(data))
                except Exception as e:
                    print(f"\n{name} Error sending message:")
                    print("=" * 80)
                    print(f"Error details: {str(e)}")
                    print("=" * 80)
                    print(f"Message that failed: {json.dumps(data, indent=2)}")
                    raise

            except websockets.exceptions.ConnectionClosed as e:
                print(f"\n{name} connection closed during message processing:")
                print("=" * 80)
                print(f"Close code: {e.code}")
                print(f"Close reason (full):")
                print("-" * 40)
                print(e.reason)
                print("=" * 80)
                break
            except Exception as e:
                print(f"\n{name} Error processing message:")
                print("=" * 80)
                print(f"Error details: {str(e)}")
                print(f"Full traceback:\n{traceback.format_exc()}")
                print("=" * 80)

    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n{name} connection closed:")
        print("=" * 80)
        print(f"Close code: {e.code}")
        print(f"Close reason (full):")
        print("-" * 40)
        print(e.reason)
        print("=" * 80)
    except Exception as e:
        print(f"\n{name} Error:")
        print("=" * 80)
        print(f"Error details: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        print("=" * 80)
    finally:
        # Clean up connections when done
        print(f"{name} cleaning up connection")
        if target_websocket in active_connections:
            active_connections.remove(target_websocket)
        try:
            await target_websocket.close()
        except:
            pass


async def create_proxy(
    client_websocket, bearer_token: str
) -> None:
    """
    Establishes a WebSocket connection to the server and creates two tasks for
    bidirectional message forwarding between the client and the server.
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer_token}",
        }

        print(f"Connecting to {SERVICE_URL}")
        async with websockets.connect(
            SERVICE_URL,
            extra_headers=headers,
            ssl=ssl.create_default_context(cafile=certifi.where()),
        ) as server_websocket:
            print("Connected to Vertex AI")
            active_connections.add(server_websocket)

            # Create bidirectional proxy tasks
            client_to_server = asyncio.create_task(
                proxy_task_aiohttp(client_websocket, server_websocket, "Client->Server")
            )
            server_to_client = asyncio.create_task(
                proxy_task_websockets(server_websocket, client_websocket, "Server->Client")
            )

            try:
                # Wait for both tasks to complete
                await asyncio.gather(client_to_server, server_to_client)
            except Exception as e:
                print(f"Error during proxy operation: {e}")
                print(f"Full traceback: {traceback.format_exc()}")
            finally:
                # Clean up tasks
                for task in [client_to_server, server_to_client]:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

    except Exception as e:
        print(f"Error creating proxy connection: {e}")
        print(f"Full traceback: {traceback.format_exc()}")


async def proxy_task_aiohttp(
    source_websocket,
    target_websocket: WebSocketCommonProtocol,
    name: str = "",
) -> None:
    """
    Forwards messages from aiohttp WebSocket to websockets WebSocket.
    """
    buffer_id = id(source_websocket)
    connection_audio_buffers[buffer_id] = []
    try:
        async for message in source_websocket:
            try:
                if message.type == web.WSMsgType.TEXT:
                    data = json.loads(message.data)
                elif message.type == web.WSMsgType.BINARY:
                    data = json.loads(message.data.decode())
                else:
                    continue

                # Log message type for debugging
                if "setup" in data:
                    print(f"{name} forwarding setup message")
                    print(f"Setup message content: {json.dumps(data, indent=2)}")
                elif "realtime_input" in data:
                    # Kullanıcıdan gelen chunk'lar Wav2Lip'e asla gönderilmesin, bu blok boş bırakıldı.
                    pass
                elif "serverContent" in data:
                    has_audio = False
                    try:
                        parts = data["serverContent"].get("modelTurn", {}).get("parts", [])
                        if parts and "inlineData" in parts[0] and "data" in parts[0]["inlineData"]:
                            base64_audio = parts[0]["inlineData"]["data"]
                            has_audio = True
                            print(f"{name} - Model yanıtı sesi biriktiriliyor.")
                            connection_audio_buffers[buffer_id].append(base64_audio)
                    except Exception as e:
                        print(f"[Wav2Lip] Model yanıtı ses yakalama hatası: {e}")
                elif "client_content" in data:
                    turn_complete = False
                    if isinstance(data["client_content"], dict):
                        turn_complete = data["client_content"].get("turn_complete", False)
                    if turn_complete:
                        print(f"{name} - Konuşma tamamlandı, model sesi WAV'a dönüştürülüp wav2lip'e gönderiliyor.")
                        all_audio = b''
                        for b64 in connection_audio_buffers[buffer_id]:
                            all_audio += base64.b64decode(b64)
                        if all_audio:
                            print("[Proxy] turn_complete tetiklendi")
                            print(f"[Proxy] Biriken chunk sayısı: {len(connection_audio_buffers[buffer_id])}")
                            try:
                                with open("deneme_gelen_pcm.raw", "wb") as f:
                                    f.write(all_audio)
                                print("[Proxy] PCM dosyası kaydedildi: deneme_gelen_pcm.raw")
                                # Bildirimi async olarak gönder
                                try:
                                    base64_pcm = base64.b64encode(all_audio).decode()
                                    msg = json.dumps({
                                        "pcm_download": True,
                                        "filename": "deneme_gelen_pcm.raw",
                                        "pcm_base64": base64_pcm
                                    })
                                    if hasattr(source_websocket, 'send'):
                                        await source_websocket.send(msg)
                                    elif hasattr(source_websocket, 'send_str'):
                                        await source_websocket.send_str(msg)
                                    print("[Proxy] PCM indirme bildirimi gönderildi.")
                                except Exception as e:
                                    print(f"[Proxy] Bildirim gönderilemedi: {e}")
                            except Exception as e:
                                print(f"[Proxy] PCM kaydı veya bildirim hatası: {e}")
                        connection_audio_buffers[buffer_id] = []
                else:
                    print(f"{name} forwarding message type: {list(data.keys())}")
                    print(f"Message content: {json.dumps(data, indent=2)}")

                # Forward the message
                try:
                    await target_websocket.send(json.dumps(data))
                except Exception as e:
                    print(f"\n{name} Error sending message:")
                    print("=" * 80)
                    print(f"Error details: {str(e)}")
                    print("=" * 80)
                    print(f"Message that failed: {json.dumps(data, indent=2)}")
                    raise

            except websockets.exceptions.ConnectionClosed as e:
                print(f"\n{name} connection closed during message processing:")
                print("=" * 80)
                print(f"Close code: {e.code}")
                print(f"Close reason (full):")
                print("-" * 40)
                print(e.reason)
                print("=" * 80)
                break
            except Exception as e:
                print(f"\n{name} Error processing message:")
                print("=" * 80)
                print(f"Error details: {str(e)}")
                print(f"Full traceback:\n{traceback.format_exc()}")
                print("=" * 80)

    except Exception as e:
        print(f"\n{name} Error:")
        print("=" * 80)
        print(f"Error details: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        print("=" * 80)
    finally:
        # Clean up connections when done
        print(f"{name} cleaning up connection")
        if target_websocket in active_connections:
            active_connections.remove(target_websocket)
        try:
            await target_websocket.close()
        except:
            pass


async def proxy_task_websockets(
    source_websocket: WebSocketCommonProtocol,
    target_websocket,
    name: str = "",
) -> None:
    """
    Forwards messages from websockets WebSocket to aiohttp WebSocket.
    """
    buffer_id = id(source_websocket)
    if buffer_id not in connection_audio_buffers:
        connection_audio_buffers[buffer_id] = []
    try:
        async for message in source_websocket:
            try:
                data = json.loads(message)

                # Log message type for debugging
                if "setup" in data:
                    print(f"{name} forwarding setup message")
                    print(f"Setup message content: {json.dumps(data, indent=2)}")
                elif "realtime_input" in data:
                    print(f"{name} forwarding audio/video input")
                elif "serverContent" in data:
                    has_audio = False
                    # Model yanıtında inlineData varsa, sesi yakala
                    try:
                        parts = data["serverContent"].get("modelTurn", {}).get("parts", [])
                        if parts and "inlineData" in parts[0] and "data" in parts[0]["inlineData"]:
                            base64_audio = parts[0]["inlineData"]["data"]
                            has_audio = True
                            print(f"{name} - Model yanıtı sesi biriktiriliyor.")
                            connection_audio_buffers[buffer_id].append(base64_audio)
                    except Exception as e:
                        print(f"[Wav2Lip] Model yanıtı ses yakalama hatası: {e}")
                else:
                    print(f"{name} forwarding message type: {list(data.keys())}")
                    print(f"Message content: {json.dumps(data, indent=2)}")

                # Forward the message
                try:
                    await target_websocket.send_str(json.dumps(data))
                except Exception as e:
                    print(f"\n{name} Error sending message:")
                    print("=" * 80)
                    print(f"Error details: {str(e)}")
                    print("=" * 80)
                    print(f"Message that failed: {json.dumps(data, indent=2)}")
                    raise

            except websockets.exceptions.ConnectionClosed as e:
                print(f"\n{name} connection closed during message processing:")
                print("=" * 80)
                print(f"Close code: {e.code}")
                print(f"Close reason (full):")
                print("-" * 40)
                print(e.reason)
                print("=" * 80)
                break
            except Exception as e:
                print(f"\n{name} Error processing message:")
                print("=" * 80)
                print(f"Error details: {str(e)}")
                print(f"Full traceback:\n{traceback.format_exc()}")
                print("=" * 80)

    except websockets.exceptions.ConnectionClosed as e:
        print(f"\n{name} connection closed:")
        print("=" * 80)
        print(f"Close code: {e.code}")
        print(f"Close reason (full):")
        print("-" * 40)
        print(e.reason)
        print("=" * 80)
    except Exception as e:
        print(f"\n{name} Error:")
        print("=" * 80)
        print(f"Error details: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        print("=" * 80)
    finally:
        # Clean up connections when done
        print(f"{name} cleaning up connection")
        if source_websocket in active_connections:
            active_connections.remove(source_websocket)
        try:
            await source_websocket.close()
        except:
            pass


async def handle_client(request):
    """
    Handles a new client connection.
    """
    print("New WebSocket connection...")
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    try:
        # Get auth token automatically
        bearer_token = await get_access_token()
        print("Retrieved bearer token automatically")

        # Send auth complete message to client
        await ws.send_json({"authComplete": True})
        print("Sent auth complete message")

        print("Creating proxy connection")
        await create_proxy(ws, bearer_token)

    except asyncio.TimeoutError:
        print("Timeout in handle_client")
        await ws.close(code=1008, message=b"Auth timeout")
    except Exception as e:
        print(f"Error in handle_client: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        await ws.close(code=1011, message=str(e).encode())
    
    return ws


async def cleanup_connections() -> None:
    """
    Periodically clean up stale connections
    """
    while True:
        print(f"Active connections: {len(active_connections)}")
        for conn in list(active_connections):
            try:
                await conn.ping()
            except:
                print("Found stale connection, removing...")
                active_connections.remove(conn)
                try:
                    await conn.close()
                except:
                    pass
        await asyncio.sleep(30)  # Check every 30 seconds


async def get_weather_data(city):
    """Hava durumu verilerini OpenWeatherMap API'sinden alır"""
    try:
        # Önce şehir için koordinatları al
        geo_url = f"https://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={OPENWEATHER_API_KEY}"
        async with aiohttp.ClientSession() as session:
            async with session.get(geo_url) as response:
                if response.status != 200:
                    return {"error": f"Geo API failed with status: {response.status}"}
                geo_data = await response.json()

        if not geo_data:
            return {"error": f"Could not find location: {city}"}

        lat, lon = geo_data[0]['lat'], geo_data[0]['lon']

        # Sonra hava durumu verilerini al
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}"
        async with aiohttp.ClientSession() as session:
            async with session.get(weather_url) as response:
                if response.status != 200:
                    return {"error": f"Weather API failed with status: {response.status}"}
                weather_data = await response.json()

        return {
            "temperature": weather_data['main']['temp'],
            "description": weather_data['weather'][0]['description'],
            "humidity": weather_data['main']['humidity'],
            "windSpeed": weather_data['wind']['speed'],
            "city": weather_data['name'],
            "country": weather_data['sys']['country']
        }
    except Exception as e:
        return {"error": f"Error fetching weather for {city}: {str(e)}"}


async def weather_handler(request):
    """Hava durumu endpoint handler'ı"""
    try:
        # CORS headers
        response = web.Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        
        if request.method == 'OPTIONS':
            response.status = 200
            return response

        # Şehir parametresini al
        city = request.query.get('city')
        if not city:
            response.status = 400
            response.text = json.dumps({"error": "City parameter is required"})
            response.content_type = 'application/json'
            return response

        # Hava durumu verilerini al
        weather_data = await get_weather_data(city)
        
        response.text = json.dumps(weather_data)
        response.content_type = 'application/json'
        return response
        
    except Exception as e:
        response = web.Response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.status = 500
        response.text = json.dumps({"error": f"Internal server error: {str(e)}"})
        response.content_type = 'application/json'
        return response


async def main() -> None:
    """
    Starts the WebSocket server and HTTP server.
    """
    print(f"DEBUG: proxy.py - main() function started")
    # Cloud Run'da PORT environment variable'ını kullan
    port = int(os.environ.get("PORT", 8080))

    # Start the cleanup task
    cleanup_task = asyncio.create_task(cleanup_connections())

    # HTTP ve WebSocket sunucusu için app oluştur
    app = web.Application()
    app.router.add_get('/weather', weather_handler)
    app.router.add_post('/weather', weather_handler)
    app.router.add_options('/weather', weather_handler)
    
    # WebSocket handler'ı ekle
    app.router.add_get('/ws', handle_client)

    # FastAPI app'i aiohttp'ya mount et
    fastapi_bridge = ASGIApp(fastapi_app)
    app.router.add_route('*', '/wav2lip/{tail:.*}', fastapi_bridge)

    # Sunucuyu başlat
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    print(f"HTTP and WebSocket server running on 0.0.0.0:{port}...")

    try:
        await asyncio.Future()  # run forever
    finally:
        cleanup_task.cancel()
        # Close all remaining connections
        for conn in list(active_connections):
            try:
                await conn.close()
            except:
                pass
        active_connections.clear()
        # Stop HTTP server
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
