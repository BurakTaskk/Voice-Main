#!/usr/bin/env python3
# ================================================================
#  stream_wav2lip.py – Ön-yüklemesiz, akışlı dudak senkronizasyonu
# ================================================================
import os, random, subprocess, tempfile, itertools
from typing import Tuple, Generator, List

import cv2, numpy as np, torch, imageio_ffmpeg
from tqdm import tqdm

# ------------------------------------------------ GPU / Determinizm
if not torch.cuda.is_available():
    raise RuntimeError("CUDA destekli GPU bulunamadı!")
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True          # ► en iyi kernel seçimi
torch.manual_seed(42); random.seed(42); np.random.seed(42)

# ------------------------------------------------ Wav2Lip modülleri
import audio                     # Wav2Lip repo içindeki utils/audio.py
from models import Wav2Lip       # Wav2Lip model tanımı

# ------------------------------------------------ MANUEL AYARLAR
CHECKPOINT_PATH      = r"checkpoints/wav2lip_gan.pth"
FACE_VIDEO_PATH      = r"C:\Users\taski\OneDrive\Desktop\wav2lip-main\asistankızpotre.mp4"     # giriş video/foto
AUDIO_PATH           = r"C:\Users\taski\OneDrive\Desktop\wav2lip-main\refses.wav"       # giriş ses
OUTPUT_VIDEO_PATH    = r"C:\Users\taski\OneDrive\Desktop\wav2lip-main\sonuc.mp4"     # çıktı mp4
PADS                 = (0, 10, 20, 10)                # yüz crop payı
STATIC_FACE          = False                          # yüz dinamik izlenir
WAV2LIP_BATCH_SIZE   = 128
USE_FP16_INFERENCE   = True
USE_FFMPEG_PIPE      = True
FFMPEG_CRF           = 23
FFMPEG_PRESET        = "ultrafast"
# ==============================================================


# ---------------------- Yardımcı akış (generator) fonksiyonları -------------
def stream_video_frames(path: str) -> Tuple[Generator[np.ndarray, None, None], float]:
    """
    Kareleri üreten generator **ve** FPS değerini döndürür.
    Tek resim verilirse FPS=25 kabul edilir.
    """
    ext = os.path.splitext(path)[1].lower()

    # Tek resim
    if ext in (".jpg", ".jpeg", ".png"):
        frame = cv2.imread(path)
        def _gen():
            yield frame
        return _gen(), 25.0

    # Video
    cap  = cv2.VideoCapture(path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0

    def _gen():
        ok, frm = cap.read()
        while ok:
            yield frm
            ok, frm = cap.read()
        cap.release()
    return _gen(), fps


def stream_mel_chunks(wav_path: str,
                      fps: float,
                      chunk_sec: float = 1.0) -> Generator[np.ndarray, None, None]:
    """
    Mel spektrogramını 1 saniyelik (varsayılan) dilimler hâlinde üretir.
    """
    wav = audio.load_wav(wav_path, 16000)
    full_mel = audio.melspectrogram(wav)              # (80, T)
    mel_step = 16                                     # Wav2Lip sabiti
    hop_size = int(chunk_sec * fps)                   # kare ↔︎ mel eşlem

    idx = 0
    while True:
        start = idx * hop_size
        end   = start + mel_step
        if end > full_mel.shape[1]:
            break
        yield full_mel[:, start:end]
        idx += 1


# ---------------------- Ana servis sınıfı ------------------------------------
class Wav2LipServiceStreaming:
    IMG_SIZE = 96

    def __init__(self,
                 ckpt_path: str,
                 batch_size: int,
                 static_face: bool,
                 fp16: bool,
                 pipe: bool):
        self.ckpt_path   = ckpt_path
        self.wav_bs      = batch_size
        self.static_face = static_face
        self.fp16        = fp16 and torch.cuda.get_device_capability()[0] >= 7
        self.pipe        = pipe

        # Lazy-load yer tutucular
        self._model     = None
        self._face_det  = None

    # ------------------ Lazy yükleyiciler -----------------------------------
    @property
    def model(self) -> Wav2Lip:
        if self._model is None:
            print("» Wav2Lip modeli yükleniyor…")
            ckpt = torch.load(self.ckpt_path, map_location=DEVICE)
            sd   = ckpt.get("state_dict", ckpt)
            sd   = {k.replace("module.", ""): v for k, v in sd.items()}
            self._model = Wav2Lip().to(DEVICE).eval()
            self._model.load_state_dict(sd)
        return self._model

    @property
    def face_det(self):
        if self._face_det is None:
            import mediapipe as mp
            self._face_det = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        return self._face_det

    # ------------------ Kamu API --------------------------------------------
    def infer(self,
              face_path: str,
              audio_path: str,
              out_path: str,
              pads: Tuple[int, int, int, int],
              ffmpeg_crf: int,
              ffmpeg_preset: str):

        print("» Akışlı dudak senkronizasyonu başlıyor…")

        vid_gen, fps  = stream_video_frames(face_path)
        first_frame   = next(vid_gen)          # ilk kare → boyut/fps tespiti

        mel_gen = stream_mel_chunks(audio_path, fps)

        # FFmpeg writer ayarı
        h, w       = first_frame.shape[:2]
        tmp_mp4    = tempfile.mktemp(suffix=".mp4")
        writer     = None
        ffmpeg_proc = None

        if self.pipe:
            cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y",
                   "-f", "rawvideo", "-vcodec", "rawvideo", "-s", f"{w}x{h}",
                   "-pix_fmt", "bgr24", "-r", str(fps), "-i", "-",
                   "-c:v", "libx264", "-preset", ffmpeg_preset,
                   "-crf", str(ffmpeg_crf), "-pix_fmt", "yuv420p",
                   "-an", tmp_mp4]
            ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                           stdout=subprocess.DEVNULL,
                                           stderr=subprocess.STDOUT)
        else:
            writer = cv2.VideoWriter(tmp_mp4,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (w, h))

        # Statik mod için ilk yüz kutusu
        p_top, p_r, p_btm, p_l = pads
        if self.static_face:
            x1, y1, x2, y2 = self._detect_face(first_frame, pads)
            static_coords  = (x1, y1, x2, y2)

        # Akışlı ana döngü
        buf_img, buf_mel, buf_frm, buf_crd = [], [], [], []
        total_frames = 0

        for frame in itertools.chain([first_frame], vid_gen):
            try:
                mel = next(mel_gen)
            except StopIteration:
                break

            if self.static_face:
                coords = static_coords
            else:
                coords = self._detect_face(frame, pads)

            face_img = self._crop_resize_face(frame, coords)

            buf_img.append(face_img)
            buf_mel.append(mel)
            buf_frm.append(frame.copy())
            buf_crd.append(coords)

            if len(buf_img) == self.wav_bs:
                self._flush_batch(buf_img, buf_mel, buf_frm, buf_crd,
                                  writer, ffmpeg_proc)
                total_frames += len(buf_img)
                buf_img, buf_mel, buf_frm, buf_crd = [], [], [], []

        # Artakalanlar
        if buf_img:
            self._flush_batch(buf_img, buf_mel, buf_frm, buf_crd,
                              writer, ffmpeg_proc)
            total_frames += len(buf_img)

        # FFmpeg / writer kapat
        if self.pipe:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        else:
            writer.release()

        # Ses–video mux
        self._mux_audio(tmp_mp4, audio_path, out_path)
        os.remove(tmp_mp4)
        print(f"» Tamamlandı ({total_frames} kare) → {out_path}")

    # ------------------ Yardımcı metotlar ------------------------------------
    def _flush_batch(self, img_b: List[np.ndarray], mel_b: List[np.ndarray],
                     frame_b: List[np.ndarray], coord_b: List[Tuple[int,int,int,int]],
                     writer, ffmpeg_proc):
        imgs = np.asarray(img_b)                        # (B,96,96,3)
        mels = np.expand_dims(np.asarray(mel_b), 3)     # (B,80,16,1)

        imgs_mask = imgs.copy(); imgs_mask[:, :, self.IMG_SIZE//2:] = 0
        imgs_in   = np.concatenate((imgs_mask, imgs), axis=3) / 255.0

        imgs_t = torch.from_numpy(imgs_in.transpose(0,3,1,2)).float().to(DEVICE)
        mels_t = torch.from_numpy(mels.transpose(0,3,1,2)).float().to(DEVICE)

        with torch.cuda.amp.autocast(enabled=self.fp16):
            with torch.no_grad():
                preds = self.model(mels_t, imgs_t)

        preds = preds.cpu().numpy().transpose(0,2,3,1) * 255.

        for p, orig, c in zip(preds, frame_b, coord_b):
            out_frame = self._blend_face(orig, p.astype(np.uint8), c)
            if self.pipe:
                ffmpeg_proc.stdin.write(out_frame.tobytes())
            else:
                writer.write(out_frame)

    def _detect_face(self, frame: np.ndarray, pads):
        h, w = frame.shape[:2]
        res  = self.face_det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.detections:
            return (0, 0, w, h)
        bbox = res.detections[0].location_data.relative_bounding_box
        p_top, p_r, p_btm, p_l = pads
        x, y, bw, bh = int(bbox.xmin*w), int(bbox.ymin*h), int(bbox.width*w), int(bbox.height*h)
        y1, y2 = max(0, y - p_top),  min(h, y + bh + p_btm)
        x1, x2 = max(0, x - p_l),    min(w, x + bw + p_r)
        return (x1, y1, x2, y2)

    def _crop_resize_face(self, frame: np.ndarray, coords):
        x1, y1, x2, y2 = coords
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            face = frame
        return cv2.resize(face, (self.IMG_SIZE, self.IMG_SIZE))

    def _blend_face(self, base: np.ndarray, new_face: np.ndarray, coords):
        x1, y1, x2, y2 = coords
        w, h = x2 - x1, y2 - y1
        if w == 0 or h == 0:
            return base
        new_face = cv2.resize(new_face, (w, h))
        base[y1:y2, x1:x2] = new_face
        return base

    def _mux_audio(self, video_path: str, audio_path: str, out_path: str):
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y",
               "-i", video_path, "-i", audio_path,
               "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
               "-shortest", out_path]
        subprocess.check_call(cmd,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.STDOUT)


# ---------------------- Ana giriş noktası ------------------------------------
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)

    service = Wav2LipServiceStreaming(
        ckpt_path   = CHECKPOINT_PATH,
        batch_size  = WAV2LIP_BATCH_SIZE,
        static_face = STATIC_FACE,
        fp16        = USE_FP16_INFERENCE,
        pipe        = USE_FFMPEG_PIPE
    )

    service.infer(FACE_VIDEO_PATH,
                  AUDIO_PATH,
                  OUTPUT_VIDEO_PATH,
                  PADS,
                  FFMPEG_CRF,
                  FFMPEG_PRESET)
