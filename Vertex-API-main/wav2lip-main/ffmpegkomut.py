import os
import platform
import subprocess

# ffmpeg'in beklenen kurulum yolu
ffmpeg_dir = r"C:\ffmpeg\bin"
ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")

# ffmpeg var mı kontrol et
if os.path.isfile(ffmpeg_exe):
    # PATH'e ekle
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
    print(f"ffmpeg bulundu ve PATH'e eklendi: {ffmpeg_dir}")
else:
    print(f"Uyarı: ffmpeg bulunamadı. Lütfen ffmpeg'i {ffmpeg_dir} klasörüne yükleyin veya PATH'e ekleyin.")

# ffmpeg komutunun kullanılacağı final_command
final_command = f'ffmpeg -y -i "{audio_path}" -i "temp/result.avi" -strict -2 -q:v 1 "{outfile}"'
subprocess.call(final_command, shell=(platform.system() != "Windows"))
