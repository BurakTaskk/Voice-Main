/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

export class AudioStreamer {
  constructor(audioContext) {
    this.context = audioContext;
    this.sampleRate = 48000; // Output sample rate - daha yüksek kalite için
    this.audioQueue = [];
    this.isPlaying = false;
    this.currentSource = null;
    this.gainNode = this.context.createGain();
    
    // SES SEVİYESİNİ MAKSİMUMA AYARLA
    this.gainNode.gain.value = 1.0; 
    
    this.gainNode.connect(this.context.destination);
    this.addPCM16 = this.addPCM16.bind(this);
    this.onComplete = () => {};
    this.playbackTimeout = null;
    this.lastPlaybackTime = 0;
    
    // Komple ses toplama için yeni değişkenler
    this.allAudioChunks = []; // Tüm gelen ses parçaları
    this.isCollecting = true; // Ses toplama modunda mı?
    this.collectedAudioBuffer = null; // Toplanan ses buffer'ı
    
    // Audio download functionality
    this.audioChunks = []; // Store all audio chunks for download
    this.isRecording = false;
    this.isDownloading = false; // Prevent multiple downloads
  }

  addPCM16(chunk) {
    // Store audio chunk for download
    this.audioChunks.push(chunk);
    console.log(`Audio chunk received and stored. Total chunks: ${this.audioChunks.length}, chunk size: ${chunk.length} bytes`);
    
    // Eğer toplama modundaysa, chunk'ı topla
    if (this.isCollecting) {
      this.allAudioChunks.push(chunk);
      console.log(`Ses toplandı. Toplam chunk sayısı: ${this.allAudioChunks.length}`);
    } else {
      // Eğer toplama modunda değilse, direkt oynat
      this.playAudioChunk(chunk);
    }
  }

  playAudioChunk(chunk) {
    // Convert incoming PCM16 data to float32
    const float32Array = new Float32Array(chunk.length / 2);
    const dataView = new DataView(chunk.buffer);

    for (let i = 0; i < chunk.length / 2; i++) {
      try {
        const int16 = dataView.getInt16(i * 2, true);
        float32Array[i] = int16 / 32768;
      } catch (e) {
        console.error(e);
      }
    }

    // Create and fill audio buffer
    const audioBuffer = this.context.createBuffer(1, float32Array.length, this.sampleRate);
    audioBuffer.getChannelData(0).set(float32Array);

    // Add to queue and start playing if needed
    this.audioQueue.push(audioBuffer);
    
    if (!this.isPlaying) {
      this.isPlaying = true;
      this.lastPlaybackTime = this.context.currentTime;
      this.playNextBuffer();
    }

    // Ensure playback continues if it was interrupted
    this.checkPlaybackStatus();
  }

  // Toplanan tüm sesi oynat
  playCollectedAudio() {
    if (this.allAudioChunks.length === 0) {
      console.log('Oynatılacak ses yok');
      return;
    }

    console.log(`Toplanan ${this.allAudioChunks.length} chunk oynatılıyor...`);
    
    // Tüm chunk'ları birleştir
    const totalLength = this.allAudioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const combinedChunk = new Uint8Array(totalLength);
    let offset = 0;
    
    for (const chunk of this.allAudioChunks) {
      combinedChunk.set(chunk, offset);
      offset += chunk.length;
    }
    
    // Birleştirilmiş sesi oynat
    this.playAudioChunk(combinedChunk);
    
    // Toplama modunu kapat
    this.isCollecting = false;
  }

  checkPlaybackStatus() {
    // Clear any existing timeout
    if (this.playbackTimeout) {
      clearTimeout(this.playbackTimeout);
    }

    // Set a new timeout to check playback status
    this.playbackTimeout = setTimeout(() => {
      const now = this.context.currentTime;
      const timeSinceLastPlayback = now - this.lastPlaybackTime;

      // If more than 1 second has passed since last playback and we have buffers to play
      if (timeSinceLastPlayback > 1 && this.audioQueue.length > 0 && this.isPlaying) {
        console.log('Playback appears to have stalled, restarting...');
        this.playNextBuffer();
      }

      // Continue checking if we're still playing
      if (this.isPlaying) {
        this.checkPlaybackStatus();
      }
    }, 1000);
  }

  playNextBuffer() {
    if (this.audioQueue.length === 0) {
      this.isPlaying = false;
      return;
    }

    // Update last playback time
    this.lastPlaybackTime = this.context.currentTime;

    try {
      const audioBuffer = this.audioQueue.shift();
      const source = this.context.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.gainNode);

      // Önceki sesi kesme - sadece referansı güncelle
      this.currentSource = source;

      // When this buffer ends, play the next one
      source.onended = () => {
        this.lastPlaybackTime = this.context.currentTime;
        if (this.audioQueue.length > 0) {
          // Smooth transition için kısa gecikme
          setTimeout(() => this.playNextBuffer(), 5);
        } else {
          this.isPlaying = false;
          this.onComplete();
        }
      };

      // Start playing immediately
      source.start(0);
    } catch (error) {
      console.error('Error during playback:', error);
      // Try to recover by playing next buffer
      if (this.audioQueue.length > 0) {
        setTimeout(() => this.playNextBuffer(), 100);
      } else {
        this.isPlaying = false;
      }
    }
  }

  stop() {
    this.isPlaying = false;
    if (this.playbackTimeout) {
      clearTimeout(this.playbackTimeout);
      this.playbackTimeout = null;
    }
    if (this.currentSource) {
      try {
        this.currentSource.stop();
        this.currentSource.disconnect();
      } catch (e) {
        // Ignore if already stopped
      }
    }
    this.audioQueue = [];
    this.allAudioChunks = []; // Toplanan ses parçalarını temizle
    this.isCollecting = true; // Toplama modunu sıfırla
    this.gainNode.gain.linearRampToValueAtTime(0, this.context.currentTime + 0.1);

    setTimeout(() => {
      this.gainNode.disconnect();
      this.gainNode = this.context.createGain();
      this.gainNode.connect(this.context.destination);
    }, 200);
    
    // Clear audio chunks when stopping
    this.audioChunks = [];
  }

  async resume() {
    if (this.context.state === 'suspended') {
      await this.context.resume();
    }
    this.lastPlaybackTime = this.context.currentTime;
    // SES SEVİYESİNİ MAKSİMUMA AYARLA
    this.gainNode.gain.setValueAtTime(1.0, this.context.currentTime);
    
    // Eğer toplama modunda değilse ve queue'da ses varsa oynat
    if (!this.isCollecting && this.audioQueue.length > 0 && !this.isPlaying) {
      this.isPlaying = true;
      this.playNextBuffer();
    }
  }

  complete() {
    console.log('AudioStreamer complete() called');
    console.log('Audio chunks available:', this.audioChunks.length);
    console.log('Total audio data size:', this.audioChunks.reduce((sum, chunk) => sum + chunk.length, 0), 'bytes');
    
    // Clear playback timeout
    if (this.playbackTimeout) {
      clearTimeout(this.playbackTimeout);
      this.playbackTimeout = null;
    }
    
    // Toplanan sesi oynat
    if (this.allAudioChunks.length > 0 && this.isCollecting) {
      console.log(`Toplanan ${this.allAudioChunks.length} chunk oynatılıyor...`);
      this.playCollectedAudio();
    }
    
    // If there are still audio chunks to download, proceed with download
    if (this.audioChunks.length > 0) {
      console.log('Audio chunks available for download, starting download...');
      this.downloadAudio();
    } else {
      console.log('No audio chunks available for download');
    }
    
    // Set a timeout to ensure download happens even if queue doesn't empty
    setTimeout(() => {
      if (this.audioChunks.length > 0 && !this.isDownloading) {
        console.log('Forcing download after timeout...');
        this.downloadAudio();
      }
    }, 2000);
    
    this.onComplete();
  }

  // New method to download audio
  downloadAudio() {
    console.log('downloadAudio() called - starting audio download process');
    
    if (this.isDownloading) {
      console.log('Download already in progress, skipping...');
      return;
    }
    
    if (this.audioChunks.length === 0) {
      console.log('No audio chunks to download');
      return;
    }

    console.log(`Starting download of ${this.audioChunks.length} audio chunks...`);
    this.isDownloading = true;

    // Show download status
    this.showDownloadStatus();
    
    // Start timing the conversion
    const startTime = performance.now();

    try {
      // Combine all audio chunks
      const totalLength = this.audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
      console.log(`Combining ${this.audioChunks.length} chunks, total length: ${totalLength} bytes`);
      
      const combinedAudio = new Uint8Array(totalLength);
      let offset = 0;
      
      for (const chunk of this.audioChunks) {
        combinedAudio.set(chunk, offset);
        offset += chunk.length;
      }

      // Create WAV file
      console.log('Creating WAV file...');
      const wavBuffer = this.createWAVFile(combinedAudio);
      console.log('WAV file created, size:', wavBuffer.byteLength, 'bytes');
      
      // Calculate conversion time
      const endTime = performance.now();
      const conversionTime = ((endTime - startTime) / 1000).toFixed(2);
      
      // Create download link
      console.log('Creating download link...');
      const blob = new Blob([wavBuffer], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `gemini_response_${new Date().toISOString().replace(/[:.]/g, '-')}.wav`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      console.log(`Audio downloaded successfully in ${conversionTime} seconds`);
      
      // Update download status with conversion time
      this.updateDownloadStatus(`Ses dosyası indirildi (${conversionTime}s)`);
      
      // Show browser notification
      this.showBrowserNotification(conversionTime);
      
      // Send to Wav2Lip for lip sync
      console.log('About to send to Wav2Lip...');
      this.sendToWav2Lip(wavBuffer);
      
      // Clear chunks after download
      this.audioChunks = [];
      
      // Hide download status after a longer delay to show the time
      setTimeout(() => {
        this.hideDownloadStatus();
      }, 3000);
    } catch (error) {
      console.error('Error downloading audio:', error);
      this.hideDownloadStatus();
    } finally {
      this.isDownloading = false;
    }
  }

  // New method to send audio to Wav2Lip
  async sendToWav2Lip(wavBuffer) {
    const base64Audio = await arrayBufferToBase64(wavBuffer);
    // Hem 8080 hem 8000 portuna paralel gönder
    const ports = [8080, 8000];
    let successCount = 0;
    let errorCount = 0;
    const sendPromises = ports.map(port => {
      return new Promise((resolve, reject) => {
        const wsUrl = `ws://localhost:${port}/wav2lip/ws/command`;
    const ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      ws.send(JSON.stringify({
        audio_b64: base64Audio,
            audio_format: "audio/wav"
      }));
    };
    ws.onmessage = (evt) => {
      const res = JSON.parse(evt.data);
      if (res.status === 'queued') {
        ws.close();
            resolve(port);
          } else if (res.status === 'error') {
            ws.close();
            reject(new Error(`Port ${port}: Wav2Lip kuyruğuna eklenemedi`));
          }
        };
        ws.onerror = (e) => {
          reject(new Error(`Port ${port}: WebSocket error`));
        };
        ws.onclose = () => {};
      });
    });
    try {
      const results = await Promise.allSettled(sendPromises);
      results.forEach(r => {
        if (r.status === 'fulfilled') successCount++;
        else errorCount++;
      });
      if (successCount > 0) {
        this.showWav2LipSendNotification();
        if (typeof window.autoStartWav2LipVideo === 'function') {
          window.autoStartWav2LipVideo();
        }
      }
      if (errorCount > 0) {
        console.warn(`Wav2Lip gönderiminde ${errorCount} hata oluştu.`);
      }
    } catch (err) {
      console.error('Wav2Lip gönderim hatası:', err);
    }
  }

  // Bildirim fonksiyonu: Wav2Lip'e gönderim başarılı
  showWav2LipSendNotification() {
    // Tarayıcı bildirimi
    if ('Notification' in window) {
      if (Notification.permission === 'default') {
        Notification.requestPermission().then(permission => {
          if (permission === 'granted') {
            this.createWav2LipSendNotification();
          }
        });
      } else if (Notification.permission === 'granted') {
        this.createWav2LipSendNotification();
      }
    }
    // Ekranda kısa süreli mesaj (toast benzeri)
    this.showWav2LipSuccessMessage();
      }

  createWav2LipSendNotification() {
    const notification = new Notification('Wav2Lip Kuyruğuna Gönderildi!', {
      body: 'Ses dosyası başarıyla Wav2Lip kuyruğuna eklendi. Dudak senkronizasyonu başlatılıyor.',
      icon: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDkuNzRMMTIgMTZMMTAuOTEgOS43NEw0IDlMMTAuOTEgOC4yNkwxMiAyWiIgZmlsbD0iIzQyYzM0MiIvPgo8L3N2Zz4K',
      tag: 'wav2lip-send-success'
    });
    setTimeout(() => notification.close(), 4000);
  }

  showWav2LipSuccessMessage() {
    // Ekranda kısa süreli başarı mesajı göster
    const successDiv = document.createElement('div');
    successDiv.id = 'wav2lipSendSuccess';
    successDiv.innerHTML = `
      <div class="success-message" style="z-index:9999;position:fixed;bottom:32px;right:32px;background:#4caf50;color:#fff;padding:16px 24px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.15);display:flex;align-items:center;gap:12px;">
        <span class="material-symbols-outlined" style="font-size:28px;">check_circle</span>
        <div>
          <div style="font-weight:bold;">Wav2Lip Kuyruğuna Gönderildi</div>
          <div style="font-size:14px;">Ses dosyası başarıyla gönderildi. Dudak senkronizasyonu başlatılıyor.</div>
        </div>
        <button style="background:none;border:none;color:#fff;font-size:20px;margin-left:12px;cursor:pointer;" onclick="this.parentElement.parentElement.remove()">×</button>
      </div>
    `;
    document.body.appendChild(successDiv);
    setTimeout(() => {
      if (successDiv.parentElement) {
        successDiv.remove();
      }
    }, 5000);
  }

  // New method to auto-start Wav2Lip video stream
  autoStartWav2LipVideo() {
    console.log('autoStartWav2LipVideo() called');
    
    // Yeni Wav2LipPlayer sistemini kullan
    if (typeof window.startWav2LipStream === 'function') {
      window.startWav2LipStream();
      console.log('Wav2Lip video stream auto-started with new system');
    } else {
      console.error('Wav2Lip start function not found');
    }
  }

  // Helper method to create Wav2Lip section if it doesn't exist
  createWav2LipSection() {
    console.log('Creating Wav2Lip section...');
    
    const wav2lipHTML = `
      <div id="wav2lipSection" class="wav2lip-section">
        <div id="wav2lipStartOverlay" class="wav2lip-start-overlay">
          <h1>Wav2Lip Yayınına Hoş Geldiniz</h1>
          <p>Yayını başlatmak için tıklayın.</p>
          <button id="wav2lipStartButton">BAŞLAT</button>
        </div>
        
        <div id="wav2lipContent" class="wav2lip-content hidden">
          <h2>Wav2Lip Canlı Yayın</h2>
          <div id="wav2lipWrapper" class="wav2lip-wrapper">
            <video id="streamA" playsinline style="display: block; opacity: 1;"></video>
            <video id="streamB" playsinline style="display: block; opacity: 0;"></video>
            <button id="wav2lipPlayPause">⏯</button>
          </div>
          <p id="wav2lipStatus">Sunucuya bağlanılıyor…</p>
        </div>
      </div>
    `;
    
    document.body.insertAdjacentHTML('beforeend', wav2lipHTML);
    console.log('Wav2Lip section created');
  }

  // Helper method to create WAV file from PCM16 data
  createWAVFile(pcmData) {
    const buffer = new ArrayBuffer(44 + pcmData.length);
    const view = new DataView(buffer);
    
    // WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + pcmData.length, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, this.sampleRate, true);
    view.setUint32(28, this.sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, pcmData.length, true);
    
    // Copy PCM data
    const pcmView = new Uint8Array(buffer, 44);
    pcmView.set(pcmData);
    
    return buffer;
  }

  // Helper methods to show/hide download status
  showDownloadStatus() {
    const downloadStatus = document.getElementById('downloadStatus');
    if (downloadStatus) {
      downloadStatus.innerHTML = '<span class="material-symbols-outlined">download</span>WAV\'a dönüştürülüyor...';
      downloadStatus.classList.remove('hidden');
    }
  }

  hideDownloadStatus() {
    const downloadStatus = document.getElementById('downloadStatus');
    if (downloadStatus) {
      downloadStatus.classList.add('hidden');
    }
  }

  updateDownloadStatus(message) {
    const downloadStatus = document.getElementById('downloadStatus');
    if (downloadStatus) {
      downloadStatus.innerHTML = `<span class="material-symbols-outlined">check_circle</span>${message}`;
    }
  }

  showBrowserNotification(conversionTime) {
    // Check if browser supports notifications
    if ('Notification' in window) {
      // Request permission if not granted
      if (Notification.permission === 'default') {
        Notification.requestPermission().then(permission => {
          if (permission === 'granted') {
            this.createNotification(conversionTime);
          }
        });
      } else if (Notification.permission === 'granted') {
        this.createNotification(conversionTime);
      }
    }
    
    // Also show a more prominent success message
    this.showSuccessMessage(conversionTime);
  }

  createNotification(conversionTime) {
    const notification = new Notification('Ses Dosyası İndirildi! 🎵', {
      body: `Gemini yanıtı ${conversionTime} saniyede WAV formatında indirildi.`,
      icon: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDkuNzRMMTIgMTZMMTAuOTEgOS43NEw0IDlMMTAuOTEgOC4yNkwxMiAyWiIgZmlsbD0iIzQyYzM0MiIvPgo8L3N2Zz4K',
      badge: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMTMuMDkgOC4yNkwyMCA5TDEzLjA5IDkuNzRMMTIgMTZMMTAuOTEgOS43NEw0IDlMMTAuOTEgOC4yNkwxMiAyWiIgZmlsbD0iIzQyYzM0MiIvPgo8L3N2Zz4K',
      tag: 'gemini-audio-download'
    });
    
    // Auto close notification after 5 seconds
    setTimeout(() => {
      notification.close();
    }, 5000);
  }

  showSuccessMessage(conversionTime) {
    // Create a more prominent success message
    const successDiv = document.createElement('div');
    successDiv.id = 'downloadSuccess';
    successDiv.innerHTML = `
      <div class="success-message">
        <span class="material-symbols-outlined">check_circle</span>
        <div class="success-text">
          <div class="success-title">Ses Dosyası İndirildi! 🎵</div>
          <div class="success-details">Downloads klasörüne kaydedildi (${conversionTime}s)</div>
        </div>
        <button class="close-btn" onclick="this.parentElement.parentElement.remove()">×</button>
      </div>
    `;
    
    document.body.appendChild(successDiv);
    
    // Auto remove after 8 seconds
    setTimeout(() => {
      if (successDiv.parentElement) {
        successDiv.remove();
      }
    }, 8000);
  }
}

// Helper: ArrayBuffer to base64
async function arrayBufferToBase64(buf) {
  const blob = new Blob([new Uint8Array(buf)], {type: 'audio/wav'});
  const dataUrl = await new Promise(r => {
    const fr = new FileReader();
    fr.onload = () => r(fr.result);
    fr.readAsDataURL(blob);
  });
  return dataUrl.split(',')[1];
}