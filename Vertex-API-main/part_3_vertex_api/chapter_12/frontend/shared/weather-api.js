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

const OPENWEATHER_API_KEY = '07e2ffbd63bb3cfbd8b0f27a4dd93104';

export async function getWeather(city) {
  try {
    // Proxy sunucusundaki weather endpoint'ini kullan
    // Cloud Run'da çalışırken proxy sunucusunun URL'sini kullan
    const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    
    let proxyUrl;
    if (isLocalhost) {
      // Local development için
      proxyUrl = `http://localhost:8080/weather?city=${encodeURIComponent(city)}`;
    } else {
      // Cloud Run'da çalışırken aynı domain'i kullan
      const protocol = window.location.protocol;
      const hostname = window.location.hostname;
      const port = window.location.port;
      proxyUrl = `https://vertex-proxy-service-638345404110.us-central1.run.app/weather?city=${encodeURIComponent(city)}`;
    }
    
    console.log('Fetching weather data from proxy:', proxyUrl);
    const response = await fetch(proxyUrl);
    
    if (!response.ok) {
      throw new Error(`Weather API failed with status: ${response.status}`);
    }
    
    const weatherData = await response.json();
    
    if (weatherData.error) {
      return weatherData;
    }

    return {
      temperature: weatherData.temperature,
      description: weatherData.description,
      humidity: weatherData.humidity,
      windSpeed: weatherData.windSpeed,
      city: weatherData.city,
      country: weatherData.country
    };
  } catch (error) {
    console.error('Detailed error:', {
      message: error.message,
      stack: error.stack,
      type: error.name
    });
    return {
      error: `Error fetching weather for ${city}: ${error.message}`
    };
  }
}

export async function googleSearch(query) {
  try {
    const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(query)}`;
    window.open(searchUrl, '_blank');
    return { success: true, message: `Google'da "${query}" için arama yapılıyor...` };
  } catch (error) {
    console.error('Google arama hatası:', error);
    return { success: false, error: 'Arama yapılırken bir hata oluştu.' };
  }
}

export async function playYouTube(searchQuery) {
  try {
    // Arama sorgusunu temizle ve doğrula
    const cleanQuery = searchQuery.trim();
    if (!cleanQuery) {
      throw new Error('Geçersiz arama sorgusu');
    }

    // YouTube arama URL'sini oluştur
    const youtubeSearchUrl = `https://www.youtube.com/results?search_query=${encodeURIComponent(cleanQuery)}`;
    console.log('Opening YouTube search URL:', youtubeSearchUrl);
    
    // Yeni sekmede aç
    window.open(youtubeSearchUrl, '_blank');

    return { success: true, message: `"${cleanQuery}" için YouTube'da arama yapılıyor...` };
  } catch (error) {
    console.error('YouTube arama hatası:', error);
    return { success: false, error: error.message || 'Arama yapılırken bir hata oluştu.' };
  }
} 