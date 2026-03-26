"""
classifier/entrenamiento.py

Fase de entrenamiento del clasificador.
Se ejecuta UNA SOLA VEZ con los 200 audios pregrabados.

Resultado: archivos .npy en data/patterns/ que representan
el espectro promedio y la autocovarianza promedio de cada clase.

Todos los audios deben ser exactamente 2 segundos a 44100 Hz
→ 88200 muestras por audio.
"""

import os
import numpy as np
import soundfile as sf

from signal_processing.analisis import normalizar_rms,autocovarianza_discreta,calcular_fft,calcular_magnitud


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE  = 44100
DURACION_SEG = 2
N_MUESTRAS   = SAMPLE_RATE * DURACION_SEG  # 88200 muestras esperadas

CARPETA_RB   = "data/recordings/ruido_blanco"
CARPETA_EM   = "data/recordings/senal_radio"
CARPETA_PAT  = "data/patterns"


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN AUXILIAR: leer un .wav y devolver array numpy estandarizado
# ─────────────────────────────────────────────────────────────────────────────

def _leer_audio(ruta_archivo: str) -> np.ndarray:
    """Lee un archivo .wav y devuelve un array numpy mono de exactamente
    N_MUESTRAS (88200) valores float32.
    Si tiene más o menos muestras de las esperadas, se recorta o rellena."""

    senal, _ = sf.read(ruta_archivo, dtype='float32')

    if senal.ndim == 2:
        senal = senal[:, 0]   # tomar solo el canal izquierdo

    if len(senal) > N_MUESTRAS:
        senal = senal[:N_MUESTRAS]
    elif len(senal) < N_MUESTRAS:
        senal = np.pad(senal, (0, N_MUESTRAS - len(senal)))

    return senal


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL: procesar una carpeta completa
# ─────────────────────────────────────────────────────────────────────────────

def _procesar_carpeta(carpeta: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Lee todos los .wav de una carpeta, calcula autocovarianza + espectro
    para cada uno, y retorna los promedios de ambos.

    Pipeline por audio:
        señal → normalizar_rms → autocovarianza_discreta → calcular_fft → calcular_magnitud

    Parámetros
    ----------
    carpeta : str
        Ruta a la carpeta con los archivos .wav

    Retorna
    -------
    espectro_promedio : np.ndarray
        Espectro de potencia promedio punto a punto. Shape: (N_MUESTRAS // 2,)

    autocov_promedio : np.ndarray
        Autocovarianza promedio punto a punto. Shape: (N_MUESTRAS,)
        Se guarda en patterns para graficarla en la GUI.
    """
    archivos = sorted([os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.endswith(".wav")])

    print(f"  → Procesando {len(archivos)} archivos en: {carpeta}")

    matriz_espectros = np.zeros((len(archivos), N_MUESTRAS // 2))
    matriz_autocovs  = np.zeros((len(archivos), N_MUESTRAS))

    for i, ruta in enumerate(archivos):
        print(f"    [{i+1}/{len(archivos)}] {os.path.basename(ruta)}")

        # 1. Leer el audio como array numpy float32
        senal = _leer_audio(ruta)

        # 2. Normalizar por RMS
        senal = normalizar_rms(senal)

        # 3. Autocovarianza discreta (Wiener-Khinchin, sin np.correlate)
        #    C_XX(τ) ≈ 0 para τ≠0 si es ruido blanco → espectro plano
        #    C_XX(τ) tiene estructura si es emisora   → espectro con picos
        _, autocov_vals = autocovarianza_discreta(senal)
        matriz_autocovs[i] = autocov_vals

        # 4. FFT sobre la AUTOCOVARIANZA → espectro de potencia (Wiener-Khinchin)
        fft_vals = calcular_fft(autocov_vals)

        # 5. Magnitud del espectro (mitad positiva, parte simétrica descartada)
        _, espectro = calcular_magnitud(fft_vals, SAMPLE_RATE)
        matriz_espectros[i] = espectro

    espectro_promedio = np.mean(matriz_espectros, axis=0)
    autocov_promedio  = np.mean(matriz_autocovs,  axis=0)

    return espectro_promedio, autocov_promedio


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRACIÓN (desactivada)
# ─────────────────────────────────────────────────────────────────────────────
#
# def _calibrar(esp_rb, esp_em, porcentaje=0.05):
#     punto_medio = (esp_rb + esp_em) / 2.0
#     esp_rb_cal  = esp_rb + porcentaje * (esp_rb - punto_medio)
#     esp_em_cal  = esp_em + porcentaje * (esp_em - punto_medio)
#     return esp_rb_cal, esp_em_cal
#
# ¿Por qué está comentada?
# En un clasificador por DISTANCIA MÍNIMA, escalar simétricamente ambos
# patrones no cambia cuál queda más cerca de la señal nueva. El resultado
# de clasificación es idéntico con o sin calibración.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def entrenar():
    """
    Ejecuta el entrenamiento completo con los 200 audios pregrabados.
    Guarda espectro promedio y autocovarianza promedio de cada clase
    en data/patterns/.
    """
    os.makedirs(CARPETA_PAT, exist_ok=True)

    print("═" * 50)
    print("ENTRENAMIENTO — Ruido blanco")
    print("═" * 50)
    esp_rb, autocov_rb = _procesar_carpeta(CARPETA_RB)

    print()
    print("═" * 50)
    print("ENTRENAMIENTO — Emisora")
    print("═" * 50)
    esp_em, autocov_em = _procesar_carpeta(CARPETA_EM)

    np.save(os.path.join(CARPETA_PAT, "espectro_promedio_rb.npy"), esp_rb)
    np.save(os.path.join(CARPETA_PAT, "espectro_promedio_em.npy"), esp_em)
    np.save(os.path.join(CARPETA_PAT, "autocov_promedio_rb.npy"),  autocov_rb)
    np.save(os.path.join(CARPETA_PAT, "autocov_promedio_em.npy"),  autocov_em)

    print()
    print("Entrenamiento completado. Archivos guardados en data/patterns/:")
    print("   - espectro_promedio_rb.npy")
    print("   - espectro_promedio_em.npy")
    print("   - autocov_promedio_rb.npy")
    print("   - autocov_promedio_em.npy")


if __name__ == "__main__":
    entrenar()