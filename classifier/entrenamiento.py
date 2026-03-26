"""
classifier/entrenamiento.py

Fase de entrenamiento del clasificador.
Se ejecuta UNA SOLA VEZ con los 200 audios pregrabados.

Resultado: dos archivos .npy en data/patterns/ que representan
el espectro promedio de cada clase (ruido blanco y emisora).

Todos los audios deben ser exactamente 2 segundos a 44100 Hz
→ 88200 muestras por audio.
"""

import os
import numpy as np
import soundfile as sf   # reemplaza a librosa — más limpio para WAV

from signal_processing.analisis import (
    normalizar_rms,
    autocovarianza_discreta,
    calcular_fft,
    calcular_magnitud,
)

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
    """
    Lee un archivo .wav y devuelve un array numpy mono de exactamente
    N_MUESTRAS (88200) valores float32.

    Se usa soundfile en lugar de librosa porque es más ligero y directo
    para archivos WAV, sin dependencias adicionales de decodificación.

    Si el archivo tiene múltiples canales, se extrae el primero para forzar mono.
    Si tiene más o menos muestras de las esperadas, se recorta o rellena.

    Parámetros
    ----------
    ruta_archivo : str
        Ruta completa al archivo .wav

    Retorna
    -------
    senal : np.ndarray
        Array 1D de 88200 muestras float32.
    """
    senal, sr = sf.read(ruta_archivo, dtype='float32')

    # Si el archivo tiene múltiples canales, extraemos solo el primero
    if senal.ndim > 1:
        senal = senal[:, 0]

    # Advertir si el sample rate del archivo no coincide con el esperado
    if sr != SAMPLE_RATE:
        print(f"    ⚠️  {os.path.basename(ruta_archivo)}: "
              f"sample rate {sr} Hz (esperado {SAMPLE_RATE} Hz)")

    # Garantizar exactamente N_MUESTRAS
    if len(senal) > N_MUESTRAS:
        senal = senal[:N_MUESTRAS]
    elif len(senal) < N_MUESTRAS:
        # Rellenar con ceros al final (no debería pasar con audios de 2s exactos)
        senal = np.pad(senal, (0, N_MUESTRAS - len(senal)))

    return senal


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL: procesar una carpeta completa
# ─────────────────────────────────────────────────────────────────────────────

def _procesar_carpeta(carpeta: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Lee todos los .wav de una carpeta, calcula autocovarianza + espectro
    para cada uno, y retorna el espectro promedio.

    Pipeline por audio:
        señal → autocovarianza_discreta → calcular_fft → calcular_magnitud

    Parámetros
    ----------
    carpeta : str
        Ruta a la carpeta con los archivos .wav

    Retorna
    -------
    espectro_promedio : np.ndarray
        Espectro promedio punto a punto de todos los audios.
        Shape: (N_MUESTRAS // 2,)

    frecuencias : np.ndarray
        Frecuencias en Hz del eje X del espectro.
        Shape: (N_MUESTRAS // 2,)
    """
    archivos = sorted([
        os.path.join(carpeta, f)
        for f in os.listdir(carpeta)
        if f.endswith(".wav")
    ])

    if len(archivos) == 0:
        raise FileNotFoundError(f"No se encontraron archivos .wav en: {carpeta}")

    print(f"  → Procesando {len(archivos)} archivos en: {carpeta}")

    matriz_espectros = np.zeros((len(archivos), N_MUESTRAS // 2))

    # Se inicializa en None para detectar si ningún audio llegó a procesarse
    # correctamente. Si al final sigue en None significa que el bucle nunca
    # completó un audio sin error, y np.save fallaría con un mensaje confuso.
    # Por eso se verifica explícitamente antes de guardar.
    frecuencias = None

    # ── NOTA SOBRE AUTOCOVARIANZA PROMEDIO ───────────────────────────────────
    # La autocovarianza se calcula por audio como paso intermedio del pipeline,
    # pero el vector resultante NO se guarda ni se usa directamente en la
    # clasificación. El clasificador opera comparando espectros promedio
    # (dominio frecuencial). Guardar la autocovarianza en bruto no aplica
    # para este laboratorio.
    # ─────────────────────────────────────────────────────────────────────────

    for i, ruta in enumerate(archivos):
        print(f"    [{i+1}/{len(archivos)}] {os.path.basename(ruta)}")

        # 1. Leer el audio como array numpy mono float32
        senal = _leer_audio(ruta)

        # 2. Normalizar por RMS para que la escala sea independiente
        #    del volumen o ganancia del micrófono con el que se grabó
        senal = normalizar_rms(senal)

        # 3. Autocovarianza discreta (Wiener-Khinchin, sin np.correlate)
        #    C_XX(τ) ≈ 0 para τ≠0 si es ruido blanco → espectro plano
        #    C_XX(τ) tiene estructura si es emisora   → espectro con picos
        _, autocov_vals = autocovarianza_discreta(senal)

        # 3. FFT aplicada sobre la AUTOCOVARIANZA (no sobre la señal cruda)
        #    Por Wiener-Khinchin, esto entrega el espectro de potencia,
        #    que es la representación descriptiva que queremos comparar.
        fft_vals = calcular_fft(autocov_vals)

        # 4. Magnitud del espectro (mitad positiva, parte simétrica descartada)
        frecuencias, espectro = calcular_magnitud(fft_vals, SAMPLE_RATE)
        matriz_espectros[i] = espectro

    # Si frecuencias sigue siendo None, ningún archivo se procesó correctamente.
    # Se lanza un error explícito para evitar un crash confuso en np.save.
    if frecuencias is None:
        raise RuntimeError(
            f"No se pudo procesar ningún archivo en: {carpeta}. "
            "Revisa que los archivos .wav sean válidos y tengan el formato correcto."
        )

    # Promedio punto a punto entre todos los espectros de la carpeta
    # axis=0 → promedio por columna (por cada punto del espectro)
    espectro_promedio = np.mean(matriz_espectros, axis=0)

    return espectro_promedio, frecuencias


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
# Esta transformación aleja linealmente cada patrón del punto medio entre
# los dos espectros. En un clasificador por DISTANCIA MÍNIMA (el que usamos),
# escalar simétricamente ambos patrones no cambia cuál queda más cerca de la
# señal nueva: el resultado de clasificación es idéntico con o sin calibración.
#
# ¿Cómo hacerla realmente útil?
# Tendría sentido recalcular el umbral de decisión con audios de validación
# reales (que no estuvieron en el entrenamiento), medir cuántos quedan mal
# clasificados cerca del límite y ajustar el factor dinámicamente. En ese
# caso dejaría de ser una escala arbitraria y pasaría a ser una corrección
# basada en error empírico.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def entrenar():
    """
    Ejecuta el entrenamiento completo con los 200 audios pregrabados.

    Lee las carpetas data/recordings/ruido_blanco/ y data/recordings/senal_radio/,
    calcula los espectros promedio de cada clase y los guarda en data/patterns/.
    """
    os.makedirs(CARPETA_PAT, exist_ok=True)

    print("═" * 50)
    print("ENTRENAMIENTO — Ruido blanco")
    print("═" * 50)
    esp_rb, frecuencias = _procesar_carpeta(CARPETA_RB)

    print()
    print("═" * 50)
    print("ENTRENAMIENTO — Emisora")
    print("═" * 50)
    esp_em, _ = _procesar_carpeta(CARPETA_EM)

    # Guardar patrones en data/patterns/
    np.save(os.path.join(CARPETA_PAT, "espectro_promedio_rb.npy"), esp_rb)
    np.save(os.path.join(CARPETA_PAT, "espectro_promedio_em.npy"), esp_em)
    np.save(os.path.join(CARPETA_PAT, "frecuencias.npy"),          frecuencias)

    print()
    print("✅ Entrenamiento completado. Archivos guardados en data/patterns/:")
    print("   - espectro_promedio_rb.npy")
    print("   - espectro_promedio_em.npy")
    print("   - frecuencias.npy")


if __name__ == "__main__":
    entrenar()