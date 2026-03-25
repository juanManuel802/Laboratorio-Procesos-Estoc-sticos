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
import librosa

from signal_processing.analisis import (
    autocovarianza_discreta,
    calcular_fft,
    calcular_magnitud,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE    = 44100
DURACION_SEG   = 2
N_MUESTRAS     = SAMPLE_RATE * DURACION_SEG  # 88200 muestras esperadas

CARPETA_RB     = "data/recordings/ruido_blanco"
CARPETA_EM     = "data/recordings/senal_radio"
CARPETA_PAT    = "data/patterns"


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN AUXILIAR: leer un .mp3 y devolver array numpy estandarizado
# ─────────────────────────────────────────────────────────────────────────────

def _leer_audio(ruta_archivo: str) -> np.ndarray:
    """
    Lee un archivo .mp3 y devuelve un array numpy mono de exactamente
    N_MUESTRAS (88200) valores float32.

    Parámetros
    ----------
    ruta_archivo : str
        Ruta completa al archivo .mp3

    Retorna
    -------
    senal : np.ndarray
        Array 1D de 88200 muestras float32.
    """
    # sr=SAMPLE_RATE  → librosa resamplea si el archivo tiene otro sample rate
    # mono=True       → convierte a un solo canal si es estéreo
    senal, _ = librosa.load(ruta_archivo, sr=SAMPLE_RATE, mono=True)

    # Garantizar exactamente N_MUESTRAS
    # Si por alguna razón el archivo tiene más o menos muestras, ajustamos
    if len(senal) > N_MUESTRAS:
        senal = senal[:N_MUESTRAS]          # recortar
    elif len(senal) < N_MUESTRAS:
        # rellenar con ceros al final (no debería pasar con audios de 2s)
        senal = np.pad(senal, (0, N_MUESTRAS - len(senal)))

    return senal


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL: procesar una carpeta completa
# ─────────────────────────────────────────────────────────────────────────────

def _procesar_carpeta(carpeta: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Lee todos los .mp3 de una carpeta, calcula autocovarianza + espectro
    para cada uno, y retorna los promedios.

    Parámetros
    ----------
    carpeta : str
        Ruta a la carpeta con los archivos .mp3

    Retorna
    -------
    espectro_promedio : np.ndarray
        Espectro promedio punto a punto de todos los audios.
        Shape: (N_MUESTRAS // 2,)

    autocov_promedio : np.ndarray
        Autocovarianza promedio punto a punto de todos los audios.
        Shape: (N_MUESTRAS,)

    frecuencias : np.ndarray
        Frecuencias en Hz del eje X del espectro.
        Shape: (N_MUESTRAS // 2,)
    """
    # Listar y ordenar todos los .mp3 de la carpeta
    archivos = sorted([
        os.path.join(carpeta, f)
        for f in os.listdir(carpeta)
        if f.endswith(".mp3")
    ])

    if len(archivos) == 0:
        raise FileNotFoundError(f"No se encontraron archivos .mp3 en: {carpeta}")

    print(f"  → Procesando {len(archivos)} archivos en: {carpeta}")

    # Matrices para acumular resultados
    # Cada fila será el espectro o autocovarianza de un audio
    matriz_espectros  = np.zeros((len(archivos), N_MUESTRAS // 2))
    matriz_autocovs   = np.zeros((len(archivos), N_MUESTRAS))
    frecuencias       = None

    for i, ruta in enumerate(archivos):
        print(f"    [{i+1}/{len(archivos)}] {os.path.basename(ruta)}")

        # 1. Leer el audio como array numpy
        senal = _leer_audio(ruta)

        # 2. Autocovarianza discreta
        # C_XX(τ) = 0 para τ≠0 si es ruido blanco
        # tiene estructura si es emisora
        _, autocov_vals = autocovarianza_discreta(senal)
        matriz_autocovs[i] = autocov_vals

        # 3. FFT
        fft_vals = calcular_fft(senal)

        # 4. Magnitud del espectro (mitad positiva)
        frecuencias, espectro = calcular_magnitud(fft_vals, SAMPLE_RATE)
        matriz_espectros[i] = espectro

    # Promediar punto a punto entre todos los audios
    # axis=0 → promedio por columna (por cada punto del espectro/autocov)
    espectro_promedio = np.mean(matriz_espectros, axis=0)
    autocov_promedio  = np.mean(matriz_autocovs,  axis=0)

    return espectro_promedio, autocov_promedio, frecuencias


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRACIÓN: ampliar separación entre patrones (máx 5%)
# ─────────────────────────────────────────────────────────────────────────────

def _calibrar(esp_rb: np.ndarray, esp_em: np.ndarray,
              porcentaje: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Amplía levemente la diferencia entre los dos espectros promedio
    para reducir el margen de error del clasificador (máximo 5%).

    La idea es alejar cada patrón del punto medio entre los dos,
    haciendo la separación entre clases más clara.

    Parámetros
    ----------
    esp_rb      : np.ndarray  Espectro promedio ruido blanco
    esp_em      : np.ndarray  Espectro promedio emisora
    porcentaje  : float       Factor de separación (default 0.05 = 5%)

    Retorna
    -------
    esp_rb_cal, esp_em_cal : np.ndarray, np.ndarray
        Espectros calibrados.
    """
    # Punto medio entre los dos patrones
    punto_medio = (esp_rb + esp_em) / 2.0

    # Alejar cada patrón del centro en el porcentaje indicado
    esp_rb_cal = esp_rb + porcentaje * (esp_rb - punto_medio)
    esp_em_cal = esp_em + porcentaje * (esp_em - punto_medio)

    return esp_rb_cal, esp_em_cal


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def entrenar(calibrar: bool = True, porcentaje_calibracion: float = 0.05):
    """
    Ejecuta el entrenamiento completo con los 200 audios pregrabados.

    Lee las carpetas data/recordings/ruido_blanco/ y data/recordings/senal_radio/,
    calcula los espectros promedio de cada clase y los guarda en data/patterns/.

    Parámetros
    ----------
    calibrar               : bool   Si True, aplica calibración al final.
    porcentaje_calibracion : float  Porcentaje de separación (default 0.05).
    """
    os.makedirs(CARPETA_PAT, exist_ok=True)

    print("═" * 50)
    print("ENTRENAMIENTO — Ruido blanco")
    print("═" * 50)
    esp_rb, autocov_rb, frecuencias = _procesar_carpeta(CARPETA_RB)

    print()
    print("═" * 50)
    print("ENTRENAMIENTO — Emisora")
    print("═" * 50)
    esp_em, autocov_em, _ = _procesar_carpeta(CARPETA_EM)

    # Calibración opcional
    if calibrar:
        print()
        print(f"Aplicando calibración ({porcentaje_calibracion*100:.0f}%)...")
        esp_rb, esp_em = _calibrar(esp_rb, esp_em, porcentaje_calibracion)

    # Guardar todos los patrones en data/patterns/
    np.save(os.path.join(CARPETA_PAT, "espectro_promedio_rb.npy"),  esp_rb)
    np.save(os.path.join(CARPETA_PAT, "espectro_promedio_em.npy"),  esp_em)
    np.save(os.path.join(CARPETA_PAT, "autocov_promedio_rb.npy"),   autocov_rb)
    np.save(os.path.join(CARPETA_PAT, "autocov_promedio_em.npy"),   autocov_em)
    np.save(os.path.join(CARPETA_PAT, "frecuencias.npy"),           frecuencias)

    print()
    print("✅ Entrenamiento completado. Archivos guardados en data/patterns/:")
    print("   - espectro_promedio_rb.npy")
    print("   - espectro_promedio_em.npy")
    print("   - autocov_promedio_rb.npy")
    print("   - autocov_promedio_em.npy")
    print("   - frecuencias.npy")


# Permite correr este archivo directamente para entrenar:
# python -m classifier.entrenamiento
if __name__ == "__main__":
    entrenar()