"""
signal_processing/analisis.py

Módulo de análisis matemático de señales de audio.
Implementa autocorrelación, autocovarianza y análisis espectral (FFT).

Este módulo NO lee archivos ni toca hardware.
Siempre recibe un array numpy y devuelve resultados matemáticos.
"""

import numpy as np
from scipy.signal import correlate


# ─────────────────────────────────────────────────────────────────────────────
# AUTOCORRELACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def calcular_autocorrelacion(senal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la autocorrelación normalizada de una señal discreta.

    La autocorrelación mide qué tanto se parece una señal con una versión
    desplazada (lag) de sí misma. Responde: ¿tiene memoria esta señal?

    - Ruido blanco:    pico alto solo en lag=0, casi cero en los demás.
    - Señal de radio:  picos periódicos en varios lags (tiene estructura).

    Parámetros
    ----------
    senal : np.ndarray
        Array 1D de muestras de audio (valores float).

    Retorna
    -------
    lags : np.ndarray
        Eje de desplazamientos temporales. Va de -(N-1) a (N-1).
        lag=0 es el centro, donde la señal se compara consigo misma.
    resultado : np.ndarray
        Valores de autocorrelación para cada lag. Siempre entre -1 y 1
        gracias a la normalización.
    """
    N = len(senal)

    # ── Paso 1: normalizar por desviación estándar ──────────────────────────
    # Sin esto, señales con distinto volumen tendrían autocorrelaciones
    # de distinta magnitud, haciéndolas incomparables entre sí.
    # std_val con ddof=0 usa N en el denominador (población completa).
    std_val = np.std(senal, ddof=0)

    # Protección: si la señal es silencio total (std=0), evitar división por cero
    if std_val == 0:
        lags = np.arange(-(N - 1), N)
        return lags, np.zeros(2 * N - 1)

    senal_norm = senal / std_val

    # ── Paso 2: correlación de la señal consigo misma ───────────────────────
    # mode='full' calcula todos los posibles desplazamientos.
    # El resultado tiene longitud 2N - 1.
    resultado = correlate(senal_norm, senal_norm, mode='full')

    # ── Paso 3: normalizar por N ─────────────────────────────────────────────
    # correlate() devuelve la SUMA de productos para cada lag.
    # Dividir por N la convierte en PROMEDIO, independiente de la duración.
    resultado = resultado / N

    # ── Paso 4: construir el eje de lags ─────────────────────────────────────
    # lag negativo → señal desplazada hacia atrás en el tiempo
    # lag=0        → comparación directa consigo misma (máximo)
    # lag positivo → señal desplazada hacia adelante en el tiempo
    lags = np.arange(-(N - 1), N)

    return lags, resultado


# ─────────────────────────────────────────────────────────────────────────────
# AUTOCOVARIANZA
# ─────────────────────────────────────────────────────────────────────────────

def calcular_autocovarianza(senal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la autocovarianza normalizada de una señal discreta.

    Idéntica a la autocorrelación pero trabajando sobre la señal centrada
    (con la media restada). Elimina el componente DC (offset del micrófono
    u otras fuentes) para medir únicamente la estructura real de la señal.

    Relación matemática:
        C_XX(τ) = R_XX(τ) - μ²
    donde μ es la media de la señal. En la práctica, centrar la señal
    antes de llamar a calcular_autocorrelacion produce el mismo resultado.

    Parámetros
    ----------
    senal : np.ndarray
        Array 1D de muestras de audio (valores float).

    Retorna
    -------
    lags : np.ndarray
        Eje de desplazamientos (igual que en autocorrelación).
    resultado : np.ndarray
        Valores de autocovarianza para cada lag.
    """
    # ── Centrar la señal: restar la media ────────────────────────────────────
    # Esto elimina cualquier offset constante.
    # Ejemplo: un micrófono con bias produce muestras como [0.5, 0.51, 0.49].
    # Centrada queda [0.0, 0.01, -0.01], que es la variación real.
    media = np.mean(senal)
    senal_centrada = senal - media

    # La autocovarianza es la autocorrelación de la señal centrada
    lags, resultado = calcular_autocorrelacion(senal_centrada)

    return lags, resultado


# ─────────────────────────────────────────────────────────────────────────────
# FFT + ESPECTRO
# ─────────────────────────────────────────────────────────────────────────────

def calcular_espectro(senal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula el espectro de amplitud de una señal mediante FFT.

    La FFT descompone la señal en sus frecuencias componentes, como un
    prisma que separa la luz blanca en colores. abs() obtiene la magnitud
    (cuánta energía hay en cada frecuencia).

    - Ruido blanco:    espectro plano (energía uniforme en todas las frecuencias).
    - Señal de radio:  picos en frecuencias específicas.

    Parámetros
    ----------
    senal : np.ndarray
        Array 1D de muestras de audio (valores float).
    sample_rate : int
        Tasa de muestreo del audio en Hz (ej: 44100).
        Necesaria para convertir índices de la FFT a frecuencias reales (Hz).

    Retorna
    -------
    frecuencias : np.ndarray
        Frecuencias en Hz correspondientes a cada punto del espectro.
        Va de 0 Hz hasta sample_rate/2 Hz (frecuencia de Nyquist).
    espectro : np.ndarray
        Magnitud de cada frecuencia. Valores positivos.
    """
    N = len(senal)

    # ── Paso 1: FFT completa ─────────────────────────────────────────────────
    # Devuelve N números complejos. Cada uno representa una frecuencia.
    fft_resultado = np.fft.fft(senal)

    # ── Paso 2: magnitud (espectro de amplitud) ──────────────────────────────
    # abs() de un número complejo = sqrt(parte_real² + parte_imaginaria²)
    # Convierte los números complejos de la FFT en magnitudes reales positivas.
    espectro = np.abs(fft_resultado)

    # ── Paso 3: quedarse solo con la mitad positiva ──────────────────────────
    # La FFT de una señal real es simétrica: la segunda mitad es espejo
    # de la primera. No aporta información nueva, se descarta.
    mitad = N // 2
    espectro = espectro[:mitad]

    # ── Paso 4: calcular frecuencias reales en Hz ────────────────────────────
    # fftfreq devuelve valores normalizados entre -0.5 y 0.5.
    # Multiplicar por sample_rate convierte a Hz reales.
    # Solo tomamos la mitad positiva (misma razón que el espectro).
    frecuencias = np.fft.fftfreq(N, d=1.0 / sample_rate)
    frecuencias = frecuencias[:mitad]

    return frecuencias, espectro


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE PATRONES
# ─────────────────────────────────────────────────────────────────────────────

def extraer_patrones(senal: np.ndarray, sample_rate: int) -> dict:
    """
    Extrae los patrones estadísticos del espectro de una señal.

    Estos tres números son los que el clasificador compara contra los
    patrones guardados en data/patterns/ para decidir la clase.

    Parámetros
    ----------
    senal : np.ndarray
        Array 1D de muestras de audio.
    sample_rate : int
        Tasa de muestreo en Hz.

    Retorna
    -------
    dict con las siguientes claves:
        energia     : float  → energía promedio del espectro
        media       : float  → valor medio del espectro
        std         : float  → desviación estándar del espectro
        espectro    : np.ndarray  → para graficar en la GUI
        frecuencias : np.ndarray  → para graficar en la GUI
    """
    frecuencias, espectro = calcular_espectro(senal, sample_rate)

    # ── Energía promedio ─────────────────────────────────────────────────────
    # sum(espectro²) / N : promedio de la potencia en cada frecuencia.
    # Ruido blanco → energía distribuida, valor moderado y estable.
    # Señal de radio → energía concentrada en picos, valor más variable.
    energia = np.sum(espectro ** 2) / len(espectro)

    # ── Media del espectro ───────────────────────────────────────────────────
    # Nivel general de actividad espectral.
    media = np.mean(espectro)

    # ── Desviación estándar ──────────────────────────────────────────────────
    # El más discriminativo de los tres.
    # std pequeña → espectro plano → ruido blanco.
    # std grande  → espectro con picos → señal de radio.
    std = np.std(espectro)

    return {
        "energia":     energia,
        "media":       media,
        "std":         std,
        "espectro":    espectro,
        "frecuencias": frecuencias,
    }