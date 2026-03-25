"""
signal_processing/analisis.py

Módulo de análisis matemático de señales de audio.
Implementa autocovarianza discreta, FFT y cálculo de magnitud espectral.

Este módulo NO lee archivos ni toca hardware.
Siempre recibe un array numpy y devuelve resultados matemáticos.

NO se usa autocorrelación en ninguna parte del sistema.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# AUTOCOVARIANZA DISCRETA
# ─────────────────────────────────────────────────────────────────────────────

def autocovarianza_discreta(senal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la autocovarianza discreta de una señal.

    Fórmula:
        C_XX(τ) = (1/N) · Σ (x[n] - μ)(x[n+τ] - μ)   para n = 0..N-τ-1

    Propiedad clave:
        - Ruido blanco:  C_XX(τ) ≈ 0 para todo τ ≠ 0
        - Señal emisora: C_XX(τ) tiene estructura (valores distintos de cero)

    Parámetros
    ----------
    senal : np.ndarray
        Array 1D de muestras de audio (float).

    Retorna
    -------
    lags : np.ndarray
        Eje de desplazamientos τ = [0, 1, 2, ..., N-1]
    resultado : np.ndarray
        C_XX(τ) para cada lag. Mismo tamaño que senal.
    """
    N = len(senal)
    media = np.mean(senal)
    senal_centrada = senal - media

    resultado = np.zeros(N)

    for tau in range(N):
        resultado[tau] = np.sum(
            senal_centrada[:N - tau] * senal_centrada[tau:]
        ) / N

    lags = np.arange(N)

    return lags, resultado


# ─────────────────────────────────────────────────────────────────────────────
# FFT
# ─────────────────────────────────────────────────────────────────────────────

def calcular_fft(senal: np.ndarray) -> np.ndarray:
    """
    Calcula la Transformada Rápida de Fourier de la señal.

    Parámetros
    ----------
    senal : np.ndarray
        Array 1D de muestras de audio (float).

    Retorna
    -------
    fft_vals : np.ndarray
        Array de N números complejos.
    """
    return np.fft.fft(senal)


# ─────────────────────────────────────────────────────────────────────────────
# MAGNITUD DEL ESPECTRO
# ─────────────────────────────────────────────────────────────────────────────

def calcular_magnitud(fft_vals: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la magnitud del espectro a partir de los valores de la FFT.

    Solo devuelve la mitad positiva porque la FFT de una señal real
    es simétrica — la segunda mitad es espejo de la primera.

    Parámetros
    ----------
    fft_vals    : np.ndarray  Salida de calcular_fft (números complejos)
    sample_rate : int         Tasa de muestreo en Hz (44100)

    Retorna
    -------
    frecuencias : np.ndarray  Frecuencias en Hz de 0 hasta sample_rate/2
    espectro    : np.ndarray  Magnitud en cada frecuencia
    """
    N = len(fft_vals)
    mitad = N // 2

    espectro    = np.abs(fft_vals[:mitad])
    frecuencias = np.fft.fftfreq(N, d=1.0 / sample_rate)[:mitad]

    return frecuencias, espectro