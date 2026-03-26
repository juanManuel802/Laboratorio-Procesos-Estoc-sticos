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
# NORMALIZACIÓN DE AMPLITUD
# ─────────────────────────────────────────────────────────────────────────────

def normalizar_rms(senal: np.ndarray) -> np.ndarray:
    """
    Normaliza la amplitud de una señal dividiendo por su RMS (Root Mean Square).

    Por qué es necesario:
        La autocovarianza es proporcional a amplitud². Si dos grabaciones del
        mismo tipo de señal tienen distinta ganancia de micrófono o volumen,
        sus espectros de potencia diferirán en escala pero NO en forma.
        Al normalizar por RMS, todas las señales se procesan como si tuvieran
        la misma "energía", haciendo que el pipeline sea independiente del
        nivel de grabación.

    El +1e-10 en el denominador evita división por cero cuando la señal
    es silencio total (RMS ≈ 0).

    Parámetros
    ----------
    senal : np.ndarray  Array 1D de muestras de audio (float).

    Retorna
    -------
    senal_norm : np.ndarray  Señal con RMS unitario. Mismo shape que senal.
    """
    rms = np.sqrt(np.mean(senal ** 2))
    return senal / (rms + 1e-10)

# ─────────────────────────────────────────────────────────────────────────────
# AUTOCOVARIANZA DISCRETA
# ─────────────────────────────────────────────────────────────────────────────

def autocovarianza_discreta(senal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la autocovarianza discreta de una señal usando el teorema de
    Wiener-Khinchin.

    Fórmula equivalente:
        C_XX(τ) = (1/N) · Σ (x[n] - μ)(x[n+τ] - μ)   para n = 0..N-τ-1

    Implementación:
        En lugar de calcular los N desplazamientos con un bucle (lento),
        se usa la relación de Wiener-Khinchin:

            C_XX = IFFT( |FFT(x - μ)|² ) / N

        La FFT de la señal centrada, elevada al cuadrado en magnitud, da el
        espectro de potencia. La IFFT de ese espectro de potencia devuelve
        directamente la autocovarianza. Resultado matemáticamente idéntico
        al bucle, pero ejecutado en C por numpy sin ninguna iteración Python.

        NO se usa np.correlate ni ninguna función de autocorrelación.

    Propiedad clave:
        - Ruido blanco:  C_XX(τ) ≈ 0 para todo τ ≠ 0  →  espectro plano
        - Señal emisora: C_XX(τ) tiene estructura      →  espectro con picos

    Parámetros
    ----------
    senal : np.ndarray
        Array 1D de muestras de audio (float).

    Retorna
    -------
    lags : np.ndarray
        Eje de desplazamientos τ = [0, 1, 2, ..., N-1]
    resultado : np.ndarray
        C_XX(τ) para cada lag. Mismo tamaño que senal. Shape: (N,)
    """
    N = len(senal)

    # 1. Centrar la señal (restar la media elimina el componente DC)
    senal_centrada = senal - np.mean(senal)

    # 2. FFT de la señal centrada
    fft_centrada = np.fft.fft(senal_centrada)

    # 3. Espectro de potencia: |FFT|²
    #    Cada punto representa la energía de la señal en esa frecuencia
    espectro_potencia = np.abs(fft_centrada) ** 2

    # 4. IFFT del espectro de potencia → autocovarianza (Wiener-Khinchin)
    #    np.real() descarta residuos imaginarios mínimos por precisión numérica
    resultado = np.real(np.fft.ifft(espectro_potencia)) / N

    lags = np.arange(N)

    return lags, resultado


# ─────────────────────────────────────────────────────────────────────────────
# FFT
# ─────────────────────────────────────────────────────────────────────────────

def calcular_fft(senal: np.ndarray) -> np.ndarray:
    """
    Calcula la Transformada Rápida de Fourier de la señal.

    Se aplica sobre el vector resultante de autocovarianza_discreta,
    no sobre la señal de audio cruda. Esto entrega el espectro de potencia
    en dominio frecuencial, que es la representación que se compara entre
    clases en el clasificador.

    Parámetros
    ----------
    senal : np.ndarray
        Array 1D (resultado de autocovarianza_discreta). Shape: (N,)

    Retorna
    -------
    fft_vals : np.ndarray
        Array de N números complejos. Shape: (N,)
    """
    return np.fft.fft(senal)


# ─────────────────────────────────────────────────────────────────────────────
# MAGNITUD DEL ESPECTRO
# ─────────────────────────────────────────────────────────────────────────────

def calcular_magnitud(fft_vals: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la magnitud del espectro a partir de los valores de la FFT.

    Solo devuelve la mitad positiva porque la FFT de una señal real
    es simétrica: la segunda mitad es espejo de la primera y no
    aporta información adicional.

    Parámetros
    ----------
    fft_vals    : np.ndarray  Salida de calcular_fft (números complejos). Shape: (N,)
    sample_rate : int         Tasa de muestreo en Hz (ej. 44100)

    Retorna
    -------
    frecuencias : np.ndarray  Frecuencias en Hz de 0 hasta sample_rate/2. Shape: (N//2,)
    espectro    : np.ndarray  Magnitud en cada frecuencia.                 Shape: (N//2,)
    """
    N = len(fft_vals)
    mitad = N // 2

    espectro    = np.abs(fft_vals[:mitad])
    frecuencias = np.fft.fftfreq(N, d=1.0 / sample_rate)[:mitad]

    return frecuencias, espectro