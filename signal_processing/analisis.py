import numpy as np
from scipy.signal import correlate


def calcular_autocorrelacion(senal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
   
    N = len(senal)

    std_val = np.std(senal, ddof=0)

    # Protección: si la señal es silencio total (std=0), evitar división por cero
    if std_val == 0:
        lags = np.arange(-(N - 1), N)
        return lags, np.zeros(2 * N - 1)

    senal_norm = senal / std_val

    
    resultado = correlate(senal_norm, senal_norm, mode='full')

   
    resultado = resultado / N

        lags = np.arange(-(N - 1), N)

    return lags, resultado


def calcular_autocovarianza(senal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
   
    media = np.mean(senal)
    senal_centrada = senal - media

    
    lags, resultado = calcular_autocorrelacion(senal_centrada)

    return lags, resultado


def calcular_espectro(senal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    
    N = len(senal)

    
    fft_resultado = np.fft.fft(senal)

    
    espectro = np.abs(fft_resultado)

    
    mitad = N // 2
    espectro = espectro[:mitad]

    frecuencias = np.fft.fftfreq(N, d=1.0 / sample_rate)
    frecuencias = frecuencias[:mitad]

    return frecuencias, espectro



def extraer_patrones(senal: np.ndarray, sample_rate: int) -> dict:

    frecuencias, espectro = calcular_espectro(senal, sample_rate)

    energia = np.sum(espectro ** 2) / len(espectro)

    media = np.mean(espectro)

    std = np.std(espectro)

    return {
        "energia":     energia,
        "media":       media,
        "std":         std,
        "espectro":    espectro,
        "frecuencias": frecuencias,
    }