import numpy as np

def normalizar_rms(senal: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(senal ** 2))
    return senal / (rms + 1e-10)

def autocovarianza_discreta(senal: np.ndarray) -> np.ndarray:
    N = len(senal)
    senal_centrada = senal - np.mean(senal)
    fft_centrada = np.fft.fft(senal_centrada)
    espectro_potencia = np.abs(fft_centrada) ** 2
    resultado = np.real(np.fft.ifft(espectro_potencia)) / N
    return resultado

def calcular_fft(senal: np.ndarray) -> np.ndarray:
    return np.fft.fft(senal)

def calcular_magnitud(fft_vals: np.ndarray, sample_rate: int) -> np.ndarray:
    N = len(fft_vals)
    mitad = N // 2
    espectro = np.abs(fft_vals[:mitad])
    return espectro