import numpy as np
from statsmodels.tsa.stattools import acovf



def normalizar(senalmean: np.ndarray) -> np.ndarray:
    if len(senalmean.shape) > 1: 
        senalmean = np.mean(senalmean, axis = 1)
    senalmean = np.mean(senalmean)
    senalmean = senalmean - np.mean(senalmean)
    return senalmean

def autocovarianza_discreta(senal: np.ndarray) -> np.ndarray:
    resultado = acovf(senal, fft=True, demean=True).astype(np.float64)
    return resultado

def calcular_fft(senal: np.ndarray) -> np.ndarray:
    return np.fft.fft(senal)

def calcular_magnitud(fft_vals: np.ndarray, sample_rate: int) -> np.ndarray:
    N = len(fft_vals)
    mitad = N // 2
    espectro = np.abs(fft_vals[:mitad])
    return espectro