import numpy as np

def normalizar(senal: np.ndarray) -> np.ndarray:
    if len(senal.shape) > 1:
        senal = np.mean(senal, axis=1)
    senal = senal - np.mean(senal)
    std = np.std(senal)
    if std > 0:
        senal = senal / std
    return senal

def calcular_espectro_directo(senal: np.ndarray) -> np.ndarray:
    fft_vals = np.fft.fft(senal)
    N = len(fft_vals)
    mitad = N // 2
    espectro = np.abs(fft_vals[:mitad]) / N
    return espectro