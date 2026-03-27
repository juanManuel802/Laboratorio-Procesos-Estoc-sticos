import os
import numpy as np
import soundfile as sf

from signal_processing.analisis import normalizar_rms,autocovarianza_discreta,calcular_fft,calcular_magnitud

SAMPLE_RATE  = 44100
DURACION_SEG = 2
N_MUESTRAS   = SAMPLE_RATE * DURACION_SEG  # 88200 muestras esperadas

CARPETA_RB   = "data/recordings/ruido_blanco"
CARPETA_EM   = "data/recordings/senal_radio"
CARPETA_PAT  = "data/patterns"

def _leer_audio(ruta_archivo: str) -> np.ndarray:

    senal, _ = sf.read(ruta_archivo, dtype='float32')

    if senal.ndim == 2:
        senal = senal[:, 0]   # tomar solo el canal izquierdo

    if len(senal) > N_MUESTRAS:
        senal = senal[:N_MUESTRAS]
    elif len(senal) < N_MUESTRAS:
        senal = np.pad(senal, (0, N_MUESTRAS - len(senal)))

    return senal

def _procesar_carpeta(carpeta: str) -> tuple[np.ndarray, np.ndarray]:

    archivos = sorted([os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.endswith(".wav")])

    print(f"  → Procesando {len(archivos)} archivos en: {carpeta}")

    matriz_espectros = np.zeros((len(archivos), N_MUESTRAS // 2))
    matriz_autocovs  = np.zeros((len(archivos), N_MUESTRAS))

    for i, ruta in enumerate(archivos):
        print(f"    [{i+1}/{len(archivos)}] {os.path.basename(ruta)}")

        senal = _leer_audio(ruta)

        senal = normalizar_rms(senal)

        _, autocov_vals = autocovarianza_discreta(senal)
        matriz_autocovs[i] = autocov_vals

        fft_vals = calcular_fft(autocov_vals)

        _, espectro = calcular_magnitud(fft_vals, SAMPLE_RATE)
        matriz_espectros[i] = espectro

    espectro_promedio = np.mean(matriz_espectros, axis=0)
    autocov_promedio  = np.mean(matriz_autocovs,  axis=0)

    return espectro_promedio, autocov_promedio

def entrenar():
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