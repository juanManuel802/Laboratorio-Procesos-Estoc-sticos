import os
import sys
import numpy as np

import librosa

# Agregar el directorio raíz al PATH para poder importar signal_processing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from signal_processing.analisis import autocovarianza_discreta, calcular_fft, calcular_magnitud

def cargar_audio(ruta_archivo, sr=44100):
    senal, _ = librosa.load(ruta_archivo, sr=sr, mono=True)
    return senal

def predecir_audio(ruta_audio):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dir_modelos = os.path.join(base_dir, 'data', 'patterns')
    
    ruta_ruido = os.path.join(dir_modelos, 'ruido_promedio.npy')
    ruta_emisora = os.path.join(dir_modelos, 'emisora_promedio.npy')
    
    if not os.path.exists(ruta_ruido) or not os.path.exists(ruta_emisora):
        print("Error: Modelos no encontrados. Por favor ejecuta la fase de entrenamiento primero.")
        sys.exit(1)
        
    promedio_ruido = np.load(ruta_ruido)
    promedio_emisora = np.load(ruta_emisora)
    
    sr = 44100
    try:
        senal = cargar_audio(ruta_audio, sr=sr)
    except Exception as e:
        print(f"Error cargando el archivo de audio: {e}")
        sys.exit(1)
    
    # 2. Calcular autocovarianza (SIN autocorrelación)
    lags, autocomv_val = autocovarianza_discreta(senal)
    
    # 3. FFT sobre la autocovarianza
    fft_vals = calcular_fft(autocomv_val)
    
    # 4. Magnitud espectral
    _, magnitud = calcular_magnitud(fft_vals, sample_rate=sr)
    
    # Recortar longitud al mínimo común (por pequeñas diferencias de duración)
    min_len_ruido = min(len(magnitud), len(promedio_ruido))
    min_len_emisora = min(len(magnitud), len(promedio_emisora))
    
    mag_ruido_recortado = magnitud[:min_len_ruido]
    prom_ruido_recortado = promedio_ruido[:min_len_ruido]
    
    mag_emisora_recortado = magnitud[:min_len_emisora]
    prom_emisora_recortado = promedio_emisora[:min_len_emisora]
    
    # Calcular Distancia Euclidiana Estandarizada o Suma de Diferencias
    dist_ruido = np.linalg.norm(mag_ruido_recortado - prom_ruido_recortado)
    dist_emisora = np.linalg.norm(mag_emisora_recortado - prom_emisora_recortado)
    
    print(f"--- RESULTADOS DE CLASIFICACIÓN ---")
    print(f"Distancia a modelo 'Ruido Blanco': {dist_ruido:.4f}")
    print(f"Distancia a modelo 'Emisora':      {dist_emisora:.4f}")
    
    if dist_ruido < dist_emisora:
        clase_predicha = "RUIDO BLANCO"
    else:
        clase_predicha = "EMISORA"
        
    print(f"\\n> DECISIÓN: El audio ingresado es: *** {clase_predicha} ***\\n")
    return clase_predicha

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 predict.py <ruta_al_archivo_de_audio_2s>")
        sys.exit(1)
    archivo = sys.argv[1]
    predecir_audio(archivo)
