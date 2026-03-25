import sys
import os

# Asegurar que el directorio raíz está en el pythonpath
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from classifier.train import main as train_main
from classifier.predict import predecir_audio

def mostrar_menu():
    print("=" * 50)
    print(" SISTEMA DE CLASIFICACIÓN DE SEÑALES DE AUDIO ")
    print("=" * 50)
    print("Uso:")
    print("  python3 main.py train            -> Inicia la fase de entrenamiento")
    print("  python3 main.py predict <ruta>   -> Clasifica el audio dado")
    print("=" * 50)

def main():
    if len(sys.argv) < 2:
        mostrar_menu()
        sys.exit(0)
        
    comando = sys.argv[1].lower()
    
    if comando == "train":
        train_main()
    elif comando == "predict":
        if len(sys.argv) < 3:
            print("Error: falta el argumento de la ruta del archivo de audio para calcular.")
            print("Uso: python3 main.py predict <ruta_al_archivo>")
            sys.exit(1)
        ruta = sys.argv[2]
        predecir_audio(ruta)
    else:
        print(f"Error: comando '{comando}' no reconocido.")
        mostrar_menu()
        sys.exit(1)

if __name__ == "__main__":
    main()