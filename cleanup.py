# cleanup.py
import os
import sys

# Usaremos exactamente la misma lógica que el programa principal para encontrar el archivo
LOCK_FILE_PATH = os.path.join(os.getenv('TEMP', '/tmp'), 'catchaneye.lock')

print(f"Buscando archivo de bloqueo en la ruta: {LOCK_FILE_PATH}")

if os.path.exists(LOCK_FILE_PATH):
    print("¡Archivo de bloqueo encontrado! Intentando eliminar...")
    try:
        os.remove(LOCK_FILE_PATH)
        print("¡Éxito! El archivo de bloqueo ha sido eliminado.")
        print("Ahora puedes ejecutar el programa principal de nuevo.")
    except Exception as e:
        print(f"\nERROR: No se pudo eliminar el archivo.")
        print(f"Causa probable: {e}")
        print("\nSOLUCIÓN:")
        print("1. Abra el Administrador de Tareas (Ctrl+Shift+Esc) -> Detalles.")
        print("2. Finalice CUALQUIER proceso llamado 'python.exe' o 'pythonw.exe'.")
        print("3. Vuelva a ejecutar este script (cleanup.py).")
else:
    print("No se encontró ningún archivo de bloqueo. El sistema está limpio.")

# Pausa para que puedas leer el mensaje
os.system("pause")