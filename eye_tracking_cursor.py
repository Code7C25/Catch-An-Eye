# --- IMPORTS ---
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
pyautogui.FAILSAFE = False

# --- CONFIGURACIÓN Y CONSTANTES ---
from collections import deque
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
import pandas as pd
import os

# --- CLASES ---
class CursorSmoother:
    """Encapsula el historial de mirada y el umbral de movimiento adaptativo."""
    def __init__(self, initial_len=10):
        self.history_len = initial_len
        self.horizontal_history = deque(maxlen=initial_len)
        self.vertical_history = deque(maxlen=initial_len)
        self.move_threshold = 10
    def update(self, horiz, vert):
        self.horizontal_history.append(horiz)
        self.vertical_history.append(vert)
        self._adapt_sensitivity()
    def get_smoothed(self):
        return np.mean(self.horizontal_history), np.mean(self.vertical_history)
    def _adapt_sensitivity(self):
        if len(self.horizontal_history) == self.horizontal_history.maxlen:
            std_h = np.std(self.horizontal_history)
            std_v = np.std(self.vertical_history)
            # El mínimo del historial ahora es 7
            if std_h < 0.01 and std_v < 0.01:
                self.move_threshold = 5
                new_len = 7
            elif std_h > 0.03 or std_v > 0.03:
                self.move_threshold = 20
                new_len = 20
            else:
                self.move_threshold = 10
                new_len = 10
            if new_len != self.horizontal_history.maxlen:
                self.horizontal_history = deque(self.horizontal_history, maxlen=new_len)
                self.vertical_history = deque(self.vertical_history, maxlen=new_len)
                self.history_len = new_len

# --- FUNCIÓN DE CALIBRACIÓN ---
def calibrate(face_mesh, cap):
    print("Iniciando calibración. Mire los puntos rojos que aparecen en pantalla.")
    calibration_gaze = []
    calibration_screen = []
    for idx, (cx, cy) in enumerate(CALIBRATION_POINTS):
        disp_wait = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.namedWindow("Catch-An-Eye - Calibracion", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Catch-An-Eye - Calibracion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.putText(disp_wait, f"Punto {idx+1}/{len(CALIBRATION_POINTS)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
        cv2.imshow("Catch-An-Eye - Calibracion", disp_wait)
        cv2.waitKey(1000)
        cv2.namedWindow("Catch-An-Eye - Calibracion", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Catch-An-Eye - Calibracion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        t_start = time.time()
        point_gaze_samples = []
        while time.time() - t_start < CALIBRATION_WAIT:
            ret, frame = cap.read()
            if not ret:
                continue
            disp_frame = frame.copy()
            cv2.circle(disp_frame, (int(cx * FRAME_WIDTH / screen_width), int(cy * FRAME_HEIGHT / screen_height)), 20, (0, 0, 255), -1)
            cv2.putText(disp_frame, f"Mire el punto {idx+1}/{len(CALIBRATION_POINTS)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Catch-An-Eye - Calibracion", disp_frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                gaze_horiz, gaze_vert = get_gaze_from_landmarks(face_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
                point_gaze_samples.append([gaze_horiz, gaze_vert])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
        if point_gaze_samples:
            arr = np.array(point_gaze_samples)
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            filtered = [s for s in arr if np.all(np.abs(s - mean) <= 1.5 * std)]
            if len(filtered) > 0:
                avg_gaze = np.mean(filtered, axis=0)
            else:
                avg_gaze = mean
            calibration_gaze.append(avg_gaze)
            calibration_screen.append([cx, cy])
    cv2.destroyWindow("Catch-An-Eye - Calibracion")
    if len(calibration_gaze) < 2:
        print("Calibración fallida. Se necesitan al menos 2 puntos. Cerrando.")
        cap.release()
        exit()
    calibration_gaze = np.array(calibration_gaze)
    calibration_screen = np.array(calibration_screen)
    # Guardar datos de calibración para futuras ejecuciones
    df = pd.DataFrame({
        "gaze_horiz": calibration_gaze[:,0],
        "gaze_vert": calibration_gaze[:,1],
        "screen_x": calibration_screen[:,0],
        "screen_y": calibration_screen[:,1]
    })
    df.to_csv(CALIBRATION_DATA_PATH, index=False)
    # --- SVR ---
    svr_x = SVR(kernel='rbf', C=100, gamma=0.1)
    svr_y = SVR(kernel='rbf', C=100, gamma=0.1)
    X = calibration_gaze
    y_x = calibration_screen[:,0]
    y_y = calibration_screen[:,1]
    svr_x.fit(X, y_x)
    svr_y.fit(X, y_y)
    print("Calibración completa (SVR). Control de cursor activado.")
    return svr_x, svr_y

# --- CONFIGURACIÓN Y CONSTANTES ---

# Configuración de pantalla
screen_width, screen_height = 1280, 720  # Fijar explícitamente para que coincida con la cámara

# Parámetros de optimización
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_SKIP = 2  # Procesar cada N frames para mejorar rendimiento


# Historial para suavizado de movimiento
smoother = CursorSmoother()

# Filtro de Kalman para suavizar el movimiento del cursor
class SimpleKalman:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0
    def input_latest_noisy_measurement(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate
        return self.posteri_estimate


CALIBRATION_POINTS = [
    (int(screen_width * x), int(screen_height * y))
    for y in [0.05, 0.25, 0.5, 0.75, 0.95]
    for x in [0.05, 0.25, 0.5, 0.75, 0.95]
]
CALIBRATION_WAIT = 1.5  # Segundos por punto de calibración
CALIBRATION_DATA_PATH = "calibration_data.csv"

# --- FUNCIONES AUXILIARES ---

def get_eye_ratios(landmarks, eye_points, w, h):
    """Calcula los ratios de posición del iris dentro del ojo."""
    try:
        left = (int(landmarks[eye_points[0]].x * w), int(landmarks[eye_points[0]].y * h))
        right = (int(landmarks[eye_points[1]].x * w), int(landmarks[eye_points[1]].y * h))
        top = (int(landmarks[eye_points[3]].x * w), int(landmarks[eye_points[3]].y * h))
        bottom = (int(landmarks[eye_points[4]].x * w), int(landmarks[eye_points[4]].y * h))
        center = (int(landmarks[eye_points[2]].x * w), int(landmarks[eye_points[2]].y * h))

        eye_width = np.linalg.norm(np.array(right) - np.array(left))
        eye_height = np.linalg.norm(np.array(bottom) - np.array(top))

        if eye_width == 0 or eye_height == 0:
            return 0.5, 0.5  # Valor neutral si no se detecta el ojo

        horiz_ratio = (center[0] - left[0]) / eye_width
        vert_ratio = (center[1] - top[1]) / eye_height
        return horiz_ratio, vert_ratio
    except:
        return 0.5, 0.5

def get_gaze_from_landmarks(face_landmarks, w, h, draw_frame=None):
    """Obtiene el ratio de mirada usando más puntos del contorno de ambos ojos (sin dibujar puntos)."""
    right_eye_points = [33, 133, 160, 159, 158, 157, 173, 153, 144, 145, 153]
    left_eye_points = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 374, 373]
    def get_eye_coords(eye_points):
        return np.array([
            [int(face_landmarks.landmark[p].x * w), int(face_landmarks.landmark[p].y * h)] for p in eye_points
        ])
    right_eye_coords = get_eye_coords(right_eye_points)
    left_eye_coords = get_eye_coords(left_eye_points)
    right_center = np.mean(right_eye_coords, axis=0)
    left_center = np.mean(left_eye_coords, axis=0)
    r_horiz = right_center[0] / w
    r_vert = right_center[1] / h
    l_horiz = left_center[0] / w
    l_vert = left_center[1] / h
    return (r_horiz + l_horiz) / 2, (r_vert + l_vert) / 2

# --- INICIALIZACIÓN ---

# Captura de cámara

# --- INICIALIZACIÓN ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
last_mouse_pos = pyautogui.position()
frame_count = 0

# Buffer para promediar las últimas N predicciones del cursor
CURSOR_AVG_BUFFER = 5
cursor_x_buffer = deque(maxlen=CURSOR_AVG_BUFFER)
cursor_y_buffer = deque(maxlen=CURSOR_AVG_BUFFER)

# --- AJUSTE AUTOMÁTICO DE BRILLO DE CÁMARA ---
AUTO_BRIGHTNESS = True
BRIGHTNESS_TARGET = 100  # Valor objetivo de brillo promedio (0-255)
BRIGHTNESS_TOLERANCE = 20
BRIGHTNESS_ADJUST_EVERY = 30  # Ajustar cada N frames
BRIGHTNESS_MIN = 0.2  # Límite inferior (0.0-1.0)
BRIGHTNESS_MAX = 0.8  # Límite superior (0.0-1.0)

# --- OPTIMIZACIÓN DE RENDIMIENTO ---
# Puedes revertir quitando el bloque de reducción de frame y restaurando el bucle principal

# --- INICIALIZACIÓN ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
last_mouse_pos = pyautogui.position()
frame_count = 0

# --- LÓGICA PRINCIPAL ---
with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
    # Si existe archivo de calibración, cargar modelo SVR
    if os.path.exists(CALIBRATION_DATA_PATH):
        df = pd.read_csv(CALIBRATION_DATA_PATH)
        X = df[["gaze_horiz", "gaze_vert"]].values
        y_x = df["screen_x"].values
        y_y = df["screen_y"].values
        svr_x = SVR(kernel='rbf', C=100, gamma=0.1)
        svr_y = SVR(kernel='rbf', C=100, gamma=0.1)
        svr_x.fit(X, y_x)
        svr_y.fit(X, y_y)
        print("Modelo SVR cargado desde calibración previa.")
    else:
        svr_x, svr_y = calibrate(face_mesh, cap)
    kalman_x = SimpleKalman()
    kalman_y = SimpleKalman()
    cv2.namedWindow("Catch-An-Eye - Control de Mouse", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Catch-An-Eye - Control de Mouse", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    MAX_FAILED_FRAMES = 30  # Frames seguidos sin detección facial antes de advertir/pausar
    failed_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cámara no detectada. Intentando reconectar...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                continue
            # Reducción de tamaño para procesamiento interno (acelerar MediaPipe)
            small_frame = cv2.resize(frame, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2), interpolation=cv2.INTER_LINEAR)
            frame = cv2.flip(frame, 1)
            frame_count += 1
            # Ajuste automático de brillo de cámara
            if AUTO_BRIGHTNESS and frame_count % BRIGHTNESS_ADJUST_EVERY == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = np.mean(gray)
                if mean_brightness < BRIGHTNESS_TARGET - BRIGHTNESS_TOLERANCE:
                    current = cap.get(cv2.CAP_PROP_BRIGHTNESS)
                    new_val = min(current + 0.05, BRIGHTNESS_MAX)
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, new_val)
                elif mean_brightness > BRIGHTNESS_TARGET + BRIGHTNESS_TOLERANCE:
                    current = cap.get(cv2.CAP_PROP_BRIGHTNESS)
                    new_val = max(current - 0.05, BRIGHTNESS_MIN)
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, new_val)
            if frame_count % FRAME_SKIP != 0:
                cv2.imshow("Catch-An-Eye - Control de Mouse", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    model_x, model_y, poly = calibrate(face_mesh, cap)
                    kalman_x = SimpleKalman()
                    kalman_y = SimpleKalman()
                    continue
                continue
            # Procesar frame reducido para MediaPipe
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_small)
            if results.multi_face_landmarks:
                failed_frames = 0
                face_landmarks = results.multi_face_landmarks[0]
                # Escalar landmarks a frame original
                scale_x = FRAME_WIDTH / (FRAME_WIDTH // 2)
                scale_y = FRAME_HEIGHT / (FRAME_HEIGHT // 2)
                for lm in face_landmarks.landmark:
                    lm.x *= scale_x
                    lm.y *= scale_y
                gaze_horiz, gaze_vert = get_gaze_from_landmarks(face_landmarks, FRAME_WIDTH, FRAME_HEIGHT, frame)
                smoother.update(gaze_horiz, gaze_vert)
                smooth_horiz, smooth_vert = smoother.get_smoothed()
                gaze_point = np.array([[smooth_horiz, smooth_vert]], dtype=np.float32)
                # Usar SVR para predecir la posición del cursor
                pred_x = int(svr_x.predict(gaze_point)[0])
                pred_y = int(svr_y.predict(gaze_point)[0])
                pred_x = int(np.clip(pred_x, 0, screen_width - 1))
                pred_y = int(np.clip(pred_y, 0, screen_height - 1))
                pred_x = int(kalman_x.input_latest_noisy_measurement(pred_x))
                pred_y = int(kalman_y.input_latest_noisy_measurement(pred_y))
                cursor_x_buffer.append(pred_x)
                cursor_y_buffer.append(pred_y)
                if len(cursor_x_buffer) == CURSOR_AVG_BUFFER and len(cursor_y_buffer) == CURSOR_AVG_BUFFER:
                    avg_x = int(np.mean(cursor_x_buffer))
                    avg_y = int(np.mean(cursor_y_buffer))
                    avg_x = int(np.clip(avg_x, 0, screen_width - 1))
                    avg_y = int(np.clip(avg_y, 0, screen_height - 1))
                    if abs(avg_x - last_mouse_pos[0]) > smoother.move_threshold or abs(avg_y - last_mouse_pos[1]) > smoother.move_threshold:
                        pyautogui.moveTo(avg_x, avg_y, duration=0.03)
                        last_mouse_pos = (avg_x, avg_y)
                    cv2.putText(frame, f"Cursor: ({avg_x}, {avg_y})", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Inicializando suavizado...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                failed_frames += 1
                if failed_frames >= MAX_FAILED_FRAMES:
                    cv2.putText(frame, "Advertencia: No se detecta el rostro. Pausando cursor.", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Cara no detectada. Cursor pausado.", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Catch-An-Eye - Control de Mouse", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                model_x, model_y, poly = calibrate(face_mesh, cap)
                kalman_x = SimpleKalman()
                kalman_y = SimpleKalman()
                continue
    finally:
        cap.release()
        cv2.destroyAllWindows()
# --- FIN DE OPTIMIZACIÓN ---
# Para revertir, elimina el bloque de reducción de frame y restaura el bucle principal
