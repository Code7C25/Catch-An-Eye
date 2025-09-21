############################################################
# FUNCIONES DE DETECCIÓN DE GUIÑO Y ASPECTO DE OJO
############################################################
def get_eye_aspect_ratio(landmarks, eye_indices):
    # Calcula la relación de aspecto del ojo (EAR) para detectar cierre
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    # Distancias verticales
    vert1 = np.linalg.norm(p2 - p6)
    vert2 = np.linalg.norm(p3 - p5)
    # Distancia horizontal
    horiz = np.linalg.norm(p1 - p4)
    ear = (vert1 + vert2) / (2.0 * horiz)
    return ear

def is_right_wink(landmarks, threshold=0.20):
    # Indices del ojo derecho según mediapipe (ojo del usuario, no de la imagen)
    right_eye_indices = [33, 160, 158, 133, 153, 144]
    ear = get_eye_aspect_ratio(landmarks, right_eye_indices)
    return ear < threshold

def is_left_wink(landmarks, threshold=0.20):
    # Indices del ojo izquierdo según mediapipe (ojo del usuario, no de la imagen)
    left_eye_indices = [263, 387, 385, 362, 380, 373]
    ear = get_eye_aspect_ratio(landmarks, left_eye_indices)
    return ear < threshold
############################################################
# IMPORTS Y CONFIGURACIÓN GLOBAL
############################################################
import cv2
import mediapipe as mpgi
import numpy as np
import pyautogui
pyautogui.FAILSAFE = False
import pygetwindow as gw
from collections import deque
import time
DEBUG = True

############################################################
# CONFIGURACIÓN Y CONSTANTES DEL SISTEMA
############################################################
CURSOR_AVG_BUFFER = 5
cursor_x_buffer = deque(maxlen=CURSOR_AVG_BUFFER)
cursor_y_buffer = deque(maxlen=CURSOR_AVG_BUFFER)
AUTO_BRIGHTNESS = True
BRIGHTNESS_TARGET = 100
BRIGHTNESS_TOLERANCE = 20
BRIGHTNESS_ADJUST_EVERY = 30
BRIGHTNESS_MIN = 0.2
BRIGHTNESS_MAX = 0.8
def focus_app_window():
    try:
        cv2.namedWindow("Catch-An-Eye - Control de Mouse", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Catch-An-Eye - Control de Mouse", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        time.sleep(0.5)
        windows = gw.getWindowsWithTitle("Catch-An-Eye - Control de Mouse")
        if windows:
            windows[0].activate()
    except Exception as e:
        print(f"No se pudo enfocar la ventana: {e}")
############################################################
# CLASES PARA SUAVIZADO Y FILTRO DE KALMAN
############################################################
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

############################################################
# FUNCIÓN DE CALIBRACIÓN DE MIRADA
############################################################
def calibrate(face_mesh, cap):
    print("Iniciando calibración. Mire los puntos rojos que aparecen en pantalla.")
    calibration_gaze = []
    calibration_screen = []
    for idx, (cx, cy) in enumerate(CALIBRATION_POINTS):
        disp_wait = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        cv2.namedWindow("Catch-An-Eye - Calibracion", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Catch-An-Eye - Calibracion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.putText(disp_wait, f"Punto {idx+1}/{len(CALIBRATION_POINTS)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
        cv2.imshow("Catch-An-Eye - Calibracion", disp_wait)
        cv2.waitKey(1000)
        cv2.namedWindow("Catch-An-Eye - Calibracion", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Catch-An-Eye - Calibracion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        t_start = time.time()
        point_gaze_samples = []
        px = int(cx * screen_width)
        py = int(cy * screen_height)
        while time.time() - t_start < CALIBRATION_WAIT:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)
            disp_frame = frame.copy()
            cv2.circle(disp_frame, (px, py), 20, (0, 0, 255), -1)
            cv2.putText(disp_frame, f"Mire el punto {idx+1}/{len(CALIBRATION_POINTS)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Catch-An-Eye - Calibracion", disp_frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                gaze_horiz, gaze_vert = get_gaze_from_landmarks(face_landmarks, screen_width, screen_height)
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
            calibration_screen.append([px, py])
    cv2.destroyWindow("Catch-An-Eye - Calibracion")
    if len(calibration_gaze) < 2:
        print("Calibración fallida. Se necesitan al menos 2 puntos. Cerrando.")
        cap.release()
        exit()
    print("Calibración completa. Control de cursor activado.")
    # Ajuste affine: calcular matriz de transformación de ratios a pantalla
    calibration_gaze = np.array(calibration_gaze)
    calibration_screen = np.array(calibration_screen)
    # Normalizar ratios de mirada (gaze) entre 0 y 1
    min_gaze = np.min(calibration_gaze, axis=0)
    max_gaze = np.max(calibration_gaze, axis=0)
    norm_gaze = (calibration_gaze - min_gaze) / (max_gaze - min_gaze + 1e-8)
    # Ajuste lineal: resolver Ax = b para mapeo affine
    A = np.hstack([norm_gaze, np.ones((norm_gaze.shape[0], 1))])
    bx = calibration_screen[:, 0]
    by = calibration_screen[:, 1]
    coeffs_x, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
    # Guardar parámetros globales
    global affine_min_gaze, affine_max_gaze, affine_coeffs_x, affine_coeffs_y
    affine_min_gaze = min_gaze
    affine_max_gaze = max_gaze
    affine_coeffs_x = coeffs_x
    affine_coeffs_y = coeffs_y
    print("[DEBUG] Parámetros de calibración affine guardados.")
    return None, None

# --- CONFIGURACIÓN Y CONSTANTES ---

# Configuración de pantalla
screen_width, screen_height = pyautogui.size()  # Usar resolución real de pantalla

# Parámetros de optimización
FRAME_WIDTH = screen_width      
FRAME_HEIGHT = screen_height   
FRAME_SKIP = 4  # Procesar cada N frames para mejorar rendimiento (aumentado)


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
    (x, y)
    for y in [0.05, 0.25, 0.5, 0.75, 0.95]
    for x in [0.05, 0.25, 0.5, 0.75, 0.95]
]
CALIBRATION_WAIT = 1.5  # Segundos por punto de calibración


############################################################
# FUNCIONES AUXILIARES DE PROCESAMIENTO DE LANDMARKS Y MIRADA
############################################################

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
    """Obtiene el ratio de mirada usando los landmarks del iris para mayor variación."""
    try:
        # Landmarks del iris derecho e izquierdo (mediapipe)
        right_iris = [474, 475, 476, 477]
        left_iris = [469, 470, 471, 472]
        right_eye = [33, 133]
        left_eye = [362, 263]
        right_iris_coords = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in right_iris])
        left_iris_coords = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in left_iris])
        right_iris_center = np.mean(right_iris_coords, axis=0)
        left_iris_center = np.mean(left_iris_coords, axis=0)
        right_eye_left = face_landmarks.landmark[right_eye[0]].x * w
        right_eye_right = face_landmarks.landmark[right_eye[1]].x * w
        left_eye_left = face_landmarks.landmark[left_eye[0]].x * w
        left_eye_right = face_landmarks.landmark[left_eye[1]].x * w
        right_ratio = (right_iris_center[0] - right_eye_left) / (right_eye_right - right_eye_left)
        left_ratio = (left_iris_center[0] - left_eye_left) / (left_eye_right - left_eye_left)
        right_vert = right_iris_center[1] / h
        left_vert = left_iris_center[1] / h
        gaze_horiz = (right_ratio + left_ratio) / 2
        gaze_vert = (right_vert + left_vert) / 2
        gaze_horiz = np.clip(gaze_horiz, 0, 1)
        gaze_vert = np.clip(gaze_vert, 0, 1)
        return gaze_horiz, gaze_vert
    except Exception as e:
        print(f"Error en get_gaze_from_landmarks (iris): {e}")
        return 0.5, 0.5
BRIGHTNESS_TOLERANCE = 20
BRIGHTNESS_ADJUST_EVERY = 30  # Ajustar cada N frames
BRIGHTNESS_MIN = 0.2  # Límite inferior (0.0-1.0)
BRIGHTNESS_MAX = 0.8  # Límite superior (0.0-1.0)

# --- OPTIMIZACIÓN DE RENDIMIENTO ---
# Puedes revertir quitando el bloque de reducción de frame y restaurando el bucle principal

############################################################
# INICIALIZACIÓN DE CÁMARA Y VARIABLES PRINCIPALES
############################################################
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
last_mouse_pos = pyautogui.position()
frame_count = 0

############################################################
# BUCLE PRINCIPAL DE PROCESAMIENTO Y CONTROL DE CURSOR
############################################################
with mpgi.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
    # Calibración forzada al inicio (sin guardar ni cargar archivos)
    calibrate(face_mesh, cap)
    # Definir función para transformar ratios de mirada a coordenadas de pantalla
    def gaze_to_screen(horiz, vert):
        # Normalizar usando parámetros de calibración
        gaze = np.array([horiz, vert])
        norm = (gaze - affine_min_gaze) / (affine_max_gaze - affine_min_gaze + 1e-8)
        A = np.array([norm[0], norm[1], 1.0])
        x = np.dot(affine_coeffs_x, A)
        y = np.dot(affine_coeffs_y, A)
        # Limitar a pantalla
        x = int(np.clip(x, 0, screen_width - 1))
        y = int(np.clip(y, 0, screen_height - 1))
        return x, y
    kalman_x = SimpleKalman()
    kalman_y = SimpleKalman()
    cv2.namedWindow("Catch-An-Eye - Control de Mouse", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Catch-An-Eye - Control de Mouse", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    focus_app_window()
    MAX_FAILED_FRAMES = 30  # Frames seguidos sin detección facial antes de advertir/pausar
    failed_frames = 0
    print("[DEBUG] Bucle principal ejecutándose...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[DEBUG] Cámara no detectada. Intentando reconectar...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                continue
            # Procesamiento directo del frame original (sin redimensionar dos veces)
            frame = cv2.flip(frame, 1)
            frame_count += 1

            # Eliminar bloqueo del mouse, siempre actualizar posición
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
                    calibrate(face_mesh, cap)
                    kalman_x = SimpleKalman()
                    kalman_y = SimpleKalman()
                    continue
                continue
            # Procesar frame original para MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                failed_frames = 0
                face_landmarks = results.multi_face_landmarks[0]
                # El escalado de landmarks es innecesario, se elimina

                # --- DETECCIÓN DE GUIÑOS ---
                right_wink = is_right_wink(face_landmarks.landmark)
                left_wink = is_left_wink(face_landmarks.landmark)
                gaze_horiz, gaze_vert = get_gaze_from_landmarks(face_landmarks, FRAME_WIDTH, FRAME_HEIGHT, frame)
                if DEBUG:
                    print(f"[DEBUG] Landmarks detectados")
                smoother.update(gaze_horiz, gaze_vert)
                smooth_horiz, smooth_vert = smoother.get_smoothed()
                if DEBUG:
                    print(f"[DEBUG] Ratios de mirada suavizados: horiz={smooth_horiz:.3f}, vert={smooth_vert:.3f}")
                pred_x, pred_y = gaze_to_screen(smooth_horiz, smooth_vert)
                if DEBUG:
                    print(f"[DEBUG] Coordenadas transformadas (affine): x={pred_x}, y={pred_y}")
                pred_x = int(kalman_x.input_latest_noisy_measurement(pred_x))
                pred_y = int(kalman_y.input_latest_noisy_measurement(pred_y))
                if DEBUG:
                    print(f"[DEBUG] Coordenadas tras Kalman: x={pred_x}, y={pred_y}")
                cursor_x_buffer.append(pred_x)
                cursor_y_buffer.append(pred_y)
                if len(cursor_x_buffer) == CURSOR_AVG_BUFFER and len(cursor_y_buffer) == CURSOR_AVG_BUFFER:
                    avg_x = int(np.mean(cursor_x_buffer))
                    avg_y = int(np.mean(cursor_y_buffer))
                    avg_x = int(np.clip(avg_x, 0, screen_width - 1))
                    avg_y = int(np.clip(avg_y, 0, screen_height - 1))
                    try:
                        pyautogui.moveTo(avg_x, avg_y, duration=0.03)
                        if DEBUG:
                            print(f"[DEBUG] Moviendo cursor a: ({avg_x}, {avg_y})")
                        last_mouse_pos = (avg_x, avg_y)
                    except Exception as e:
                        print(f"Error moviendo el cursor: {e}")
                    # --- CLICK POR GUIÑOS ---
                    if left_wink and not right_wink:
                        pyautogui.click(avg_x, avg_y, button='left')
                        if DEBUG:
                            print("[DEBUG] Click izquierdo por guiño izquierdo")
                    elif right_wink and not left_wink:
                        pyautogui.click(avg_x, avg_y, button='right')
                        if DEBUG:
                            print("[DEBUG] Click derecho por guiño derecho")
                    # Si ambos ojos guiñan, no hacer nada
                else:
                    pass
            else:
                cv2.putText(frame, "Rostro no detectado", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                pass
            cv2.imshow("Catch-An-Eye - Control de Mouse", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                calibrate(face_mesh, cap)
                kalman_x = SimpleKalman()
                kalman_y = SimpleKalman()
    except Exception as e:
        print(f"Error en el bucle principal: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
