############################################################
# IMPORTS Y CONFIGURACIÓN GLOBAL
############################################################
import cv2
import mediapipe as mpgi
import pyautogui
pyautogui.FAILSAFE = False
import pygetwindow as gw
import sys
import time
from enum import Enum
import math
import numpy as np
import threading
from pynput import keyboard
import os
import atexit

############################################################
# CONFIGURACIÓN Y CONSTANTES DEL SISTEMA
############################################################
class OperatingMode(Enum):
    CURSOR = 1
    SCROLL = 2
    DRAG = 3
    PAUSED = 4

LONG_WINK_THRESHOLD = 0.8
BOTH_EYES_CLOSED_THRESHOLD = 1.2
DOUBLE_CLICK_INTERVAL = 0.5
APPLY_HORIZONTAL_FLIP = True
LEFT_WINK_DEBOUNCE = 0.3

is_paused = threading.Event()
needs_recalibration = threading.Event()
exit_program = threading.Event()

if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))
LOCK_FILE_PATH = os.path.join(application_path, 'catchaneye.lock')

############################################################
# CLASES Y FUNCIONES AUXILIARES
############################################################
class OneEuroFilter:
    def __init__(self, freq=30, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.freq, self.mincutoff, self.beta, self.dcutoff = freq, mincutoff, beta, dcutoff
        self.x_prev, self.dx_prev, self.t_prev = None, None, None
    def __call__(self, x, t_e):
        if self.t_prev is None: self.t_prev, self.x_prev, self.dx_prev = t_e, x, 0.0; return x
        dt = t_e - self.t_prev
        if dt <= 1e-6: return self.x_prev
        alpha_d = self._smoothing_factor(dt, self.dcutoff)
        dx = (x - self.x_prev) / dt
        if self.dx_prev is None: self.dx_prev = dx
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        alpha = self._smoothing_factor(dt, cutoff)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t_e
        return x_hat
    def _smoothing_factor(self, dt, cutoff):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

def get_eye_aspect_ratio(landmarks, eye_indices):
    points = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    vert1 = np.linalg.norm(points[1] - points[5])
    vert2 = np.linalg.norm(points[2] - points[4])
    horiz = np.linalg.norm(points[0] - points[3])
    if horiz == 0: return 0.3
    return (vert1 + vert2) / (2.0 * horiz)

def is_right_wink(landmarks, threshold=0.20):
    return get_eye_aspect_ratio(landmarks, [33, 160, 158, 133, 153, 144]) < threshold

def is_left_wink(landmarks, threshold=0.20):
    return get_eye_aspect_ratio(landmarks, [263, 387, 385, 362, 380, 373]) < threshold

def calibrate(face_mesh, cap):
    print("Iniciando calibración...")
    calibration_data, calibration_screen = [], []
    cal_window_name = "Catch-An-Eye - Calibracion"
    cv2.namedWindow(cal_window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(cal_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    time.sleep(0.7)
    try:
        win = gw.getWindowsWithTitle(cal_window_name)[0]
        if win: win.activate(); win.maximize()
    except Exception: pass

    for idx, (cx, cy) in enumerate(CALIBRATION_POINTS):
        if exit_program.is_set(): break
        disp_wait = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        cv2.putText(disp_wait, f"Punto {idx+1}/{len(CALIBRATION_POINTS)}", (50, 120), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 3)
        cv2.imshow(cal_window_name, disp_wait)
        cv2.waitKey(1); time.sleep(1.0)
        t_start, point_nose_samples = time.time(), []
        px, py = int(cx * screen_width), int(cy * screen_height)
        while time.time() - t_start < CALIBRATION_WAIT:
            if exit_program.is_set(): break
            ret, frame = cap.read()
            if not ret: continue
            if APPLY_HORIZONTAL_FLIP: frame = cv2.flip(frame, 1)
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                nose_x, nose_y = face_landmarks.landmark[1].x * FRAME_WIDTH, face_landmarks.landmark[1].y * FRAME_HEIGHT
                point_nose_samples.append([nose_x, nose_y])
            disp_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            cv2.circle(disp_frame, (px, py), 25, (0, 0, 255), -1)
            cv2.putText(disp_frame, "Apunta con tu nariz al punto", (50, 70), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
            cv2.imshow(cal_window_name, disp_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): exit_program.set()
        if point_nose_samples:
            arr = np.array(point_nose_samples)
            calibration_data.append(np.mean(arr, axis=0))
            calibration_screen.append([px, py])
    cv2.destroyWindow(cal_window_name)
    if len(calibration_data) < 2:
        print("Calibración fallida."), exit_program.set(); return
    print("Calibración completa.")
    calibration_data, calibration_screen = np.array(calibration_data), np.array(calibration_screen)
    min_gaze, max_gaze = np.min(calibration_data, axis=0), np.max(calibration_data, axis=0)
    norm_gaze = (calibration_data - min_gaze) / (max_gaze - min_gaze + 1e-8)
    A = np.hstack([norm_gaze, np.ones((norm_gaze.shape[0], 1))])
    coeffs_x, _, _, _ = np.linalg.lstsq(A, calibration_screen[:, 0], rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(A, calibration_screen[:, 1], rcond=None)
    global affine_min_gaze, affine_max_gaze, affine_coeffs_x, affine_coeffs_y
    affine_min_gaze, affine_max_gaze, affine_coeffs_x, affine_coeffs_y = min_gaze, max_gaze, coeffs_x, coeffs_y

############################################################
# GESTOR DE ATAJOS DE TECLADO GLOBALES
############################################################
def start_hotkey_listener():
    # --- CORRECCIÓN CLAVE AQUÍ ---
    hotkeys = {
        '<ctrl>+<alt>+p': toggle_pause,
        '<ctrl>+<alt>+r': trigger_recalibration, # Añadido el '+' que faltaba
        '<ctrl>+<alt>+q': quit_program
    }
    with keyboard.GlobalHotKeys(hotkeys) as listener:
        print("Listener de atajos de teclado iniciado.")
        print(" - Ctrl+Alt+P: Pausar/Reanudar")
        print(" - Ctrl+Alt+R: Recalibrar")
        print(" - Ctrl+Alt+Q: Salir")
        listener.join()

def toggle_pause(): is_paused.clear() if is_paused.is_set() else is_paused.set(); print("Pausa Alternada.")
def trigger_recalibration(): needs_recalibration.set(); print("Recalibración solicitada.")
def quit_program(): print("Saliendo..."); exit_program.set()

############################################################
# BUCLE PRINCIPAL
############################################################
def main_loop():
    global FRAME_WIDTH, FRAME_HEIGHT, screen_width, screen_height, CALIBRATION_POINTS, CALIBRATION_WAIT
    FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
    screen_width, screen_height = pyautogui.size()
    FRAME_SKIP = 2
    CALIBRATION_POINTS = [(x, y) for y in [0.05, 0.5, 0.95] for x in [0.05, 0.5, 0.95]]
    CALIBRATION_WAIT = 1.5

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Error: No se pudo abrir la cámara."); exit_program.set(); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH), cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    current_mode = OperatingMode.CURSOR
    is_right_winking, right_wink_start_time, last_right_wink_time = False, 0, 0
    are_both_eyes_closed, both_eyes_closed_start_time = False, 0
    last_left_wink_time = 0
    frame_counter = 0
    
    try:
        with mpgi.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
            calibrate(face_mesh, cap)

            def map_to_screen(nose_x, nose_y):
                nose_pos = np.array([nose_x, nose_y])
                norm = (nose_pos - affine_min_gaze) / (affine_max_gaze - affine_min_gaze + 1e-8)
                A = np.array([norm[0], norm[1], 1.0])
                return int(np.clip(np.dot(affine_coeffs_x, A), 0, screen_width - 1)), int(np.clip(np.dot(affine_coeffs_y, A), 0, screen_height - 1))
            
            one_euro_filter_x = OneEuroFilter(freq=30, mincutoff=1.0, beta=0.4)
            one_euro_filter_y = OneEuroFilter(freq=30, mincutoff=1.0, beta=0.4)
            
            while not exit_program.is_set():
                if needs_recalibration.is_set():
                    calibrate(face_mesh, cap); needs_recalibration.clear()
                
                if is_paused.is_set():
                    time.sleep(0.1); continue

                ret, frame = cap.read()
                if not ret: continue
                
                frame_counter += 1
                if frame_counter % FRAME_SKIP != 0: continue
                
                if APPLY_HORIZONTAL_FLIP: frame = cv2.flip(frame, 1)
                
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    nose_x, nose_y = face_landmarks.landmark[1].x * FRAME_WIDTH, face_landmarks.landmark[1].y * FRAME_HEIGHT
                    raw_x, raw_y = map_to_screen(nose_x, nose_y)
                    timestamp = time.time()
                    cursor_x, cursor_y = int(one_euro_filter_x(raw_x, timestamp)), int(one_euro_filter_y(raw_y, timestamp))
                    
                    if current_mode in [OperatingMode.CURSOR, OperatingMode.DRAG]:
                        pyautogui.moveTo(cursor_x, cursor_y, duration=0.0)
                    elif current_mode == OperatingMode.SCROLL:
                        min_nose_y, max_nose_y = affine_min_gaze[1], affine_max_gaze[1]
                        y_range = max_nose_y - min_nose_y
                        if y_range > 1:
                            norm_y = (nose_y - min_nose_y) / y_range; centered_y = norm_y - 0.5
                            scroll_speed = -int(centered_y * 50) 
                            if abs(scroll_speed) > 2: pyautogui.scroll(scroll_speed)
                    
                    left_wink, right_wink = is_left_wink(face_landmarks.landmark), is_right_wink(face_landmarks.landmark)
                    current_time = time.time()
                    if left_wink and right_wink:
                        if not are_both_eyes_closed: are_both_eyes_closed, both_eyes_closed_start_time = True, current_time
                        elif current_time - both_eyes_closed_start_time > BOTH_EYES_CLOSED_THRESHOLD:
                            current_mode = OperatingMode.CURSOR if current_mode == OperatingMode.SCROLL else OperatingMode.SCROLL
                            both_eyes_closed_start_time = current_time 
                    else: are_both_eyes_closed = False
                    if left_wink and not right_wink and current_mode == OperatingMode.CURSOR:
                        if current_time - last_left_wink_time > LEFT_WINK_DEBOUNCE:
                            pyautogui.click(button='right'); last_left_wink_time = current_time
                    if right_wink and not left_wink:
                        if not is_right_winking: is_right_winking, right_wink_start_time = True, current_time
                    else:
                        if is_right_winking:
                            wink_duration = current_time - right_wink_start_time
                            if current_mode == OperatingMode.DRAG: pyautogui.mouseUp(button='left'); current_mode = OperatingMode.CURSOR
                            elif current_mode == OperatingMode.CURSOR:
                                if wink_duration < LONG_WINK_THRESHOLD:
                                    if current_time - last_right_wink_time < DOUBLE_CLICK_INTERVAL: pyautogui.doubleClick(); last_right_wink_time = 0
                                    else: pyautogui.click(button='left'); last_right_wink_time = current_time
                        is_right_winking = False
                    if is_right_winking and current_mode == OperatingMode.CURSOR and current_time - right_wink_start_time > LONG_WINK_THRESHOLD:
                        current_mode = OperatingMode.DRAG; pyautogui.mouseDown(button='left'); is_right_winking = False
    
    except Exception as e:
        print(f"\nERROR INESPERADO: {e}"); import traceback; traceback.print_exc()
        exit_program.set()
    finally:
        print("Cerrando recursos...")
        cap.release()
        cv2.destroyAllWindows()

############################################################
# PUNTO DE ENTRADA PRINCIPAL
############################################################
if __name__ == "__main__":
    if os.path.exists(LOCK_FILE_PATH):
        print("Error: Otra instancia ya está en ejecución.")
        sys.exit(1)
    
    @atexit.register
    def cleanup_lock_file():
        print("Ejecutando limpieza final...")
        if os.path.exists(LOCK_FILE_PATH):
            os.remove(LOCK_FILE_PATH)
            print("Archivo de bloqueo liberado.")

    try:
        with open(LOCK_FILE_PATH, 'w') as f: f.write(str(os.getpid()))
        
        keyboard_thread = threading.Thread(target=start_hotkey_listener, daemon=True)
        keyboard_thread.start()
        
        main_loop_thread = threading.Thread(target=main_loop)
        main_loop_thread.start()
        
        main_loop_thread.join()

    except Exception as e:
        print(f"Error fatal en el punto de entrada: {e}")
    finally:
        exit_program.set()
        print("Programa finalizado.")