############################################################
# IMPORTS Y CONFIGURACIÓN GLOBAL
############################################################
import cv2
import mediapipe as mpgi
import pyautogui
pyautogui.FAILSAFE = False
import sys
import time
from enum import Enum
import math
import numpy as np
import threading
from pynput import keyboard

DEBUG = False

############################################################
# CONFIGURACIÓN Y CONSTANTES DEL SISTEMA
############################################################
class OperatingMode(Enum):
    CURSOR = 1
    SCROLL = 2
    DRAG = 3
    PAUSED = 4

# --- Parámetros de Gestos y Modos ---
LONG_WINK_THRESHOLD = 0.8
BOTH_EYES_CLOSED_THRESHOLD = 1.2
DOUBLE_CLICK_INTERVAL = 0.5
FIXATION_RADIUS = 25
FIXATION_TIME_THRESHOLD = 0.2
DAMPING_FACTOR = 0.25

APPLY_HORIZONTAL_FLIP = True

# --- Variables de control globales para comunicación entre hilos ---
is_paused = threading.Event()
needs_recalibration = threading.Event()
exit_program = threading.Event()

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

def get_gaze_from_landmarks(face_landmarks, w, h):
    try:
        right_iris_center = np.mean(np.array([[lm.x * w, lm.y * h] for lm in [face_landmarks.landmark[i] for i in [474, 475, 476, 477]]]), axis=0)
        left_iris_center = np.mean(np.array([[lm.x * w, lm.y * h] for lm in [face_landmarks.landmark[i] for i in [469, 470, 471, 472]]]), axis=0)
        right_eye_left, right_eye_right = face_landmarks.landmark[33].x * w, face_landmarks.landmark[133].x * w
        left_eye_left, left_eye_right = face_landmarks.landmark[362].x * w, face_landmarks.landmark[263].x * w
        right_ratio = (right_iris_center[0] - right_eye_left) / (right_eye_right - right_eye_left + 1e-6)
        left_ratio = (left_iris_center[0] - left_eye_left) / (left_eye_right - left_eye_left + 1e-6)
        right_vert, left_vert = right_iris_center[1] / h, left_iris_center[1] / h
        gaze_horiz = np.clip((right_ratio + left_ratio) / 2, 0, 1)
        gaze_vert = np.clip((right_vert + left_vert) / 2, 0, 1)
        return gaze_horiz, gaze_vert
    except: return 0.5, 0.5

def calibrate(face_mesh, cap):
    print("Iniciando calibración. Mire los puntos rojos...")
    calibration_gaze, calibration_screen = [], []
    cv2.namedWindow("Catch-An-Eye - Calibracion", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Catch-An-Eye - Calibracion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    for idx, (cx, cy) in enumerate(CALIBRATION_POINTS):
        if exit_program.is_set(): break
        disp_wait = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        cv2.putText(disp_wait, f"Punto {idx+1}/{len(CALIBRATION_POINTS)}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
        cv2.imshow("Catch-An-Eye - Calibracion", disp_wait)
        if cv2.waitKey(1000) & 0xFF == ord('q'): exit_program.set()
        t_start, point_gaze_samples = time.time(), []
        px, py = int(cx * screen_width), int(cy * screen_height)
        while time.time() - t_start < CALIBRATION_WAIT:
            if exit_program.is_set(): break
            ret, frame = cap.read()
            if not ret: continue
            if APPLY_HORIZONTAL_FLIP: frame = cv2.flip(frame, 1)
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                gaze_horiz, gaze_vert = get_gaze_from_landmarks(results.multi_face_landmarks[0], FRAME_WIDTH, FRAME_HEIGHT)
                point_gaze_samples.append([gaze_horiz, gaze_vert])
            disp_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            cv2.circle(disp_frame, (px, py), 20, (0, 0, 255), -1)
            cv2.putText(disp_frame, f"Mire el punto...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Catch-An-Eye - Calibracion", disp_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): exit_program.set()
        if point_gaze_samples:
            arr = np.array(point_gaze_samples)
            mean, std = np.mean(arr, axis=0), np.std(arr, axis=0)
            filtered = [s for s in arr if np.all(np.abs(s - mean) <= 1.5 * std)]
            avg_gaze = np.mean(filtered, axis=0) if len(filtered) > 0 else mean
            calibration_gaze.append(avg_gaze)
            calibration_screen.append([px, py])
    cv2.destroyWindow("Catch-An-Eye - Calibracion")
    if len(calibration_gaze) < 2:
        print("Calibración fallida."), exit_program.set()
        return
    print("Calibración completa.")
    calibration_gaze, calibration_screen = np.array(calibration_gaze), np.array(calibration_screen)
    min_gaze, max_gaze = np.min(calibration_gaze, axis=0), np.max(calibration_gaze, axis=0)
    norm_gaze = (calibration_gaze - min_gaze) / (max_gaze - min_gaze + 1e-8)
    A = np.hstack([norm_gaze, np.ones((norm_gaze.shape[0], 1))])
    coeffs_x, _, _, _ = np.linalg.lstsq(A, calibration_screen[:, 0], rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(A, calibration_screen[:, 1], rcond=None)
    global affine_min_gaze, affine_max_gaze, affine_coeffs_x, affine_coeffs_y
    affine_min_gaze, affine_max_gaze, affine_coeffs_x, affine_coeffs_y = min_gaze, max_gaze, coeffs_x, coeffs_y

############################################################
# GESTOR DE ATAJOS DE TECLADO GLOBALES
############################################################
def on_release(key):
    # Esta función se activa cuando se suelta una combinación de teclas
    pass

def start_hotkey_listener():
    hotkeys = {
        '<ctrl>+<alt>+p': toggle_pause,
        '<ctrl>+<alt>+r': trigger_recalibration,
        '<ctrl>+<alt>+q': quit_program
    }
    listener = keyboard.GlobalHotKeys(hotkeys)
    listener.start()
    print("Listener de atajos de teclado iniciado.")
    print(" - Ctrl+Alt+P: Pausar/Reanudar")
    print(" - Ctrl+Alt+R: Recalibrar")
    print(" - Ctrl+Alt+Q: Salir")
    listener.join()

def toggle_pause():
    if is_paused.is_set():
        is_paused.clear()
        print("Control reanudado.")
    else:
        is_paused.set()
        print("Control pausado.")

def trigger_recalibration():
    if not needs_recalibration.is_set():
        print("Recalibración solicitada.")
        needs_recalibration.set()

def quit_program():
    print("Saliendo del programa...")
    exit_program.set()

############################################################
# BUCLE PRINCIPAL (DENTRO DE UNA FUNCIÓN)
############################################################
def main_loop():
    global FRAME_WIDTH, FRAME_HEIGHT, screen_width, screen_height, CALIBRATION_POINTS, CALIBRATION_WAIT
    FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
    screen_width, screen_height = pyautogui.size()
    FRAME_SKIP = 2
    CALIBRATION_POINTS = [(x, y) for y in [0.05, 0.5, 0.95] for x in [0.05, 0.5, 0.95]]
    CALIBRATION_WAIT = 1.5

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        exit_program.set()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    current_mode = OperatingMode.CURSOR
    is_left_winking, left_wink_start_time, last_left_wink_time = False, 0, 0
    are_both_eyes_closed, both_eyes_closed_start_time = False, 0
    is_fixating, fixation_start_time = False, 0
    last_gaze_pos_for_fixation = (0, 0)
    
    try:
        with mpgi.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
            calibrate(face_mesh, cap)

            def gaze_to_screen(horiz, vert):
                gaze = np.array([horiz, vert])
                norm = (gaze - affine_min_gaze) / (affine_max_gaze - affine_min_gaze + 1e-8)
                A = np.array([norm[0], norm[1], 1.0])
                x = int(np.clip(np.dot(affine_coeffs_x, A), 0, screen_width - 1))
                y = int(np.clip(np.dot(affine_coeffs_y, A), 0, screen_height - 1))
                return x, y
                
            one_euro_filter_x = OneEuroFilter(freq=30, mincutoff=0.8, beta=0.05)
            one_euro_filter_y = OneEuroFilter(freq=30, mincutoff=0.8, beta=0.05)
            
            status_window_name = "Status"
            cv2.namedWindow(status_window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(status_window_name, screen_width - 100, 0)
            cv2.resizeWindow(status_window_name, 100, 50)
            
            while not exit_program.is_set():
                if needs_recalibration.is_set():
                    calibrate(face_mesh, cap)
                    needs_recalibration.clear()

                status_img = np.zeros((50, 100, 3), dtype=np.uint8)
                status_char = ''

                if is_paused.is_set():
                    status_char = 'P'
                    cv2.putText(status_img, status_char, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.imshow(status_window_name, status_img)
                    cv2.waitKey(100)
                    continue

                ret, frame = cap.read()
                if not ret: continue
                if APPLY_HORIZONTAL_FLIP: frame = cv2.flip(frame, 1)

                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if current_mode == OperatingMode.CURSOR: status_char = 'F' if is_fixating else 'C'
                elif current_mode == OperatingMode.SCROLL: status_char = 'S'
                elif current_mode == OperatingMode.DRAG: status_char = 'D'
                cv2.putText(status_img, status_char, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.imshow(status_window_name, status_img)
                cv2.waitKey(1)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    horiz_ratio, vert_ratio = get_gaze_from_landmarks(face_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
                    raw_x, raw_y = gaze_to_screen(horiz_ratio, vert_ratio)
                    timestamp = time.time()
                    gaze_x = int(one_euro_filter_x(raw_x, timestamp))
                    gaze_y = int(one_euro_filter_y(raw_y, timestamp))
                    
                    if current_mode == OperatingMode.CURSOR:
                        distance = math.sqrt((gaze_x - last_gaze_pos_for_fixation[0])**2 + (gaze_y - last_gaze_pos_for_fixation[1])**2)
                        if distance < FIXATION_RADIUS:
                            if not is_fixating and time.time() - fixation_start_time > FIXATION_TIME_THRESHOLD:
                                is_fixating = True
                        else:
                            is_fixating = False
                            last_gaze_pos_for_fixation = (gaze_x, gaze_y)
                            fixation_start_time = time.time()
                        
                        if is_fixating:
                            current_mouse_x, current_mouse_y = pyautogui.position()
                            new_mouse_x = int(current_mouse_x + (gaze_x - current_mouse_x) * DAMPING_FACTOR)
                            new_mouse_y = int(current_mouse_y + (gaze_y - current_mouse_y) * DAMPING_FACTOR)
                            pyautogui.moveTo(new_mouse_x, new_mouse_y, duration=0.0)
                        else:
                            pyautogui.moveTo(gaze_x, gaze_y, duration=0.0)

                    elif current_mode == OperatingMode.SCROLL:
                        scroll_speed = int((0.5 - vert_ratio) * 100)
                        if abs(scroll_speed) > 5: pyautogui.scroll(scroll_speed)

                    elif current_mode == OperatingMode.DRAG:
                        pyautogui.moveTo(gaze_x, gaze_y, duration=0.0)
                    
                    left_wink, right_wink = is_left_wink(face_landmarks.landmark), is_right_wink(face_landmarks.landmark)

                    # --- LÓGICA DE GESTOS RECONSTRUIDA PARA CLARIDAD Y SIN ERRORES DE SINTAXIS ---
                    if left_wink and right_wink:
                        if not are_both_eyes_closed:
                            are_both_eyes_closed = True
                            both_eyes_closed_start_time = time.time()
                        elif time.time() - both_eyes_closed_start_time > BOTH_EYES_CLOSED_THRESHOLD:
                            current_mode = OperatingMode.CURSOR if current_mode == OperatingMode.SCROLL else OperatingMode.SCROLL
                            both_eyes_closed_start_time = time.time() 
                    else:
                        are_both_eyes_closed = False

                    if left_wink and not right_wink:
                        if not is_left_winking:
                            is_left_winking = True
                            left_wink_start_time = time.time()
                    else:
                        if is_left_winking:
                            wink_duration = time.time() - left_wink_start_time
                            if current_mode == OperatingMode.DRAG:
                                pyautogui.mouseUp(button='left')
                                current_mode = OperatingMode.CURSOR
                            elif current_mode == OperatingMode.CURSOR:
                                if wink_duration < LONG_WINK_THRESHOLD:
                                    current_time = time.time()
                                    if current_time - last_left_wink_time < DOUBLE_CLICK_INTERVAL:
                                        pyautogui.doubleClick()
                                        last_left_wink_time = 0
                                    else:
                                        pyautogui.click(button='left')
                                        last_left_wink_time = current_time
                        is_left_winking = False
                        
                    if is_left_winking and current_mode == OperatingMode.CURSOR and time.time() - left_wink_start_time > LONG_WINK_THRESHOLD:
                        current_mode = OperatingMode.DRAG
                        pyautogui.mouseDown(button='left')
                        is_left_winking = False 

                    if right_wink and not left_wink and current_mode == OperatingMode.CURSOR:
                        pyautogui.click(button='right')

    # NUEVO: Bloque except para capturar errores y asegurar un cierre limpio
    except Exception as e:
        print(f"\nERROR INESPERADO EN EL BUCLE PRINCIPAL: {e}")
        import traceback
        traceback.print_exc() # Imprime información detallada del error
        exit_program.set() # Señal para que todo se detenga
    
    finally:
        print("Cerrando recursos...")
        cap.release()
        cv2.destroyAllWindows()

############################################################
# PUNTO DE ENTRADA PRINCIPAL
############################################################
if __name__ == "__main__":
    keyboard_thread = threading.Thread(target=start_hotkey_listener, daemon=True)
    keyboard_thread.start()
    
    main_loop()

    print("Programa finalizado.")