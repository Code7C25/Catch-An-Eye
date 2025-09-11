import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Inicializar Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Cola para suavizar las lecturas
history_len = 5
horizontal_history = deque(maxlen=history_len)
vertical_history = deque(maxlen=history_len)

# Función para obtener ratios horizontal y vertical
def get_eye_ratios(landmarks, eye_points, w, h):
    left = (int(landmarks[eye_points[0]].x * w), int(landmarks[eye_points[0]].y * h))
    right = (int(landmarks[eye_points[1]].x * w), int(landmarks[eye_points[1]].y * h))
    top = (int(landmarks[eye_points[3]].x * w), int(landmarks[eye_points[3]].y * h))
    bottom = (int(landmarks[eye_points[4]].x * w), int(landmarks[eye_points[4]].y * h))
    center = (int(landmarks[eye_points[2]].x * w), int(landmarks[eye_points[2]].y * h))

    # Ratios
    eye_width = right[0] - left[0]
    eye_height = bottom[1] - top[1]
    horiz_ratio = (center[0] - left[0]) / eye_width
    vert_ratio = (center[1] - top[1]) / eye_height

    return horiz_ratio, vert_ratio

# Captura de cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Índices para cada ojo (extremos, pupila, párpados)
            right_eye_points = [33, 133, 159, 158, 145]  # izq, der, pupila, párpado sup, párpado inf
            left_eye_points = [362, 263, 386, 385, 373]

            # Ratios de ambos ojos
            r_horiz, r_vert = get_eye_ratios(face_landmarks.landmark, right_eye_points, w, h)
            l_horiz, l_vert = get_eye_ratios(face_landmarks.landmark, left_eye_points, w, h)

            # Promedio para mayor estabilidad
            gaze_horiz = (r_horiz + l_horiz) / 2
            gaze_vert = (r_vert + l_vert) / 2

            # Guardar en historial
            horizontal_history.append(gaze_horiz)
            vertical_history.append(gaze_vert)

            # Suavizado usando promedio móvil
            smooth_horiz = np.mean(horizontal_history)
            smooth_vert = np.mean(vertical_history)

            # Determinar dirección horizontal
            if smooth_horiz < 0.42:
                horiz_dir = "IZQUIERDA"
            elif smooth_horiz > 0.58:
                horiz_dir = "DERECHA"
            else:
                horiz_dir = "CENTRO"

            # Determinar dirección vertical
            if smooth_vert < 0.42:
                vert_dir = "ARRIBA"
            elif smooth_vert > 0.58:
                vert_dir = "ABAJO"
            else:
                vert_dir = "CENTRO"

            # Mostrar en pantalla
            cv2.putText(frame, f"H: {horiz_dir} | V: {vert_dir}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Dibujar puntos de referencia
            for idx in right_eye_points + left_eye_points:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

    cv2.imshow("Catch-An-Eye - Direccion 2D", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
