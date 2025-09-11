import cv2
import mediapipe as mp
import numpy as np

# Inicializar Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Función para obtener la relación entre la pupila y el ojo
def get_eye_ratio(landmarks, eye_points, w, h):
    left = (int(landmarks[eye_points[0]].x * w), int(landmarks[eye_points[0]].y * h))
    right = (int(landmarks[eye_points[1]].x * w), int(landmarks[eye_points[1]].y * h))
    center = (int(landmarks[eye_points[2]].x * w), int(landmarks[eye_points[2]].y * h))

    # Distancia horizontal entre extremos del ojo
    eye_width = right[0] - left[0]
    # Posición relativa de la pupila al ancho del ojo
    ratio = (center[0] - left[0]) / eye_width
    return ratio

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

            # Índices para cada ojo
            right_eye_points = [33, 133, 159]  # esquina izquierda, esquina derecha, centro pupila
            left_eye_points = [362, 263, 386]

            # Ratios de ambos ojos
            right_ratio = get_eye_ratio(face_landmarks.landmark, right_eye_points, w, h)
            left_ratio = get_eye_ratio(face_landmarks.landmark, left_eye_points, w, h)

            # Promediar para mayor estabilidad
            gaze_ratio = (right_ratio + left_ratio) / 2

            # Determinar dirección de la mirada
            if gaze_ratio < 0.42:
                direction = "DERECHA"
            elif gaze_ratio > 0.58:
                direction = "IZQUIERDA"
            else:
                direction = "CENTRO"

            # Mostrar texto en pantalla
            cv2.putText(frame, f"Direccion: {direction}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Dibujar puntos de referencia
            for idx in right_eye_points + left_eye_points:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

    cv2.imshow("Catch-An-Eye - Direccion de Mirada", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()