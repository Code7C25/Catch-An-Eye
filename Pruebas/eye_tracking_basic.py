import cv2
import mediapipe as mp
import numpy as np

# Inicializar Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Capturar cámara
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
            # Landmarks principales de los ojos
            eye_indices = [33, 133, 362, 263]
            for idx in eye_indices:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Cálculo del centro del ojo derecho
            right_eye_center = np.mean([
                (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),
                (face_landmarks.landmark[133].x * w, face_landmarks.landmark[133].y * h)
            ], axis=0)

            # Cálculo del centro del ojo izquierdo
            left_eye_center = np.mean([
                (face_landmarks.landmark[362].x * w, face_landmarks.landmark[362].y * h),
                (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h)
            ], axis=0)

            # Dibujar puntos centrales
            cv2.circle(frame, (int(right_eye_center[0]), int(right_eye_center[1])), 3, (255, 0, 0), -1)
            cv2.circle(frame, (int(left_eye_center[0]), int(left_eye_center[1])), 3, (255, 0, 0), -1)

    cv2.imshow("Catch-An-Eye - Basic Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
