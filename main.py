# Importar librerías
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from collections import defaultdict, deque
import math
import os

# Configuración inicial
MODEL_NAME = "movenet_lightning"  # "movenet_lightning" o "movenet_thunder"

# Cargar el modelo MoveNet desde TensorFlow Hub
if MODEL_NAME == "movenet_lightning":
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    input_size = 192
elif MODEL_NAME == "movenet_thunder":
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    input_size = 256
else:
    raise ValueError("Modelo no soportado")

# Mapeo de keypoints
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Colores para visualización
COLORS = [
    (255, 0, 0),    # Rojo
    (0, 255, 0),    # Verde
    (0, 0, 255),    # Azul
    (255, 255, 0),  # Amarillo
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cian
]

def draw_keypoints(frame, keypoints, confidence_threshold=0.3):
    """Dibuja los keypoints en el frame"""
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, conf = kp
        if conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def draw_connections(frame, keypoints, edges, confidence_threshold=0.3):
    """Dibuja las conexiones entre keypoints"""
    y, x, _ = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

def calculate_angle(a, b, c):
    """Calcula el ángulo entre tres puntos"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def detect_pose(keypoints, confidence_threshold=0.3):
    """Detecta posturas específicas basadas en los keypoints"""
    poses = []

    # Extraer keypoints con confianza suficiente
    kp = {name: (keypoints[0][idx][0], keypoints[0][idx][1], keypoints[0][idx][2])
          for name, idx in KEYPOINT_DICT.items()}

    # Detectar manos arriba
    if (kp['left_wrist'][2] > confidence_threshold and
        kp['left_shoulder'][2] > confidence_threshold and
        kp['left_wrist'][0] < kp['left_shoulder'][0]):
        poses.append("Mano izquierda arriba")

    if (kp['right_wrist'][2] > confidence_threshold and
        kp['right_shoulder'][2] > confidence_threshold and
        kp['right_wrist'][0] < kp['right_shoulder'][0]):
        poses.append("Mano derecha arriba")

    # Detectar sentado
    if (kp['left_hip'][2] > confidence_threshold and
        kp['left_knee'][2] > confidence_threshold and
        kp['left_ankle'][2] > confidence_threshold and
        abs(kp['left_knee'][0] - kp['left_hip'][0]) < 0.1 * abs(kp['left_ankle'][0] - kp['left_hip'][0])):
        poses.append("Sentado (lado izquierdo)")

    if (kp['right_hip'][2] > confidence_threshold and
        kp['right_knee'][2] > confidence_threshold and
        kp['right_ankle'][2] > confidence_threshold and
        abs(kp['right_knee'][0] - kp['right_hip'][0]) < 0.1 * abs(kp['right_ankle'][0] - kp['right_hip'][0])):
        poses.append("Sentado (lado derecho)")

    return poses

class PersonTracker:
    """Sistema de seguimiento para mantener identidades de personas"""
    def __init__(self, max_distance=50, max_missing_frames=5):
        self.next_id = 0
        self.tracks = {}
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.keypoint_history = defaultdict(lambda: deque(maxlen=10))

    def update(self, current_keypoints):
        # Si no hay tracks y hay keypoints actuales, inicializar nuevos tracks
        if not self.tracks and current_keypoints:
            for kp in current_keypoints:
                self.tracks[self.next_id] = {
                    'keypoints': kp,
                    'missing_frames': 0,
                    'color': COLORS[self.next_id % len(COLORS)]
                }
                self.next_id += 1
            return self.tracks

        # Si no hay keypoints actuales, incrementar missing_frames para todos
        if not current_keypoints:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['missing_frames'] += 1
                if self.tracks[tid]['missing_frames'] > self.max_missing_frames:
                    del self.tracks[tid]
            return self.tracks

        # Emparejamiento de keypoints existentes con los nuevos
        matched_tracks = set()
        matched_keypoints = set()
        matches = []

        # Calcular distancias entre todos los tracks y keypoints
        for tid, track in self.tracks.items():
            for i, kp in enumerate(current_keypoints):
                # Usar la distancia entre los centros de los cuerpos como métrica
                track_center = np.mean(track['keypoints'][[KEYPOINT_DICT['left_hip'],
                                                         KEYPOINT_DICT['right_hip']], :2], axis=0)
                kp_center = np.mean(kp[[KEYPOINT_DICT['left_hip'],
                                      KEYPOINT_DICT['right_hip']], :2], axis=0)
                dist = distance.euclidean(track_center, kp_center)

                if dist < self.max_distance:
                    matches.append((dist, tid, i))

        # Ordenar matches por distancia
        matches.sort()

        # Asignar los mejores matches primero
        for dist, tid, i in matches:
            if tid not in matched_tracks and i not in matched_keypoints:
                matched_tracks.add(tid)
                matched_keypoints.add(i)
                self.tracks[tid]['keypoints'] = current_keypoints[i]
                self.tracks[tid]['missing_frames'] = 0
                self.keypoint_history[tid].append(current_keypoints[i])

        # Incrementar missing_frames para tracks no emparejados
        for tid in list(self.tracks.keys()):
            if tid not in matched_tracks:
                self.tracks[tid]['missing_frames'] += 1
                if self.tracks[tid]['missing_frames'] > self.max_missing_frames:
                    del self.tracks[tid]
                    if tid in self.keypoint_history:
                        del self.keypoint_history[tid]

        # Crear nuevos tracks para keypoints no emparejados
        for i, kp in enumerate(current_keypoints):
            if i not in matched_keypoints:
                self.tracks[self.next_id] = {
                    'keypoints': kp,
                    'missing_frames': 0,
                    'color': COLORS[self.next_id % len(COLORS)]
                }
                self.keypoint_history[self.next_id].append(kp)
                self.next_id += 1

        return self.tracks

# Definir conexiones y colores para visualización
EDGES = {
    (0, 1): (255, 0, 0),      # Nariz - Ojo izquierdo
    (0, 2): (0, 0, 255),      # Nariz - Ojo derecho
    (1, 3): (255, 0, 0),      # Ojo izquierdo - Oreja izquierda
    (2, 4): (0, 0, 255),      # Ojo derecho - Oreja derecha
    (0, 5): (255, 0, 0),      # Nariz - Hombro izquierdo
    (0, 6): (0, 0, 255),      # Nariz - Hombro derecho
    (5, 6): (0, 255, 0),      # Hombro izquierdo - Hombro derecho
    (5, 7): (255, 0, 0),      # Hombro izquierdo - Codo izquierdo
    (7, 9): (255, 0, 0),      # Codo izquierdo - Muñeca izquierda
    (6, 8): (0, 0, 255),      # Hombro derecho - Codo derecho
    (8, 10): (0, 0, 255),     # Codo derecho - Muñeca derecha
    (5, 11): (255, 0, 0),     # Hombro izquierdo - Cadera izquierda
    (6, 12): (0, 0, 255),     # Hombro derecho - Cadera derecha
    (11, 12): (0, 255, 0),    # Cadera izquierda - Cadera derecha
    (11, 13): (255, 0, 0),    # Cadera izquierda - Rodilla izquierda
    (13, 15): (255, 0, 0),    # Rodilla izquierda - Tobillo izquierdo
    (12, 14): (0, 0, 255),    # Cadera derecha - Rodilla derecha
    (14, 16): (0, 0, 255)     # Rodilla derecha - Tobillo derecho
}

# ...importaciones y definiciones previas...

def run_webcam_realtime():
    cap = cv2.VideoCapture(0)  # Usa la webcam por defecto
    if not cap.isOpened():
        print("No se pudo abrir la webcam")
        return

    tracker = PersonTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = tf.image.resize_with_pad(np.expand_dims(rgb_frame, axis=0), input_size, input_size)
        rgb_frame = tf.cast(rgb_frame, dtype=tf.int32)

        outputs = module.signatures['serving_default'](rgb_frame)
        keypoints = outputs['output_0'].numpy().reshape((-1, 17, 3))

        valid_keypoints = []
        for person_keypoints in keypoints:
            if (person_keypoints[KEYPOINT_DICT['left_shoulder'], 2] > 0.2 and
                person_keypoints[KEYPOINT_DICT['right_shoulder'], 2] > 0.2 and
                person_keypoints[KEYPOINT_DICT['left_hip'], 2] > 0.2 and
                person_keypoints[KEYPOINT_DICT['right_hip'], 2] > 0.2):
                valid_keypoints.append(person_keypoints)

        tracks = tracker.update(valid_keypoints)
        display_frame = frame.copy()

        for tid, track in tracks.items():
            color = track['color']
            kp = track['keypoints']
            # Dibuja conexiones
            for edge in EDGES:
                p1, p2 = edge
                if p1 < len(kp) and p2 < len(kp) and kp[p1][2] > 0.3 and kp[p2][2] > 0.3:
                    y1, x1, _ = kp[p1]
                    y2, x2, _ = kp[p2]
                    cv2.line(display_frame,
                             (int(x1*width), int(y1*height)),
                             (int(x2*width), int(y2*height)),
                             color, 2)
            # Dibuja keypoints
            for point_idx in range(len(kp)):
                if kp[point_idx][2] > 0.3:
                    ky, kx, _ = kp[point_idx]
                    cv2.circle(display_frame, (int(kx*width), int(ky*height)), 4, color, -1)

        cv2.imshow('MoveNet Webcam', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Para ejecutar:
run_webcam_realtime()