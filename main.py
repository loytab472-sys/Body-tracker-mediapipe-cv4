import cv2
import mediapipe as mp
import urllib.request
import ssl
import os
import numpy as np
import threading
import time

# Фикс SSL на macOS
ssl._create_default_https_context = ssl._create_unverified_context

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ─── Цвета ────────────────────────────────────────────────────────────────────
DOT_COLOR  = (255, 255, 255)
LINE_COLOR = (0, 200, 255)
TEXT_COLOR = (0, 255, 120)
POSE_DOT   = (80, 200, 80)
POSE_LINE  = (80, 256, 121)
FACE_DOT   = (200, 180, 50)

# ─── Связи руки (21 точка) ────────────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ─── Связи тела + голова → тело ───────────────────────────────────────────────
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),
    (0,11),(0,12),
    (7,11),(8,12),
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(12,14),(14,16),
    (15,17),(15,19),(15,21),(17,19),
    (16,18),(16,20),(16,22),(18,20),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]

# ─── Звук из mp3 файла ────────────────────────────────────────────────────────

# 🔊 Укажи имя своего mp3 файла (должен лежать рядом с main.py)
SOUND_FILE = "alert.mp3"

_sound_playing = False
_last_alert_time = 0
ALERT_COOLDOWN = 3.0  # секунд между звуками

_sound_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), SOUND_FILE)
if os.path.exists(_sound_path):
    print(f"Звук загружен: {_sound_path}")
else:
    print(f"⚠ Файл {SOUND_FILE} не найден в папке со скриптом!")
    _sound_path = None

def play_alert(reason="object"):
    global _sound_playing, _last_alert_time
    now = time.time()
    if _sound_playing or (now - _last_alert_time) < ALERT_COOLDOWN:
        return
    _sound_playing = True
    _last_alert_time = now

    def _play():
        global _sound_playing
        if _sound_path:
            os.system(f'afplay "{_sound_path}"')
        else:
            print('\a', end='', flush=True)
            time.sleep(ALERT_COOLDOWN)
        _sound_playing = False

    threading.Thread(target=_play, daemon=True).start()


# ─── Загрузка моделей ─────────────────────────────────────────────────────────
def get_model(filename, url):
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(save_path):
        print(f"Скачиваю {filename}...")
        urllib.request.urlretrieve(url, save_path)
        print("Скачано:", save_path)
    return save_path


def draw_connections(frame, landmarks, connections, w, h, dot_color, line_color, dot_r=4):
    pts = {}
    for idx, lm in enumerate(landmarks):
        px, py = int(lm.x * w), int(lm.y * h)
        pts[idx] = (px, py)
        cv2.circle(frame, (px, py), dot_r, dot_color, -1)
    for a, b in connections:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], line_color, 2)
    return pts


def main():
    hand_model = get_model(
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    pose_model = get_model(
        "pose_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    )
    face_model = get_model(
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hand_model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pose_model),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    face_detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=face_model),
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: камера не найдена!")
        return

    print("Трекер запущен. Нажми Q для выхода.")

    # ─── Детектор движения ────────────────────────────────────────────────────
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=False
    )
    MOTION_THRESHOLD = 3000  # минимальная площадь движения в пикселях

    timestamp_ms = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        hand_result = hand_detector.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_detector.detect_for_video(mp_image, timestamp_ms)
        face_result = face_detector.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33

        # ── Детекция движения (фоновая субтракция) ────────────────────────────
        fg_mask = bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                                   np.ones((5, 5), np.uint8))
        motion_area = cv2.countNonZero(fg_mask)
        motion_detected = motion_area > MOTION_THRESHOLD

        # ── Детекция человека (поза или лицо) ─────────────────────────────────
        person_detected = bool(
            (pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0) or
            (face_result.face_landmarks and len(face_result.face_landmarks) > 0)
        )

        # ── Звук ──────────────────────────────────────────────────────────────
        if person_detected:
            play_alert("person")
        elif motion_detected:
            play_alert("motion")

        # ── Статус тревоги на экране ──────────────────────────────────────────
        if person_detected:
            status_text = "⚠ PERSON DETECTED"
            status_color = (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 80), -1)
        elif motion_detected:
            status_text = "⚡ MOTION DETECTED"
            status_color = (0, 165, 255)
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 60, 80), -1)
        else:
            status_text = "● MONITORING"
            status_color = (0, 200, 0)

        cv2.putText(frame, status_text, (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 2)

        # ── Лицо (mesh) ───────────────────────────────────────────────────────
        if face_result.face_landmarks:
            for landmarks in face_result.face_landmarks:
                for lm in landmarks:
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (px, py), 1, FACE_DOT, -1)

        # ── Поза ──────────────────────────────────────────────────────────────
        if pose_result.pose_landmarks:
            for landmarks in pose_result.pose_landmarks:
                draw_connections(frame, landmarks, POSE_CONNECTIONS, w, h,
                                 POSE_DOT, POSE_LINE, dot_r=4)

        # ── Руки ──────────────────────────────────────────────────────────────
        num_hands = 0
        if hand_result.hand_landmarks:
            num_hands = len(hand_result.hand_landmarks)
            for i, landmarks in enumerate(hand_result.hand_landmarks):
                pts = draw_connections(frame, landmarks, HAND_CONNECTIONS, w, h,
                                       DOT_COLOR, LINE_COLOR, dot_r=5)
                label = "Unknown"
                if hand_result.handedness and i < len(hand_result.handedness):
                    label = hand_result.handedness[i][0].display_name
                wrist = pts.get(0, (30, 80))
                cv2.putText(frame, label, (wrist[0] - 20, wrist[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2)

        # ── Инфо ──────────────────────────────────────────────────────────────
        cv2.putText(frame, f"Hands: {num_hands}  Motion: {motion_area}px",
                    (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(frame, "Q = exit", (w - 110, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.imshow("Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()
    pose_detector.close()
    face_detector.close()
    print("Трекер остановлен.")


if __name__ == "__main__":
    main()