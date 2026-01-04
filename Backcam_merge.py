import cv2
import time
import math
import mediapipe as mp
import numpy as np
from ultralytics import YOLO


# =========================
# 모델 로드
# =========================
# trt engine model use

model = YOLO("mediapipe_best.engine", task="detect")
yolo_phone_model = YOLO("yolov8n.engine", task="detect")


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

# =========================
# Landmark Index
# =========================
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LEFT_IRIS_CENTER = 468


# =========================
# EAR 설정
# =========================
EAR_WINDOW_SIZE = 60          # 약 2초
EAR_CLOSED_MIN = 0.15
EAR_OPEN_FLOOR = 0.30

ear_buffer = []
prev_ear = None


# =========================
# 눈 감김 / 깜빡임
# =========================
EYE_CLOSE_PERCENT_THRESHOLD = 30
EYE_CLOSE_DURATION_THRESHOLD = 15
eye_close_start = None

BLINK_COUNT_THRESHOLD = 5
BLINK_TIME_WINDOW = 5
blink_timestamps = []
blink_flag = False


# =========================
# YOLO 경고
# =========================
YOLO_ALERT_DURATION = 2
yolo_alert_start = None
YOLO_ALERTED = False


# =========================
# 유틸 함수
# =========================
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def get_precise_ear(landmarks, idx):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in idx]
    v1 = distance(p2, p6)
    v2 = distance(p3, p5)
    h = distance(p1, p4)
    return (v1 + v2) / (2.0 * h) if h > 0 else 0


def dynamic_closure_percent(ear, ear_open):
    ratio = (ear_open - ear) / (ear_open - EAR_CLOSED_MIN)
    return np.clip(ratio, 0.0, 1.0) * 100


def boxes_overlap(face_box, obj_box):
    fx, fy, fw, fh = face_box
    ox, oy, ow, oh = obj_box
    ix1 = max(fx, ox)
    iy1 = max(fy, oy)
    ix2 = min(fx + fw, ox + ow)
    iy2 = min(fy + fh, oy + oh)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1) > 0


def get_gaze_direction(landmarks, eye_idx, iris_idx):
    left = landmarks[eye_idx[0]]
    right = landmarks[eye_idx[3]]
    iris = landmarks[iris_idx]
    x_pos = (iris.x - left.x) / (right.x - left.x)

    if x_pos < 0.45:
        return "Right"
    elif x_pos > 0.6:
        return "Left"
    return "Center"


# =========================
# 메인 실행 함수
# =========================
def run():
    global prev_ear, eye_close_start, blink_flag
    global yolo_alert_start, YOLO_ALERTED

    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        current_time = time.time()

        face_box = None
        sunglasses_detected = False
        phone_detected = False

        # =========================
        # 얼굴 박스 계산
        # =========================
        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0]
            landmarks = face.landmark
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            x_min, y_min = int(min(xs) * w), int(min(ys) * h)
            x_max, y_max = int(max(xs) * w), int(max(ys) * h)
            face_box = (x_min, y_min, x_max - x_min, y_max - y_min)

        # =========================
        # 선글라스 탐지
        # =========================
        for box in model(frame, verbose=False)[0].boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_box = (x1, y1, x2 - x1, y2 - y1)

            if cls == 2 and face_box and boxes_overlap(face_box, obj_box):
                sunglasses_detected = True
                if yolo_alert_start is None:
                    yolo_alert_start = current_time
                elif current_time - yolo_alert_start >= YOLO_ALERT_DURATION:
                    YOLO_ALERTED = True
            else:
                yolo_alert_start = None
                YOLO_ALERTED = False

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if YOLO_ALERTED:
            cv2.putText(frame, "Sunglasses Detected", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # =========================
        # EAR / 시선 / 깜빡임
        # =========================
        if face_box and not sunglasses_detected:
            left_ear = get_precise_ear(landmarks, LEFT_EYE_IDX)
            right_ear = get_precise_ear(landmarks, RIGHT_EYE_IDX)
            ear = (left_ear + right_ear) / 2

            if prev_ear is None:
                prev_ear = ear
            ear = 0.2 * ear + 0.8 * prev_ear
            prev_ear = ear

            ear_buffer.append(ear)
            if len(ear_buffer) > EAR_WINDOW_SIZE:
                ear_buffer.pop(0)

            ear_open = max(max(ear_buffer), EAR_OPEN_FLOOR)
            closure = dynamic_closure_percent(ear, ear_open)

            cv2.putText(frame, f"Eye Closure: {closure:.1f}%",
                        (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

            if closure >= EYE_CLOSE_PERCENT_THRESHOLD:
                if eye_close_start is None:
                    eye_close_start = current_time
                elif current_time - eye_close_start >= EYE_CLOSE_DURATION_THRESHOLD:
                    cv2.putText(frame, "Eyes Closed Long",
                                (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                eye_close_start = None

            if closure >= EYE_CLOSE_PERCENT_THRESHOLD and not blink_flag:
                blink_timestamps.append(current_time)
                blink_flag = True
            if closure < EYE_CLOSE_PERCENT_THRESHOLD:
                blink_flag = False

            blink_timestamps[:] = [
                t for t in blink_timestamps
                if current_time - t <= BLINK_TIME_WINDOW
            ]

            if len(blink_timestamps) >= BLINK_COUNT_THRESHOLD:
                cv2.putText(frame, "Excessive Blinking",
                            (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            gaze = get_gaze_direction(landmarks, LEFT_EYE_IDX, LEFT_IRIS_CENTER)
            cv2.putText(frame, f"Gaze: {gaze}",
                        (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

        # =========================
        # 휴대폰 탐지
        # =========================
        for box in yolo_phone_model(frame, verbose=False)[0].boxes:
            if yolo_phone_model.names[int(box.cls[0])] == "cell phone":
                phone_detected = True
                frame_count += 1
                cv2.putText(frame, "Cell Phone Detected",
                            (30, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
                break

        cv2.imshow("BackCam - DMS", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================
# 단독 실행
# =========================
if __name__ == "__main__":
    run()
