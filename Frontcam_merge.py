import cv2
import torch
import numpy as np
import time
import logging
import cProfile
from datetime import datetime
from bisenetv2 import BiSeNetV2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
from collections import defaultdict


# =========================
# YOLO 모델 로드
# =========================
signal_model = YOLO("all_signal_augmentation.pt")
yolo_models = YOLO("best.pt")
split_model = YOLO("sp.pt")


COCO_CLASSES = {
    0: "person",
    15: "cat",
    16: "dog",
    80: "unknown"
}

SIGNAL_CLASSES = {
    0: "red",
    1: "yellow",
    2: "yellow",
    3: "yellow_green",
    4: "green"
}

CLASS_COLORS = {
    0: (0, 0, 0),           # 배경
    1: (255, 255, 255),     # 철로
}

MOVEMENT_THRESHOLD = 0.3


# =========================
# 스위치
# =========================
signal_switch = True
rail_stop_switch = True
red_signal_switch = True
yolo_detection_switch = True
do_inference = True
image_veiw_switch = True


# =========================
# 유틸 함수들
# =========================
def decode_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in CLASS_COLORS.items():
        color_mask[mask == class_idx] = color
    return color_mask


def postprocess_mask(mask, target_class=1, min_area=2500):
    binary = (mask == target_class).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    result = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            result[labels == i] = target_class
    return result


def preprocess_frame(frame):
    transform = A.Compose([
        A.Resize(512, 1024),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    transformed = transform(image=frame)
    return transformed["image"].unsqueeze(0)


def apply_roi_mask_by_params(
    mask,
    bottom_x1_ratio=0.3,
    bottom_x2_ratio=0.7,
    top_x1_ratio=0.45,
    top_x2_ratio=0.55,
    top_y_ratio=0.6
):
    h, w = mask.shape

    x1_bot = int(w * bottom_x1_ratio)
    x2_bot = int(w * bottom_x2_ratio)
    x1_top = int(w * top_x1_ratio)
    x2_top = int(w * top_x2_ratio)
    top_y = int(h * top_y_ratio)
    bottom_y = h - 1

    pts = np.array([
        [x1_bot, bottom_y],
        [x2_bot, bottom_y],
        [x2_top, top_y],
        [x1_top, top_y]
    ], dtype=np.int32)

    roi_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(roi_mask, [pts], 1)

    return mask * roi_mask, pts


def is_bbox_inside_roi(bbox, roi_mask):
    x1, y1, x2, y2 = map(int, bbox)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    if 0 <= cy < roi_mask.shape[0] and 0 <= cx < roi_mask.shape[1]:
        return roi_mask[cy, cx] > 0
    return False


def compute_optical_flow_with_mask(prev_gray, curr_gray, roi_mask):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=1, winsize=9,
        iterations=1, poly_n=3, poly_sigma=0.9, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    masked_mag = mag * roi_mask
    return np.sum(masked_mag) / (np.sum(roi_mask) + 1e-6)


# =========================
# 메인 실행 함수
# =========================
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BiSeNetV2(num_classes=2)
    model.load_state_dict(torch.load("bisenetv2_rail3.pth", map_location=device))
    model.to(device)
    model.eval()

    video_path = "C0001_VID.mp4"
    cap = cv2.VideoCapture(video_path)

    ret, origin_prev_frame = cap.read()
    if not ret:
        print("Cannot open video")
        return

    origin_resize_frame = cv2.resize(origin_prev_frame, (0, 0), fx=0.3, fy=0.3)
    origin_resize_gray = cv2.cvtColor(origin_resize_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_origin = frame.copy()
        frame = cv2.resize(frame, (1024, 512))
        frame_count += 1

        input_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess_frame(input_rgb).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

        if frame_count % 5 == 0:
            cleaned_mask = postprocess_mask(pred_mask, target_class=1, min_area=3000)
        else:
            cleaned_mask = pred_mask

        roi_mask, roi_pts = apply_roi_mask_by_params(cleaned_mask)

        overlay = frame.copy()
        if image_veiw_switch:
            overlay = cv2.addWeighted(overlay, 1.0, decode_mask(roi_mask), 0.3, 0)

        cv2.imshow("FrontCam - Detection", overlay)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================
# 단독 실행
# =========================
if __name__ == "__main__":
    cProfile.run("run()", "frontcam_profile.out")
