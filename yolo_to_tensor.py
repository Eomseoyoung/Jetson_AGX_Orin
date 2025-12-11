# export_to_trt_engine.py (하드코딩 버전)
from ultralytics import YOLO

PT_PATH   = "yolov8n.pt"  # 학습된 nano 가중치
IMGSZ     = (640, 640)                               # 학습/추론 동일 크기 권장
OUT_ENGINE = "ylolv8n.engine"

if __name__ == "__main__":
    model = YOLO(PT_PATH)
    model.export(
        format="engine",   # TensorRT
        half=True,         # FP16
        dynamic=False,     # 정적 640x640 권장
        imgsz=IMGSZ,
        device=0           # Jetson GPU
    )
    print("Exported:", OUT_ENGINE)
