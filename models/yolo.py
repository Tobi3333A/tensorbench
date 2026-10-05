from ultralytics import YOLO


def load_model(model_path="yolo11n.pt"):
    return YOLO(model_path)