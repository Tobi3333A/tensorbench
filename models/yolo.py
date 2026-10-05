from ultralytics import YOLO


def load_model(model_path="yolo11n.pt"):
    return YOLO(model_path)

from ultralytics import YOLO


def load_model(model_path="yolo11n.pt"):
    return YOLO(model_path)


def export_onnx(model_path="yolo11n.pt"):
    model = YOLO(model_path)
    return model.export(format="onnx", imgsz=640)
