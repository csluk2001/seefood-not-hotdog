from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.yaml")  # build a new model from scratch

# Use the modelS
model.train(data="../config.yaml", imgsz=640, epochs=100)  # train the model

