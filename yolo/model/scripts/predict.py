from ultralytics import YOLO

model = YOLO("./yolo/model/train_result/trained_model.pt")

# source = "../test_data/test_video_1.mp4"
source = "./yolo/model/test/test1.jpeg"

model.predict(source=source, show=True, save=True, conf=0.7)
