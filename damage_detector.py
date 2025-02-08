from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8s.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="D:/5.Semester/Freie_Entwurf/3.Kolloq/mapillary/yolov8/config.yaml", epochs=200)


