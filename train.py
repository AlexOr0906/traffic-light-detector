from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data=r'C:\Users\smena\Desktop\AI\Traffic Light Detection.v2i.yolov8\data.yaml',
    epochs=50,
    imgsz=320,
    batch=4,
    device='cpu',
    workers=2,
    name='traffic_lights'
)