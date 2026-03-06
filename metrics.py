from ultralytics import YOLO

model = YOLO(r'C:\Users\smena\Desktop\AI\runs\detect\traffic_lights4\weights\best.pt')

results = model.val(
    data=r'C:\Users\smena\Desktop\AI\Traffic Light Detection.v2i.yolov8\data.yaml'
)

print(f"mAP50:     {results.box.map50:.3f}")
print(f"Precision: {results.box.mp:.3f}")
print(f"Recall:    {results.box.mr:.3f}")