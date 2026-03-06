from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64


app = Flask(__name__)
CORS(app)

model = YOLO(r'C:\Users\smena\Desktop\AI\runs\detect\traffic_lights4\weights\best.pt')

CLASS_COLORS = {
    'red':    (0, 0, 255),
    'green':  (0, 255, 0),
    'yellow': (0, 255, 255),
    'off':    (128, 128, 128),
}

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400

    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img, imgsz=640)

    detections = []
    for r in results:
        img_annotated = r.plot()
        for box in r.boxes:
            cls = r.names[int(box.cls)]
            conf = box.conf.item()
            detections.append({
                'class': cls,
                'confidence': round(conf, 2)
            })

    _, buffer = cv2.imencode('.jpg', img_annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': img_base64,
        'detections': detections
    })

if __name__ == '__main__':
    print("Сервер запущен на http://localhost:5000")
    app.run(debug=False, port=5000)