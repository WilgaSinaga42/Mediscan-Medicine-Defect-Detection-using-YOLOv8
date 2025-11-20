from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# --- KONFIGURASI MODEL ---
MODEL_PATH = "runs/detect/Train_GPU_RTX2050/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    print(f"⚠️ PERINGATAN: Model tidak ditemukan di {MODEL_PATH}")
else:
    print(f"✅ Model dimuat dari: {MODEL_PATH}")

MODEL = YOLO(MODEL_PATH)

# --- SETTING FOKUS (PENTING) ---
# 1. Confidence tinggi: Hanya deteksi jika yakin > 60%
CONF_THRESHOLD = 0.60 
# 2. IOU rendah: Hapus kotak tumpang tindih (biar tidak double)
IOU_THRESHOLD = 0.50  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No frame received'}), 400
        
    file = request.files['file']
    
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(image)
        
        # --- JALANKAN DETEKSI DENGAN FILTER KETAT ---
        results = MODEL.predict(
            source=img_np, 
            conf=CONF_THRESHOLD, 
            iou=IOU_THRESHOLD,   # Filter kotak tumpang tindih
            max_det=1,           # OPSIONAL: Paksa cuma deteksi 1 objek utama (kalau obatnya pasti cuma 1)
            verbose=False
        )
        
        is_defective = False
        detections = []
        
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = MODEL.names[class_id]
                confidence = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                
                detections.append({
                    'class': class_name,
                    'class_id': class_id,
                    'confidence': round(confidence, 2),
                    'coords': coords
                })
                
                # Logika Cacat
                if class_id == 1: # Asumsi 1 = Cacat
                    is_defective = True

        status = "TIDAK LAYAK" if is_defective else "LAYAK"
        
        return jsonify({
            'status_kelayakan': status,
            'detections': detections
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)