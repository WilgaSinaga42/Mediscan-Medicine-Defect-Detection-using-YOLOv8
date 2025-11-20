from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# --- KONFIGURASI MODEL ---
MODEL_PATH = "runs/detect/Train_GPU_RTX20507/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    print(f"⚠️ PERINGATAN: Model tidak ditemukan di {MODEL_PATH}")
else:
    print(f"✅ Model dimuat dari: {MODEL_PATH}")

MODEL = YOLO(MODEL_PATH)
CONF_THRESHOLD = 0.50

# === DEFINISI NAMA KELAS MANUAL (SOLUSI PASTI) ===
# Kita definisikan mapping sendiri. Tidak perlu mengubah internal model.
# 0 = Kemasan, 1 = Cacat
CUSTOM_NAMES = {0: 'Kemasan', 1: 'Cacat'}
# =================================================

# --- FUNGSI ANALISIS GAMBAR ---
def analyze_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    
    results = MODEL.predict(source=img_np, conf=CONF_THRESHOLD, verbose=False)
    
    is_defective = False
    detections = []
    
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            
            # --- MENGGUNAKAN CUSTOM MAPPING ---
            # Gunakan dictionary kita sendiri, bukan MODEL.names
            class_name = CUSTOM_NAMES.get(class_id, "Unknown") 
            
            confidence = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            
            detections.append({
                'class': class_name,
                'class_id': class_id,
                'confidence': round(confidence, 2),
                'coords': coords
            })
            
            # Logika Utama: Jika ada Kelas 1 (Cacat), maka TIDAK LAYAK
            if class_id == 1:
                is_defective = True

    status = "TIDAK LAYAK" if is_defective else "LAYAK"
    return status, detections

# --- ROUTES / HALAMAN ---
@app.route('/')
def index():
    return render_template('index.html')

# --- API PREDIKSI ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    
    try:
        image_bytes = file.read()
        status, detections = analyze_image(image_bytes)
        
        return jsonify({
            'status_kelayakan': status,
            'detections': detections,
            'total_cacat': len([d for d in detections if d['class_id'] == 1])
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)