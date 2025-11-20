# C:\mediscan\predict_yolo.py
from ultralytics import YOLO
import time
import os

# --- KONFIGURASI ---
# Menggunakan os.path.join untuk kompatibilitas path di Windows
MODEL_NAME_FOLDER = "FinalRun3" 
MODEL_PATH = os.path.join("runs", "detect", MODEL_NAME_FOLDER, "weights", "best.pt")

VIDEO_SOURCE = 0      # 0 untuk webcam, atau path ke file video (misalnya, "input.mp4")
CONF_THRESHOLD = 0.50 # Ambang batas keyakinan: HANYA deteksi di atas 50% yang akan diperhitungkan

# --- INISIALISASI ---
try:
    # Memastikan model best.pt dari FinalRun3 dimuat
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file tidak ditemukan di: {MODEL_PATH}")
        
    model = YOLO(MODEL_PATH)
    print(f"✅ Model {MODEL_PATH} berhasil dimuat.")
except Exception as e:
    print(f"❌ ERROR: Gagal memuat model. Pastikan path sudah benar. {e}")
    exit()

print("\n--- Memulai Deteksi Real-Time ---")
print("Status Layak/Tidak Layak akan diperbarui di terminal. Tekan Ctrl+C untuk keluar.")

# --- LOOP DETEKSI REAL-TIME (Output ke Terminal) ---
results = model.predict(
    source=VIDEO_SOURCE, 
    conf=CONF_THRESHOLD, 
    stream=True,        # Streaming data dari webcam
    verbose=False       # Menjaga terminal tetap bersih
)
PREV_TIME = time.time()

for r in results:
    
    # Hitung FPS
    NEW_TIME = time.time()
    fps = 1 / (NEW_TIME - PREV_TIME)
    PREV_TIME = NEW_TIME

    # Logika Kelayakan: Cek apakah ada deteksi untuk Kelas 1 (Cacat)
    # Kuncinya: Cacat adalah Kelas 1 (index 1) dalam data.yaml Anda.
    defect_boxes = r.boxes[r.boxes.cls == 1] 

    if len(defect_boxes) > 0:
        # Status: TIDAK LAYAK
        # Mengambil detail keyakinan dari setiap cacat yang terdeteksi
        detail = ", ".join([f"Cacat ({box.conf[0]:.2f})" for box in defect_boxes])
        print(f"\r⚠️ Status: TIDAK LAYAK | FPS: {fps:.1f} | Deteksi: {detail}", end="", flush=True)
    else:
        # Status: LAYAK
        print(f"\r🟢 Status: LAYAK (Aman) | FPS: {fps:.1f} ", end="", flush=True)

# Akhir Loop
print("\nDeteksi real-time selesai.")