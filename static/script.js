document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('webcam-video');
    const canvas = document.getElementById('webcam-canvas');
    const statusOverlay = document.getElementById('status-overlay');
    
    let isProcessing = false;

    // 1. Nyalakan Kamera Otomatis
    async function startCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 640 }, // Meminta resolusi kotak/standar
                audio: false 
            });
            video.srcObject = stream;
            
            // Mulai loop deteksi setelah video play
            video.onloadedmetadata = () => {
                video.play();
                loopDetection();
            };
        } catch (err) {
            statusOverlay.innerText = "❌ Gagal Akses Kamera";
            statusOverlay.style.color = "red";
            console.error(err);
        }
    }

    // 2. Loop Mengirim Frame ke Backend
    async function loopDetection() {
        if (isProcessing) {
            requestAnimationFrame(loopDetection);
            return;
        }
        
        isProcessing = true;

        // Siapkan canvas temp untuk snapshot
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        // Kirim ke backend
        tempCanvas.toBlob(async (blob) => {
            if (!blob) { isProcessing = false; requestAnimationFrame(loopDetection); return; }

            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');

            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                
                // Update UI
                drawResult(data);
            } catch (e) {
                console.error(e);
            } finally {
                isProcessing = false;
                requestAnimationFrame(loopDetection); // Lanjut ke frame berikutnya
            }
        }, 'image/jpeg', 0.6); // Kualitas JPEG 0.6 biar ngebut
    }

    // 3. Gambar Kotak & Status
    function drawResult(data) {
        // Sesuaikan ukuran canvas overlay dengan video asli
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Update Status Bar
        if (data.status_kelayakan === "TIDAK LAYAK") {
            statusOverlay.innerText = "⚠️ TIDAK LAYAK";
            statusOverlay.className = "status-rusak";
        } else {
            statusOverlay.innerText = "✅ LAYAK";
            statusOverlay.className = "status-layak";
        }

        // Gambar Kotak
        data.detections.forEach(det => {
            const [x1, y1, x2, y2] = det.coords;
            const color = det.class_id === 1 ? "red" : "#00ff00"; // Merah untuk Cacat

            ctx.lineWidth = 4;
            ctx.strokeStyle = color;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Label
            ctx.fillStyle = color;
            ctx.font = "bold 18px Arial";
            ctx.fillText(`${det.class} ${Math.round(det.confidence * 100)}%`, x1, y1 - 10);
        });
    }

    // Mulai aplikasi
    startCamera();
});