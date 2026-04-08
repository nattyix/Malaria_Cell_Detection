"""
Malaria Detection - 5G Ready Flask Web App
==========================================
Simulates a real-world 5G-enabled telemedicine system where:
- A mobile/field device captures a blood smear image
- Sends it over 5G network to this server
- Gets instant AI prediction back
"""

from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time
import datetime

app = Flask(__name__)

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
print("Loading malaria detection model...")
model = tf.keras.models.load_model("malaria_model_final.h5")
print("Model loaded successfully!")

IMG_SIZE = (128, 128)

# Store prediction history (acts like a mini database)
prediction_history = []

# ─────────────────────────────────────────────
#  HTML INTERFACE (Clean UI)
# ─────────────────────────────────────────────
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Malaria Detection System — 5G Enabled</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            min-height: 100vh;
            color: white;
            padding: 20px;
        }

        .container { max-width: 850px; margin: 0 auto; }

        .header {
            text-align: center;
            padding: 30px 0 20px;
        }

        .header h1 { font-size: 2rem; margin-bottom: 8px; }

        .badge-5g {
            display: inline-block;
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }

        .subtitle { color: #a0c4ff; font-size: 0.95rem; }

        .stats-bar {
            display: flex;
            justify-content: center;
            gap: 30px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 15px;
            margin: 20px 0;
            flex-wrap: wrap;
        }

        .stat { text-align: center; }
        .stat-value { font-size: 1.4rem; font-weight: bold; color: #00c6ff; }
        .stat-label { font-size: 0.75rem; color: #aaa; margin-top: 2px; }

        .card {
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 20px;
        }

        .card h2 { margin-bottom: 20px; font-size: 1.1rem; color: #00c6ff; }

        .upload-area {
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 20px;
        }

        .upload-area:hover { border-color: #00c6ff; background: rgba(0,198,255,0.05); }
        .upload-area p { color: #aaa; margin-top: 10px; font-size: 0.9rem; }

        #imagePreview {
            max-width: 200px;
            max-height: 200px;
            border-radius: 10px;
            display: none;
            margin: 15px auto;
        }

        input[type="file"] { display: none; }

        .btn {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }

        .btn-primary {
            background: linear-gradient(90deg, #00c6ff, #0072ff);
            color: white;
        }

        .btn-primary:hover { opacity: 0.9; transform: translateY(-1px); }
        .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        .result-card {
            display: none;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 20px;
            text-align: center;
        }

        .result-infected {
            background: rgba(255, 70, 70, 0.15);
            border: 1px solid rgba(255, 70, 70, 0.4);
        }

        .result-healthy {
            background: rgba(0, 220, 130, 0.15);
            border: 1px solid rgba(0, 220, 130, 0.4);
        }

        .result-emoji { font-size: 3rem; margin-bottom: 10px; }
        .result-label { font-size: 1.5rem; font-weight: bold; margin-bottom: 8px; }
        .result-confidence { font-size: 1rem; color: #ccc; margin-bottom: 15px; }

        .confidence-bar-bg {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            height: 10px;
            margin: 10px auto;
            max-width: 400px;
        }

        .confidence-bar-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 1s ease;
        }

        .meta-info {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }

        .meta-badge {
            background: rgba(255,255,255,0.1);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            color: #ccc;
        }

        .history-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 15px;
            background: rgba(255,255,255,0.04);
            border-radius: 10px;
            margin-bottom: 8px;
            font-size: 0.88rem;
        }

        .history-dot {
            width: 10px; height: 10px;
            border-radius: 50%;
            margin-right: 10px;
            flex-shrink: 0;
        }

        .dot-infected { background: #ff4646; }
        .dot-healthy  { background: #00dc82; }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #00c6ff;
        }

        .spinner {
            border: 3px solid rgba(255,255,255,0.1);
            border-top: 3px solid #00c6ff;
            border-radius: 50%;
            width: 40px; height: 40px;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
<div class="container">

    <div class="header">
        <div class="badge-5g">⚡ 5G ENABLED</div>
        <h1>🦟 Malaria Detection System</h1>
        <p class="subtitle">AI-powered blood smear analysis via 5G telemedicine network</p>
    </div>

    <div class="stats-bar">
        <div class="stat">
            <div class="stat-value">94.3%</div>
            <div class="stat-label">Model Accuracy</div>
        </div>
        <div class="stat">
            <div class="stat-value">0.9846</div>
            <div class="stat-label">AUC-ROC Score</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="totalPredictions">0</div>
            <div class="stat-label">Total Predictions</div>
        </div>
        <div class="stat">
            <div class="stat-value" id="avgLatency">—</div>
            <div class="stat-label">Avg Latency (ms)</div>
        </div>
    </div>

    <div class="card">
        <h2>📤 Upload Blood Smear Image</h2>
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div style="font-size:2.5rem">🔬</div>
            <p>Click to upload or drag & drop a cell image</p>
            <p style="font-size:0.8rem; margin-top:5px;">PNG, JPG supported</p>
        </div>
        <input type="file" id="fileInput" accept="image/*" onchange="previewImage(event)">
        <img id="imagePreview" src="" alt="Preview">
        <button class="btn btn-primary" id="predictBtn" onclick="predict()" disabled>
            🚀 Send via 5G & Analyze
        </button>
    </div>

    <div class="loading" id="loading">
        <div class="spinner"></div>
        <p>Transmitting over 5G network... Running AI analysis...</p>
    </div>

    <div class="result-card" id="resultCard">
        <div class="result-emoji" id="resultEmoji"></div>
        <div class="result-label" id="resultLabel"></div>
        <div class="result-confidence" id="resultConfidence"></div>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" id="confidenceBar"></div>
        </div>
        <div class="meta-info" id="metaInfo"></div>
    </div>

    <div class="card">
        <h2>📋 Prediction History</h2>
        <div id="historyList">
            <p style="color:#666; font-size:0.9rem;">No predictions yet. Upload an image to begin.</p>
        </div>
    </div>

</div>

<script>
    let totalPredictions = 0;
    let totalLatency = 0;
    let selectedFile = null;

    function previewImage(event) {
        const file = event.target.files[0];
        if (!file) return;
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            const preview = document.getElementById('imagePreview');
            preview.src = e.target.result;
            preview.style.display = 'block';
            document.getElementById('predictBtn').disabled = false;
        };
        reader.readAsDataURL(file);
    }

    async function predict() {
        if (!selectedFile) return;

        document.getElementById('loading').style.display = 'block';
        document.getElementById('resultCard').style.display = 'none';
        document.getElementById('predictBtn').disabled = true;

        const formData = new FormData();
        formData.append('image', selectedFile);

        const startTime = performance.now();

        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();
            const latency = Math.round(performance.now() - startTime);

            document.getElementById('loading').style.display = 'none';
            showResult(data, latency);
            addToHistory(data, latency);
            updateStats(latency);

        } catch (err) {
            document.getElementById('loading').style.display = 'none';
            alert('Error connecting to server: ' + err.message);
        }

        document.getElementById('predictBtn').disabled = false;
    }

    function showResult(data, latency) {
        const card = document.getElementById('resultCard');
        const isInfected = data.prediction === 'Parasitized';

        card.className = 'result-card ' + (isInfected ? 'result-infected' : 'result-healthy');
        card.style.display = 'block';

        document.getElementById('resultEmoji').textContent = isInfected ? '🦟' : '✅';
        document.getElementById('resultLabel').textContent = isInfected
            ? 'Malaria Detected (Parasitized)'
            : 'No Malaria (Uninfected)';
        document.getElementById('resultLabel').style.color = isInfected ? '#ff6b6b' : '#00dc82';

        document.getElementById('resultConfidence').textContent = `Confidence: ${data.confidence}`;

        const bar = document.getElementById('confidenceBar');
        bar.style.width = '0%';
        bar.style.background = isInfected
            ? 'linear-gradient(90deg, #ff4646, #ff8c00)'
            : 'linear-gradient(90deg, #00dc82, #00c6ff)';
        setTimeout(() => { bar.style.width = data.confidence; }, 100);

        document.getElementById('metaInfo').innerHTML = `
            <span class="meta-badge">⚡ 5G Latency: ${latency}ms</span>
            <span class="meta-badge">🕐 ${data.timestamp}</span>
            <span class="meta-badge">🤖 MobileNetV2</span>
        `;
    }

    function addToHistory(data, latency) {
        const list = document.getElementById('historyList');
        const isInfected = data.prediction === 'Parasitized';

        if (list.children[0]?.style.color === 'rgb(102, 102, 102)') {
            list.innerHTML = '';
        }

        const item = document.createElement('div');
        item.className = 'history-item';
        item.innerHTML = `
            <div style="display:flex; align-items:center;">
                <div class="history-dot ${isInfected ? 'dot-infected' : 'dot-healthy'}"></div>
                <span>${isInfected ? '🦟 Parasitized' : '✅ Uninfected'}</span>
            </div>
            <span style="color:#aaa">${data.confidence} confidence</span>
            <span class="meta-badge">⚡ ${latency}ms</span>
            <span style="color:#666; font-size:0.8rem">${data.timestamp}</span>
        `;
        list.insertBefore(item, list.firstChild);
    }

    function updateStats(latency) {
        totalPredictions++;
        totalLatency += latency;
        document.getElementById('totalPredictions').textContent = totalPredictions;
        document.getElementById('avgLatency').textContent =
            Math.round(totalLatency / totalPredictions) + 'ms';
    }
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def home():
    return render_template_string(HTML_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    start_time = time.time()

    # Read and preprocess image
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run prediction
    prob = float(model.predict(img_array, verbose=0)[0][0])
    inference_time = round((time.time() - start_time) * 1000, 1)

    # ─────────────────────────────────────────────
    # FIX: classes=["Uninfected", "Parasitized"]
    #   → Uninfected = 0, Parasitized = 1
    #   → sigmoid output (prob) represents P(Parasitized)
    #   → prob > 0.5  means  Parasitized
    # ─────────────────────────────────────────────
    prediction = "Parasitized" if prob > 0.5 else "Uninfected"
    confidence = prob if prob > 0.5 else 1 - prob
    timestamp  = datetime.datetime.now().strftime("%H:%M:%S")

    result = {
        "prediction":      prediction,
        "confidence":      f"{confidence:.1%}",
        "raw_probability": round(prob, 4),
        "inference_ms":    inference_time,
        "timestamp":       timestamp,
        "model":           "MobileNetV2",
        "network":         "5G Simulated"
    }

    prediction_history.append(result)
    return jsonify(result)


@app.route("/history")
def history():
    """API endpoint to get all prediction history"""
    return jsonify({
        "total_predictions": len(prediction_history),
        "history": prediction_history
    })


@app.route("/stats")
def stats():
    """API endpoint to get model + network stats"""
    if not prediction_history:
        return jsonify({"message": "No predictions yet"})

    infected = sum(1 for p in prediction_history if p["prediction"] == "Parasitized")
    avg_ms   = round(sum(p["inference_ms"] for p in prediction_history) / len(prediction_history), 1)

    return jsonify({
        "total_predictions": len(prediction_history),
        "infected_count":    infected,
        "uninfected_count":  len(prediction_history) - infected,
        "infection_rate":    f"{infected / len(prediction_history):.1%}",
        "avg_inference_ms":  avg_ms,
        "network_type":      "5G Simulated",
        "model_accuracy":    "94.3%",
        "model":             "MobileNetV2 (Fine-tuned)"
    })


# ─────────────────────────────────────────────
#  RUN SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  🦟 Malaria Detection — 5G Server")
    print("="*50)
    print("  Open in browser: http://localhost:5000")
    print("  API endpoints:")
    print("    POST /predict  — Upload image, get result")
    print("    GET  /history  — All predictions")
    print("    GET  /stats    — System statistics")
    print("="*50 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000)