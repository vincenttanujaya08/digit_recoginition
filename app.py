# app.py
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Mengizinkan semua origin agar frontend dari port lain (misal 8000) bisa mengakses

# -------------------------------
# 1. Muat bobot NumPy hasil pelatihan
# -------------------------------
data = np.load('mnist_weights.npz')
W1, b1 = data['W1'], data['b1']
W2, b2 = data['W2'], data['b2']

# -------------------------------
# 2. Fungsi aktivasi dan softmax
# -------------------------------
def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# -------------------------------
# 3. Threshold confidence untuk “gambar ulang”
# -------------------------------
CONFIDENCE_THRESHOLD = 0.8  # ubah sesuai kebutuhan (misal 0.75 atau 0.80)

# -------------------------------
# 4. Endpoint /predict
# -------------------------------
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Preflight CORS (OPTIONS):
    if request.method == 'OPTIONS':
        return '', 200

    # Tangkap payload JSON
    data_json = request.get_json()
    if not data_json or 'image' not in data_json:
        return jsonify({'error': 'Missing "image" field'}), 400

    arr28 = np.array(data_json['image'], dtype=np.float32)
    if arr28.shape != (28, 28):
        return jsonify({'error': 'Format image harus 28×28'}), 400

    # Flatten → forward pass
    x = arr28.reshape(1, 28 * 28)      # (1, 784)
    Z1 = np.dot(x, W1) + b1            # (1, 128)
    A1 = relu(Z1)                      # (1, 128)
    Z2 = np.dot(A1, W2) + b2           # (1, 10)
    A2 = softmax(Z2)                   # (1, 10)

    # Ambil prediksi dan confidence
    probs = A2[0]                      # bentuk (10,)
    pred = int(np.argmax(probs))       # kelas dengan probabilitas tertinggi
    confidence = float(np.max(probs))  # nilai probabilitas tertinggi

    # Jika kurang yakin, kirim digit=None untuk meminta gambar ulang
    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({'digit': None, 'confidence': confidence})

    # Kalau yakin, kirim digit dan confidence
    return jsonify({'digit': pred, 'confidence': confidence})

# -------------------------------
# 5. Jalankan server Flask
# -------------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)
