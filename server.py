from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import joblib
import json
import numpy as np
import cv2
import pywt
import threading
import time

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.join(os.path.dirname(__file__), 'Model')
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model.pkl')
CLASS_DICT_PATH = os.path.join(BASE_DIR, 'class_dictionary.json')
PLAYER_INFO_PATH = os.path.join(BASE_DIR, 'player_info.json')
HAAR_DIR = os.path.join(BASE_DIR, 'opencv', 'haarcascades')

# Model state (loaded asynchronously to avoid blocking server start)
model = None
class_dict = {}
model_loaded = False
model_error = None
model_lock = threading.Lock()

def load_model_async():
    global model, model_loaded, model_error, class_dict
    try:
        # small sleep to allow server to start cleanly
        time.sleep(0.1)
        m = joblib.load(MODEL_PATH)
        with model_lock:
            model = m
            model_loaded = True
            model_error = None
    except Exception as e:
        with model_lock:
            model = None
            model_loaded = False
            model_error = str(e)

    # try to load class dictionary too (best-effort)
    try:
        with open(CLASS_DICT_PATH, 'r', encoding='utf-8') as f:
            class_dict = json.load(f)
    except Exception:
        class_dict = {}
    # load player info mapping
    try:
        with open(PLAYER_INFO_PATH, 'r', encoding='utf-8') as f:
            player_info = json.load(f)
    except Exception:
        player_info = {}
    # attach to global for use in responses
    globals()['player_info'] = player_info

# cascades (safe to construct now)
face_cascade = cv2.CascadeClassifier(os.path.join(HAAR_DIR, 'haarcascade_frontalface_default.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(HAAR_DIR, 'haarcascade_eye.xml'))

def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

def detect_face_with_fallback(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color, '2_eyes'
        if len(eyes) == 1:
            return roi_color, '1_eye'
        return roi_color, 'face'
    return None, None

def extract_features(img):
    scalled_raw_img = cv2.resize(img, (32, 32))
    img_har = w2d(img, 'db1', 5)
    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((
        scalled_raw_img.reshape(32*32*3, 1),
        scalled_img_har.reshape(32*32, 1)
    ))
    return combined_img.reshape(1, -1).astype(float)


@app.route('/health')
def health():
    with model_lock:
        return jsonify({'model_loaded': model_loaded, 'model_error': model_error, 'num_classes': len(class_dict)})


@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded', 'details': model_error}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    cropped, detection = detect_face_with_fallback(img)
    if cropped is None:
        return jsonify({'error': 'No face detected'}), 200

    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    features = extract_features(cropped_rgb)
    try:
        pred = int(model.predict(features)[0])
        proba = model.predict_proba(features)[0].tolist()
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    # map label to internal key (e.g. 'Mohamed Salah')
    name_key = None
    for n, label in class_dict.items():
        if label == pred:
            name_key = n
            break
    # friendly display name from player_info if available
    display_name = None
    player_info = globals().get('player_info', {})
    if name_key and name_key in player_info:
        display_name = player_info[name_key].get('full_name')
    if not display_name:
        display_name = name_key

    prob_map = {n: float(round(proba[label] * 100, 4)) for n, label in class_dict.items()}

    _, imbuf = cv2.imencode('.jpg', cropped)
    import base64
    img_b64 = base64.b64encode(imbuf.tobytes()).decode('utf-8')

    return jsonify({'player_name': display_name, 'player_key': name_key, 'prediction': pred, 'probabilities': prob_map, 'face_image': img_b64, 'detection': detection})


if __name__ == '__main__':
    # start model loading in background so the HTTP server is available immediately
    t = threading.Thread(target=load_model_async, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=True)
