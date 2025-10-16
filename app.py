import streamlit as st
import numpy as np
import joblib
import json
import cv2
import pywt
from PIL import Image
import tempfile
import os
import json as _json

# Resolve resource paths relative to this file
BASE_DIR = os.path.join(os.path.dirname(__file__), "Model")
MODEL_PATH = os.path.join(BASE_DIR, "saved_model.pkl")
CLASS_DICT_PATH = os.path.join(BASE_DIR, "class_dictionary.json")
HAAR_DIR = os.path.join(BASE_DIR, "opencv", "haarcascades")

st.title("⚽ African Football Player Classifier")
st.write("Upload an image to predict which player it is!")

# Load model and class dictionary with friendly errors
model = None
class_dict = {}
try:
	model = joblib.load(MODEL_PATH)
except Exception as e:
	st.error(f"Model file not found or could not be loaded: {MODEL_PATH} -- {e}")

try:
	with open(CLASS_DICT_PATH, "r") as f:
		class_dict = json.load(f)
except Exception:
	st.error(f"Class dictionary not found or could not be read: {CLASS_DICT_PATH}")

# Load player info
player_info = {}
PLAYER_INFO_PATH = os.path.join(BASE_DIR, "player_info.json")
try:
	with open(PLAYER_INFO_PATH, 'r', encoding='utf-8') as pf:
		player_info = _json.load(pf)
except Exception:
	st.warning("Player info file not found or invalid: Model/player_info.json")

# Load Haar cascades
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
		roi_gray = gray[y:y + h, x:x + w]
		roi_color = img[y:y + h, x:x + w]
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
		scalled_raw_img.reshape(32 * 32 * 3, 1),
		scalled_img_har.reshape(32 * 32, 1)
	))
	return combined_img.reshape(1, -1).astype(float)


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
	with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
		tmp_file.write(uploaded_file.read())
		img_path = tmp_file.name

	img = cv2.imread(img_path)
	if img is None:
		st.error("⚠️ Could not read image file. Try again.")
	else:
		cropped, detection = detect_face_with_fallback(img)
		if cropped is None:
			st.warning("😕 No face detected. Try another image.")
		else:
			cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
			st.image(cropped_rgb, caption="Detected Face", use_container_width=True)

			if model is None:
				st.error("Model not loaded. Check `Model/saved_model.pkl` and requirements.")
			else:
				features = extract_features(cropped_rgb)
				try:
					prediction = model.predict(features)[0]
					proba = model.predict_proba(features)[0]
				except Exception as e:
					st.error(f"Prediction failed: {e}")
				else:
					player_name = [name for name, label in class_dict.items() if label == prediction][0]
					st.subheader(f"🎯 Predicted Player: {player_name}")
					st.write(f"Detection: {detection}")
					result_table = {name: float(proba[label] * 100) for name, label in class_dict.items()}
					st.table(result_table)

					# Debug toggle: show raw model outputs and mappings to diagnose wrong predictions
					if st.checkbox("Show debug info"):
						st.write("model.classes_:", getattr(model, 'classes_', None))
						st.write("raw prediction (model.predict):", int(prediction))
						# convert probabilities to a list for display
						try:
							proba_list = [float(x) for x in proba]
						except Exception:
							proba_list = list(map(float, proba))
						st.write("probabilities:", proba_list)
						st.write("class_dict (name->label):", class_dict)
						label_to_name = {v: k for k, v in class_dict.items()}
						st.write("label_to_name (label->name):", label_to_name)
						mapped_name = label_to_name.get(int(prediction), None)
						st.write("mapped name via class_dict:", mapped_name)

						# Fancy result layout: columns and player card
						col1, col2 = st.columns([1, 1.25])
						with col1:
							st.image(cropped_rgb, caption="Detected Face", use_container_width=True)
						with col2:
							st.markdown("### Player information")
							if mapped_name and mapped_name in player_info:
								p = player_info[mapped_name]
								# player photo (if available) else show name only
								try:
									img_path = p.get('image')
									if img_path and os.path.exists(img_path):
										st.image(img_path, width=220)
								except Exception:
									pass
								st.markdown(f"**Name:** {p.get('full_name', mapped_name)}")
								st.markdown(f"**Position:** {p.get('position', 'N/A')}")
								st.markdown(f"**Club:** {p.get('club', 'N/A')}")
								st.markdown(f"**Country:** {p.get('country', 'N/A')}")
								st.markdown(f"**About:** {p.get('bio', '')}")
							else:
								st.info(f"No player info available for {mapped_name}")
							# Probabilities chart
							try:
								import pandas as pd
								probs_df = pd.DataFrame([
									{"player": name, "prob": float(proba[label])}
									for name, label in class_dict.items()
								])
								probs_df = probs_df.sort_values('prob', ascending=True)
								st.bar_chart(probs_df.set_index('player')['prob'])
							except Exception:
								st.write("Could not render probability chart")
