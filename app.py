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
import base64

# Page Config
st.set_page_config(
	page_title="African Football Player Classifier",
	page_icon="⚽",
	layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
	/* Import Google Font */
	@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

	html, body, [class*="css"] {
		font-family: 'Outfit', sans-serif;
	}

	/* Main container padding */
	.main .block-container {
		padding-top: 2rem;
		padding-bottom: 2rem;
	}

	/* Hero Section */
	.hero-header {
		text-align: center;
		margin-bottom: 3rem;
	}
	.hero-title {
		font-size: 3rem;
		font-weight: 700;
		background: -webkit-linear-gradient(45deg, #1e3a8a, #3b82f6);
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		margin-bottom: 0.5rem;
	}
	.hero-subtitle {
		font-size: 1.1rem;
		color: #555;
	}

	/* Player Avatars Row */
	.avatar-container {
		display: flex;
		justify-content: center;
		gap: 2rem;
		margin-bottom: 3rem;
		flex-wrap: wrap;
	}
	.player-avatar {
		display: flex;
		flex-direction: column;
		align-items: center;
		width: 120px;
	}
	.avatar-img {
		width: 100px;
		height: 100px;
		object-fit: cover;
		border-radius: 50%;
		border: 3px solid #eee;
		transition: all 0.3s ease;
		box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
	}
	.avatar-img:hover {
		transform: scale(1.05);
		border-color: #3b82f6;
		box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
	}
	.player-name-label {
		margin-top: 0.5rem;
		font-weight: 600;
		font-size: 0.9rem;
		color: #333;
		text-align: center;
	}

	/* Upload Zone Styling */
	div[data-testid="stFileUploader"] {
		width: 100%;
		padding: 2rem;
		border: 2px dashed #cbd5e1;
		border-radius: 1rem;
		text-align: center;
		transition: border-color 0.3s;
	}
	/* Streamlit doesn't expose clean css hooks for uploader, but we can style the container */

	/* Results Area */
	.prediction-card {
		background: white;
		border-radius: 1.5rem;
		padding: 2rem;
		box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
		text-align: center;
		margin-top: 2rem;
	}
	.pred-img {
		width: 180px;
		height: 180px;
		object-fit: cover;
		border-radius: 50%;
		border: 5px solid #3b82f6;
		margin-bottom: 1rem;
	}
	.pred-name {
		font-size: 2rem;
		font-weight: 700;
		color: #1f2937;
		margin-bottom: 0.5rem;
	}
	.pred-prob {
		font-size: 1.2rem;
		color: #3b82f6;
		font-weight: 600;
		margin-bottom: 1.5rem;
	}

	/* Progress Bars customization */
	.stProgress > div > div > div > div {
		background-image: linear-gradient(to right, #3b82f6, #60a5fa);
	}

</style>
""", unsafe_allow_html=True)


# ---- Constants & Logic ----
BASE_DIR = os.path.join(os.path.dirname(__file__), "Model")
MODEL_PATH = os.path.join(BASE_DIR, "saved_model.pkl")
CLASS_DICT_PATH = os.path.join(BASE_DIR, "class_dictionary.json")
PLAYER_INFO_PATH = os.path.join(BASE_DIR, "player_info.json")
HAAR_DIR = os.path.join(BASE_DIR, "opencv", "haarcascades")

# Initialize global vars
model = None
class_dict = {}
player_info = {}

# Load Assets
try:
	model = joblib.load(MODEL_PATH)
except Exception as e:
	st.error(f"Failed to load model: {e}")

try:
	with open(CLASS_DICT_PATH, "r") as f:
		class_dict = json.load(f)
except:
	pass

try:
	with open(PLAYER_INFO_PATH, 'r', encoding='utf-8') as pf:
		player_info = _json.load(pf)
except:
	pass

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


def get_base64_of_bin_file(bin_file):
	with open(bin_file, 'rb') as f:
		data = f.read()
	import base64
	return base64.b64encode(data).decode()


# ---- UI Structure ----

# 1. Hero Section
st.markdown("""
<div class="hero-header">
	<div class="hero-title">African Football Player Classifier</div>
	<div class="hero-subtitle">Upload a photo and the AI will predict which player it is.</div>
</div>
""", unsafe_allow_html=True)

# 2. Player Avatars (Native Streamlit Layout)
if player_info:
	# Create 4 columns for the avatars
	cols = st.columns(len(player_info))
	for idx, (pid, info) in enumerate(player_info.items()):
		with cols[idx]:
			st.markdown(f"<div style='text-align: center; font-weight: 600;'>{info.get('full_name', pid)}</div>", unsafe_allow_html=True)
			img_path = info.get('image', '').replace('\\', '/')
			if img_path and os.path.exists(img_path):
				try:
					st.image(img_path, output_format="JPEG", use_container_width=True)
				except Exception:
					st.warning(f"Could not load image for {info.get('full_name', pid)}")
			else:
				st.warning(f"No image for {info.get('full_name', pid)}")


# 3. Main Layout: Upload & Result
st.markdown("---") # Visual separator
col1, col2 = st.columns([1, 1], gap="large")

with col1:
	st.markdown("### 📸 Upload Image")
	uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

	if uploaded_file is None:
		st.info("Supported formats: .jpg, .jpeg, .png")


with col2:
	if uploaded_file and model:
		# Process Image
		with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
			tmp_file.write(uploaded_file.read())
			img_path = tmp_file.name

		img = cv2.imread(img_path)
		if img is None:
			st.error("Error loading image.")
		else:
			cropped, detection = detect_face_with_fallback(img)
			if cropped is None:
				st.warning("No face detected. Please use a clearer image.")
			else:
				# Prediction
				cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
				features = extract_features(cropped_rgb)
				try:
					prediction = model.predict(features)[0]
					proba = model.predict_proba(features)[0]

					# Get Info
					label_to_name = {v: k for k, v in class_dict.items()}
					mapped_name = label_to_name.get(int(prediction), "Unknown")
					
					p_data = player_info.get(mapped_name, {})
					display_name = p_data.get('full_name', mapped_name)
					
					# Get highest probability
					max_prob = max(proba) * 100
					
					# --- Result Card ---
					# Show the cropped face as the circle
					from io import BytesIO
					pil_img = Image.fromarray(cropped_rgb)
					buff = BytesIO()
					pil_img.save(buff, format="JPEG")
					img_str = base64.b64encode(buff.getvalue()).decode()
					
					st.markdown(f"""
					<div class="prediction-card">
						<img src="data:image/jpeg;base64,{img_str}" class="pred-img">
						<div class="pred-name">{display_name}</div>
						<div class="pred-prob">Confidence: {max_prob:.1f}%</div>
					</div>
					""", unsafe_allow_html=True)
					
					st.markdown("### Probability Breakdown")
					# Detailed Probs
					probs_list = []
					for name, label in class_dict.items():
						probs_list.append({'name': player_info.get(name, {}).get('full_name', name), 'prob': proba[label]})
					
					# Sort descending
					probs_list.sort(key=lambda x: x['prob'], reverse=True)
					
					for item in probs_list:
						c1, c2 = st.columns([1, 3])
						c1.markdown(f"**{item['name']}**")
						c2.progress(item['prob'])
						
				except Exception as e:
					st.error(f"Prediction error: {e}")
	else:
		st.markdown("""
		<div style="text-align: center; color: #888; padding: 5rem 0;">
			<div style="font-size: 3rem; margin-bottom: 1rem;">👈</div>
			<div>Select an image to see the magic happen.</div>
		</div>
		""", unsafe_allow_html=True)

