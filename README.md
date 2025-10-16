<img width="1689" height="871" alt="image" src="https://github.com/user-attachments/assets/0fa77dc7-df3d-4509-970e-f853a7a73a98" />

# African-Football-Player-Face-Classifier-Hakimi-Salah-Mahrez-Minteh-
This project is a machine learning–based image classification system that identifies African football players from face images. It uses computer vision (OpenCV) for face detection, wavelet transforms for feature extraction, and a Support Vector Machine (SVM) classifier for prediction.
The entire system is integrated into a Streamlit web application, where users can upload a photo and get the predicted player name.



## Overview

This project demonstrates the integration of **AI and Web Development**.  
It automatically detects faces from uploaded images, extracts features using **Wavelet Transform**, and predicts the correct player using a pre-trained **SVM model**.  

Users interact with a simple web interface built using **HTML, CSS, and JavaScript**, while the backend is powered by **Python Flask**.

---

##  Players in the Dataset

| Player Name | Country |
|--------------|----------|
| Achraf Hakimi | 🇲🇦 Morocco |
| Mohamed Salah | 🇪🇬 Egypt |
| Riyad Mahrez | 🇩🇿 Algeria |
| Yankuba Minteh | 🇬🇲 Gambia |

> *Note: The dataset was manually collected. Yankuba Minteh’s dataset was limited, so image augmentation techniques  were applied to increase the samples.*

---

## Machine Learning Overview

- **Face Detection:** Haar Cascade Classifier (OpenCV)  
- **Feature Extraction:** Wavelet Transform (`pywt`) + raw pixel features  
- **Model Used:** Support Vector Machine (SVM)  
- **Framework:** scikit-learn  
- **Accuracy:** ~90% (SVM performed best among tested models)

---

## 📊 Model Performance

| Model | Accuracy | Best Parameters |
|--------|-----------|----------------|
| 🧩 SVM | **0.90** | {'C': 1, 'kernel': 'linear'} |
| 🌲 Random Forest | 0.72 | {'n_estimators': 10} |
| 📈 Logistic Regression | 0.84 | {'C': 1} |

**Confusion Matrix (SVM):**


---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mlkinteh2/African-Football-Player-Face-Classifier-Hakimi-Salah-Mahrez-Minteh.git
cd African-Football-Player-Face-Classifier-Hakimi-Salah-Mahrez-Minteh

