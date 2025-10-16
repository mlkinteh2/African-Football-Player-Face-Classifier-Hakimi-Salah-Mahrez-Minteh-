<img width="1645" height="880" alt="image" src="https://github.com/user-attachments/assets/461569c1-644a-48eb-8ac9-5a3c9f1d0874" />

#  African Football Player Face Classifier (Flask Web App)

A Flask-based web application that recognizes African football players from their facial images using **Machine Learning** and **Computer Vision**.  
The app uses **OpenCV**, **Wavelet Transform**, and a **Support Vector Machine (SVM)** model to classify uploaded images of players.

---

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

> *Note: The dataset was manually collected. Yankuba Minteh’s dataset was limited, so image augmentation techniques (flipping, rotation, brightness adjustment) were applied to increase the samples.*

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

## 🧩 Tech Stack

| Area | Technology |
|------|-------------|
| Backend | Python (Flask) |
| Frontend | HTML, CSS, JavaScript |
| Machine Learning | scikit-learn (SVM) |
| Computer Vision | OpenCV |
| Feature Extraction | PyWavelets |
| Data Handling | NumPy, Pandas |

---



## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mkinteh2/African-Football-Player-Face-Classifier-Hakimi-Salah-Mahrez-Minteh.git
cd African-Football-Player-Face-Classifier-Hakimi-Salah-Mahrez-Minteh

