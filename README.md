# African-Football-Player-Face-Classifier-Hakimi-Salah-Mahrez-Minteh-
This project is a machine learning–based image classification system that identifies African football players from face images. It uses computer vision (OpenCV) for face detection, wavelet transforms for feature extraction, and a Support Vector Machine (SVM) classifier for prediction.
The entire system is integrated into a Streamlit web application, where users can upload a photo and get the predicted player name.

Dataset Preparation

Dataset collected manually from Google Images for 4 players:

Achraf Hakimi

Mohamed Salah

Riyad Mahrez

Yankuba Minteh

Images are stored under:
./Dataset/
├── Achraf Hakimi/
├── Mohamed Salah/
├── Riyad Mahrez/
└── Yankuba Minteh/

Preprocessing & Feature Extraction

Face Detection:

Used OpenCV’s haarcascade_frontalface_default.xml and haarcascade_eye.xml.

Function get_cropped_image_if_2_eyes() ensures that only faces with both eyes visible are cropped and saved.

Data Augmentation (especially for Yankuba Minteh):

Increased small dataset images using rotation, brightness change, and horizontal flipping.

Helps balance the dataset and improve model generalization.

Wavelet Transform:

Applied discrete wavelet transform (pywt.dwt2) to extract texture-based features from face images.

Combined wavelet and raw pixel features into a single feature vector for model training.

🧩 Model Training

Three models were tested:

SVM (Support Vector Machine)

Random Forest

Logistic Regression

Cross-validation and grid search were used for hyperparameter tuning.

Best performing model:

Model	Accuracy	Best Parameters
SVM	0.90	{'C': 1, 'kernel': 'linear'}
Logistic Regression	0.84	{'C': 1}
Random Forest	0.72	{'n_estimators': 10}

Web Application

Built using Streamlit, the web app allows users to:

Upload an image.

Detect and crop the face.

Display predicted player name.

View class probabilities and confidence.

Core technologies:

OpenCV – Face detection

PyWavelets – Feature extraction

scikit-learn – Model training (SVM)

python flask – Web deployment



Results

Achieved ~90% accuracy using SVM.

Real-time face detection and classification works well for most test images.

Limited data for Yankuba Minteh affects balance slightly, but augmentation improved performance.

🏁 Future Improvements

Collect more images, especially for underrepresented players.

Use CNN (e.g., MobileNet or ResNet) for end-to-end feature learning.

Add more African players to make the dataset more diverse.

🙌 Acknowledgments

Dataset collected manually by Modou [Your Full Name].

Inspired by Grameen-style project approach (community-built dataset).

Built as part of portfolio work for internship applications.
