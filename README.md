# African Player Classification

This project provides a Streamlit application that predicts which African football player appears in an uploaded photo.

Quick steps to run locally (PowerShell on Windows):

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

Notes:
- The trained model is at `Model/saved_model.pkl` and the class mapping is `Model/class_dictionary.json`.
- Haar cascades are in `Model/opencv/haarcascades` and `app.py` loads them from there.
- If unpickling fails with ModuleNotFoundError for `sklearn` or other libraries, ensure those packages are installed in your environment.
