from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import uvicorn
import os  

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model, scaler, PCA
model = load_model("models/heart_cnn_model.h5")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    age: float = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: float = Form(...),
    chol: float = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: float = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...),
    slope: int = Form(...)
):
    df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "chest pain type": cp,
        "resting bp s": trestbps,
        "cholesterol": chol,
        "fasting blood sugar": fbs,
        "resting ecg": restecg,
        "max heart rate": thalach,
        "exercise angina": exang,
        "oldpeak": oldpeak,
        "ST slope": slope
    }])

    # Interaction features
    df['age_resting_bp'] = df['age'] * df['resting bp s']
    df['age_oldpeak'] = df['age'] * df['oldpeak']
    df['cholesterol_max_heart_rate'] = df['cholesterol'] * df['max heart rate']
    df['resting_bp_oldpeak'] = df['resting bp s'] * df['oldpeak']
    df['sex_chest_pain_type'] = df['sex'] * df['chest pain type']
    df['sex_fasting_blood_sugar'] = df['sex'] * df['fasting blood sugar']
    df['sex_resting_ecg'] = df['sex'] * df['resting ecg']
    df['sex_exercise_angina'] = df['sex'] * df['exercise angina']
    df['sex_st_slope'] = df['sex'] * df['ST slope']
    df['chest_pain_type_fasting_blood_sugar'] = df['chest pain type'] * df['fasting blood sugar']
    df['chest_pain_type_resting_ecg'] = df['chest pain type'] * df['resting ecg']
    df['chest_pain_type_exercise_angina'] = df['chest pain type'] * df['exercise angina']
    df['chest_pain_type_st_slope'] = df['chest pain type'] * df['ST slope']
    df['fasting_blood_sugar_resting_ecg'] = df['fasting blood sugar'] * df['resting ecg']
    df['fasting_blood_sugar_st_slope'] = df['fasting blood sugar'] * df['ST slope']
    df['resting_ecg_exercise_angina'] = df['resting ecg'] * df['exercise angina']
    df['resting_ecg_st_slope'] = df['resting ecg'] * df['ST slope']
    df['exercise_angina_st_slope'] = df['exercise angina'] * df['ST slope']

    features_scaled = scaler.transform(df)
    features_pca = pca.transform(features_scaled)
    features_cnn = features_pca.reshape(features_pca.shape[0], features_pca.shape[1], 1)

    prediction = model.predict(features_cnn)[0][0]
    result = "Heart Disease Detected" if prediction > 0.5 else "No Heart Disease"

    return templates.TemplateResponse("result.html", {"request": request, "result": result})

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # auto-detect PORT
    uvicorn.run(app, host="0.0.0.0", port=port)
