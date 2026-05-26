import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import normalize
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from supabase import create_client, Client
import asyncio

# 1. SUPABASE CONFIG
SUPABASE_URL = "https://svzrazwfbojkdshpudck.supabase.co"
SUPABASE_KEY = 
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. FASTAPI INIT
app = FastAPI(title="NeuroPredict Platform")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = r"E:\OneDrive\Desktop\AD\uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def is_mri_image(file_path: str) -> bool:
    try:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False

        h, w = img.shape
        # MRIs are usually square-ish
        if h < 64 or w < 64 or abs(h - w) > 40:
            return False

        # Contrast must be enough
        if np.std(img) < 25:
            return False

        # Mean intensity check (MRIs are usually mid-range)
        mean_intensity = np.mean(img)
        if mean_intensity < 30 or mean_intensity > 220:
            return False

        # Optional: basic edge detection to check for brain-like structures
        edges = cv2.Canny(img, 100, 200)
        if np.sum(edges) < 1000:  # too few edges -> unlikely MRI
            return False

        return True
    except:
        return False


# 3. CNN + SVM PIPELINE CLASS
class NeuroPredictor:
    def __init__(self, cnn_path, svm_path, scaler_path):
        # Load CNN model
        self.cnn_model = tf.keras.models.load_model(cnn_path, compile=False)
        _ = self.cnn_model.predict(tf.random.normal([1, 128, 128, 1]))  # Warm-up

        # Feature extractor: output of layer before Dense (256 features)
        self.feature_extractor = tf.keras.Model(
            inputs=self.cnn_model.inputs[0],
            outputs=self.cnn_model.layers[11].output
        )

        # Load SVM + scaler
        self.svm = joblib.load(svm_path)
        self.scaler = joblib.load(scaler_path)

        # Categories
        self.CATEGORIES = ['Non Demented', 'Very mild Dementia', 'Mild Dementia', 'Moderate Dementia']

    def preprocess_image(self, file_path: str):
        """Load and preprocess a single MRI image."""
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found: {file_path}")
        img = cv2.resize(img, (128, 128))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)  # (128,128,1)
        img = np.expand_dims(img, axis=0)   # (1,128,128,1)
        return img

    def extract_features(self, img_path: str):
        img_tensor = self.preprocess_image(img_path)
        features = self.feature_extractor(img_tensor, training=False).numpy()
        return features

    def predict(self, img_path: str, normalize_features: bool = True):
        features = self.extract_features(img_path)
        if normalize_features:
            features = normalize(features, norm='l2')
        features_scaled = self.scaler.transform(features)
        pred_class = self.svm.predict(features_scaled)[0]
        pred_probs = self.svm.predict_proba(features_scaled)[0]
        label = self.CATEGORIES[pred_class]
        return {
            "predicted_class": label,
            "probabilities": {self.CATEGORIES[i]: float(pred_probs[i]) for i in range(len(self.CATEGORIES))}
        }

# 4. LOAD MODELS ON STARTUP
neuro_ai = NeuroPredictor(
    cnn_path='models/final_alzheimer_model.h5',
    svm_path='models/svm_256.pkl',
    scaler_path='models/scaler_256.pkl'
)

# 5. API ENDPOINTS

# Home / dashboard
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/assess{num}.html", response_class=HTMLResponse)
async def serve_pages(request: Request, num: int):
    return templates.TemplateResponse(f"assess{num}.html", {"request": request})

@app.get("/dashboard.html", response_class=HTMLResponse)
async def dashboard_redirect(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# Upload MRI scan
@app.post("/api/upload-scan")
async def upload_scan(mri_file: UploadFile = File(...)):
    try:
        content = await mri_file.read()
        file_path = os.path.join(UPLOAD_DIR, mri_file.filename)
        with open(file_path, "wb") as f:
            f.write(content)

        # Check if the uploaded image is MRI
        if not is_mri_image(file_path):
            os.remove(file_path)
            return JSONResponse(
                {"status": "error", "message": "Uploaded file does not appear to be a valid MRI scan."},
                status_code=400
            )

        return JSONResponse({"status": "success", "filename": mri_file.filename})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


# Predict MRI scan
@app.post("/api/predict")
async def predict_scan(mri_filename: str = Form(...)):
    file_path = os.path.join(UPLOAD_DIR, mri_filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        result = neuro_ai.predict(file_path, normalize_features=True)
        return JSONResponse({"status": "success", **result})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# Submit assessment (keep your existing DB logic)
@app.post("/api/submit-assessment")
async def submit_assessment(
    patientName: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    hand: str = Form(...),
    education: int = Form(...),
    mmse: float = Form(...),
    mri_filename: str = Form(...),
    prediction: str = Form(...)
):
    local_path = os.path.join(UPLOAD_DIR, mri_filename)
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="MRI file not found")
    try:
        supabase_path = f"scans/{patientName.replace(' ', '_')}_{mri_filename}"

        # Upload MRI scan
        def upload_file():
            with open(local_path, "rb") as f:
                return supabase.storage.from_("mri-scans").upload(
                    supabase_path,
                    f.read(),
                    file_options={"x-upsert": "true"}
                )

        await asyncio.to_thread(upload_file)

        # Insert patient record
        def insert_db():
            return supabase.table("patients").insert({
                "patient_name": patientName,
                "age": age,
                "gender": gender,
                "hand": hand,
                "education_years": education,
                "mmse_score": mmse,
                "mri_scan_url": supabase_path,
                "prediction_result": prediction
            }).execute()

        res = await asyncio.to_thread(insert_db)

        return {"status": "success", "patient_id": res.data[0]["id"]}

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# 6. RUN SERVER
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
