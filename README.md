# NeuroPredict Platform (Alzheimer's Detection & Prediction)

FastAPI web app that classifies brain MRI scans into dementia stages using a
CNN + SVM pipeline, stores patient assessments and scan images in Supabase,
and serves a multi-step assessment UI.

## Features

- MRI upload with basic MRI validity checks
- CNN feature extraction + SVM classification (4 classes)
- Supabase storage for scans and patient records
- Multi-page assessment flow and dashboard (Jinja2 templates)
- Runs with Uvicorn (hot reload supported)

## Tech Stack

- Python 3.10
- FastAPI, Uvicorn, Jinja2
- TensorFlow/Keras, scikit-learn, OpenCV, NumPy, joblib
- Supabase (Postgres + Storage)

## Project Structure

```
Alzheimer-s-Detection-and-Prediction/
├── main.py                  # FastAPI app + ML pipeline
├── check_db.py              # Supabase connection check
├── requirements.txt         # Python dependencies
├── models/                  # CNN (.h5), SVM (.pkl), scaler (.pkl)
├── templates/               # HTML pages
├── uploads/                 # Uploaded MRI scans (runtime)
├── .venv/                   # Local virtual environment (not committed)
└── README.md
```

## Dataset

Training data is based on the **OASIS** (Open Access Series of Imaging Studies)
MRI dataset. The deployed app uses pre-trained artifacts in `models/`
(`final_alzheimer_model.h5`, `svm_256.pkl`, `scaler_256.pkl`).

## Prerequisites

- Python 3.10
- (Recommended) `uv` for fast installs: https://docs.astral.sh/uv/

## Setup

### Using uv (recommended)

```bash
cd Alzheimer-s-Detection-and-Prediction
uv venv --python 3.10 .venv
uv pip install -r requirements.txt
```

### Using pip

```bash
cd Alzheimer-s-Detection-and-Prediction
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Interpreter for editors

Use this interpreter to resolve imports (removes yellow warnings):

```
F:\intern\AD\Alzheimer-s-Detection-and-Prediction\.venv\Scripts\python.exe
```

## Environment Variables

Supabase credentials live in `main.py`/`check_db.py`. For a cleaner setup,
move them to a `.env` file (already gitignored):

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-secret-key
```

## Run

```bash
uv run uvicorn main:app --reload --port 8000
```

Or with the activated venv:

```bash
uvicorn main:app --reload --port 8000
```

Open http://127.0.0.1:8000

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard |
| GET | `/dashboard.html` | Dashboard |
| GET | `/assess{num}.html` | Assessment pages |
| POST | `/api/upload-scan` | Upload MRI image |
| POST | `/api/predict` | Predict dementia stage |
| POST | `/api/submit-assessment` | Save assessment + upload scan to Supabase |

## How It Works

1. User uploads an MRI; `is_mri_image()` validates it.
2. `NeuroPredictor` loads the CNN, extracts 256 features, normalizes and scales them.
3. SVM predicts one of: Non Demented, Very mild Dementia, Mild Dementia, Moderate Dementia.
4. Assessment data and scan are stored in Supabase (`patients` table + `mri-scans` bucket).

## GitHub: Clone / Pull

Clone your repo:

```bash
git clone https://github.com/lijala-shakya/Alzheimer-s-Detection-and-Prediction.git
cd Alzheimer-s-Detection-and-Prediction
```

Pull latest changes:

```bash
git pull origin main
```

## Notes

- `models/`, `uploads/`, and `.venv/` are gitignored where applicable.
- Keep Supabase keys out of version control.
"