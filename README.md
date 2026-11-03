# NeuroPredict Platform (Alzheimer's Detection & Prediction)

FastAPI web app that classifies brain MRI scans into four dementia stages using a
**CNN feature extractor + SVM** pipeline, stores patient assessments and scan
images in Supabase, and serves a multi-step assessment UI. A **Vision Transformer
(ViT)** model is also evaluated on the same task and its results are reported below
for comparison.

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset & Classes](#dataset--classes)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Run](#run)
- [API Endpoints](#api-endpoints)
- [How It Works](#how-it-works)
- [Model Performance](#model-performance)
  - [Evaluation Setup](#evaluation-setup)
  - [Results Summary](#results-summary)
  - [CNN + SVM (External Test)](#cnn--svm-external-test)
  - [Vision Transformer (ViT)](#vision-transformer-vit)
  - [Confusion Matrices](#confusion-matrices)
  - [Observations](#observations)
- [Verify Supabase Connection](#verify-supabase-connection)
- [Troubleshooting](#troubleshooting)
- [GitHub: Clone / Pull](#github-clone--pull)
- [Notes](#notes)

## Features

- MRI upload with basic MRI validity checks (contrast, dimensions, edge detection)
- CNN feature extraction + SVM classification (4 classes)
- Vision Transformer (ViT) evaluated on the same 4-class task (results reported below)
- Supabase storage for scans and patient records
- 6-step multi-page assessment flow + dashboard (Jinja2 templates)
- Patient assessment form with MMSE scoring
- Runs with Uvicorn (hot reload supported)
- Environment-based configuration via `.env` file

## Tech Stack

- Python 3.10
- FastAPI, Uvicorn, Jinja2
- TensorFlow/Keras 3.x, scikit-learn, OpenCV, NumPy, joblib
- Supabase (Postgres + Storage)
- python-dotenv for environment management

## Project Structure

```
Alzheimer-s-Detection-and-Prediction/
├── main.py                  # FastAPI app + ML pipeline
├── check_db.py              # Supabase connection check
├── requirements.txt         # Python dependencies
├── models/                  # CNN (.h5), SVM (.pkl), scaler (.pkl)
│   ├── final_alzheimer_model.h5
│   ├── svm_256.pkl
│   └── scaler_256.pkl
├── templates/               # HTML pages
│   ├── dashboard.html       # Main dashboard
│   ├── assessment.html      # Assessment landing page
│   ├── assess1.html - assess6.html  # 6-step assessment forms
│   ├── patientAssessment.html
│   └── download_file.html
├── uploads/                 # Uploaded MRI scans (runtime)
├── .venv/                   # Local virtual environment (not committed)
├── .env                     # Environment variables (not committed)
└── README.md
```

> The ViT training/inference code is not part of this repository; only its
> evaluation results are included for comparison.

## Dataset & Classes

The models are trained/evaluated on brain MRI data based on the **OASIS** (Open
Access Series of Imaging Studies) dataset. The deployed app uses the pre-trained
artifacts in `models/` (`final_alzheimer_model.h5`, `svm_256.pkl`,
`scaler_256.pkl`).

The task is 4-class dementia staging:

| Class index | Class name | External test support |
|-------------|------------|-----------------------|
| 0 | Non Demented | 1,334 |
| 1 | Very Mild Dementia | 1,014 |
| 2 | Mild Dementia | 1,127 |
| 3 | Moderate Dementia | 1,307 |
| | **Total** | **4,782** |

## Prerequisites

- Python 3.10
- (Recommended) `uv` for fast installs: https://docs.astral.sh/uv/
- Supabase account with:
  - `patients` table (patient_name, age, gender, hand, education_years, mmse_score, mri_scan_url, prediction_result)
  - `mri-scans` storage bucket

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

Create a `.env` file in the project root (already gitignored):

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-secret-key
```

The app uses `python-dotenv` to load these automatically.

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
| GET | `/` | Dashboard (home) |
| GET | `/dashboard.html` | Dashboard page |
| GET | `/assess{num}.html` | Assessment pages (1-6) |
| POST | `/api/upload-scan` | Upload MRI image (validates MRI format) |
| POST | `/api/predict` | Predict dementia stage from uploaded MRI |
| POST | `/api/submit-assessment` | Save patient assessment + upload scan to Supabase |

## How It Works

1. **Upload**: User uploads an MRI; `is_mri_image()` validates it (checks dimensions, contrast, mean intensity, edge detection).
2. **Feature Extraction**: `NeuroPredictor` loads the CNN model, extracts 256 features from layer 11.
3. **Preprocessing**: Features are L2-normalized and scaled using the fitted scaler.
4. **Classification**: SVM predicts one of 4 classes: Non Demented, Very Mild Dementia, Mild Dementia, Moderate Dementia.
5. **Assessment**: Multi-step form collects patient demographics, MMSE score, handedness, education.
6. **Storage**: Assessment data stored in Supabase `patients` table; MRI scan uploaded to `mri-scans` bucket.

## Model Performance

### Evaluation Setup

- **Task**: 4-class dementia staging (Non Demented, Very Mild Dementia, Mild Dementia, Moderate Dementia).
- **Test set**: 4,782 MRI images with class supports shown in [Dataset & Classes](#dataset--classes).
- **Metrics**: per-class precision, recall, F1-score, support; plus macro/weighted averages.
- **CNN + SVM**: results below are from the **retrained SVM and scaler** on the external test set.
- **ViT**: results below are from the Vision Transformer evaluation. Class indices `0–3`
  in the ViT report follow the same class order used by the CNN + SVM report.

### Results Summary

| Model | Macro Precision | Macro Recall | Macro F1 |
|-------|-----------------|--------------|----------|
| CNN + SVM (External Test) | 0.89 | 0.88 | 0.88 |
| Vision Transformer (ViT) | 0.8871 | 0.8752 | 0.8771 |

### CNN + SVM (External Test)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Non Demented | 0.83 | 0.95 | 0.88 | 1,334 |
| Very Mild Dementia | 0.84 | 0.77 | 0.80 | 1,014 |
| Mild Dementia | 0.92 | 0.89 | 0.90 | 1,127 |
| Moderate Dementia | 0.98 | 0.91 | 0.95 | 1,307 |

| Aggregate | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Macro avg | 0.89 | 0.88 | 0.88 | 4,782 |
| Weighted avg | 0.89 | 0.89 | 0.89 | 4,782 |

### Vision Transformer (ViT)

The ViT model is evaluated as an alternative to the CNN + SVM pipeline. It handles
the same 4-class task and the same 4,782-image test set.

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Non Demented (0) | 0.8293 | 0.9873 | 0.9014 | 1,334 |
| Very Mild Dementia (1) | 0.7749 | 0.8215 | 0.7975 | 1,014 |
| Mild Dementia (2) | 0.9477 | 0.8199 | 0.8792 | 1,127 |
| Moderate Dementia (3) | 0.9965 | 0.8722 | 0.9302 | 1,307 |

| Aggregate | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Macro avg | 0.8871 | 0.8752 | 0.8771 | 4,782 |
| Weighted avg | 0.8914 | 0.8812 | 0.8820 | 4,782 |

### Confusion Matrices

Class order: `Non Demented, Very Mild Dementia, Mild Dementia, Moderate Dementia`.

**CNN + SVM (External Test)**

```
[[1269   15   48    2]
 [ 184  783   36   11]
 [  58   59 1004    6]
 [  26   77    9 1195]]
```

**Vision Transformer (ViT)**

```
[[1317   15    2    0]
 [ 159  833   18    4]
 [ 102  101  924    0]
 [  10  126   31 1140]]
```

### Observations

- CNN + SVM has slightly higher macro precision (0.89 vs 0.8871), macro recall
  (0.88 vs 0.8752), and macro F1 (0.88 vs 0.8771) on the reported results.
- ViT has higher recall on **Non Demented** (0.9873 vs 0.95) and **Very Mild
  Dementia** (0.8215 vs 0.77), and higher precision on **Mild Dementia**
  (0.9477 vs 0.92) and **Moderate Dementia** (0.9965 vs 0.98).
- CNN + SVM has higher recall on **Mild Dementia** (0.89 vs 0.8199) and
  **Moderate Dementia** (0.91 vs 0.8722).
- **Very Mild Dementia** is the hardest class for both models (F1 ≈ 0.80), and
  most misclassifications occur between adjacent stages.

## Verify Supabase Connection

```bash
python check_db.py
```

## Troubleshooting

- `Error loading ASGI app. Could not import module "main"`: run uvicorn from the
  project directory with the venv interpreter.
- Yellow import warnings in editor: select the `.venv` interpreter.
- Old/stale paths: `UPLOAD_DIR` is relative to the project folder; remove
  `__pycache__/` to clear cached paths.
- Model load errors: confirm `models/final_alzheimer_model.h5`, `models/svm_256.pkl`,
  and `models/scaler_256.pkl` are present.
- Supabase errors: verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env`, and that the
  `patients` table and `mri-scans` bucket exist.

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

- `models/`, `uploads/`, `.venv/`, and `.env` are gitignored.
- Keep Supabase keys out of version control.
- The CNN model expects 128x128 grayscale input images.
- MRI validation rejects images that are too small, non-square, low contrast, or lack brain-like edge structures.
- The deployed pipeline uses CNN + SVM. ViT results are included for comparison; the
  ViT implementation/training code is not part of this repository.
- Reported CNN + SVM metrics come from a retrained SVM and scaler evaluated on the
  external test set. Reported ViT metrics come from the ViT evaluation.
