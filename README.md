# ForteBank Fraud Detection Hackathon

## 🎯 Goal
Develop an end‑to‑end ML anti‑fraud solution for mobile banking transactions, covering data preparation, feature engineering, model training, interpretation, and a demo MVP (Streamlit UI + FastAPI API).

## 📂 Repository Structure
```
forte/
├─ data/                 # raw & processed data
├─ models/               # trained models (*.pkl)
├─ src/                  # source code
│   ├─ features/         # feature engineering logic
│   ├─ models/           # model wrappers
│   └─ utils/            # helpers
├─ reports/              # metrics, SHAP plots, evaluation tables
├─ api.py                # FastAPI service
├─ app.py                # Streamlit demo UI
├─ train.py              # Main training script
├─ inference.py          # Inference logic
├─ requirements.txt      # Python dependencies
├─ QUICKSTART.md         # Fast start guide
└─ README.md             # Project overview
```

## 🚀 How to Run

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Train Model (Optional)
If you have data in `data/`:
```bash
python train.py
```
*Note: A pre-trained model is already included in `models/`.*

### 3. Run Demo UI
```bash
streamlit run app.py
```
Open http://localhost:8501

### 4. Run API (Optional)
```bash
python api.py
```
API docs: http://localhost:8000/docs

## 📊 What is Delivered
- **ML model** (CatBoost) with probability output and configurable threshold.
- **Feature set** (30+ engineered features) covering transaction, temporal, amount, client behavior, and velocity.
- **Interpretability** via SHAP (global & local) integrated into UI.
- **MVP**: Streamlit web app + FastAPI endpoint.
- **Metrics**: precision, recall, Fβ, ROC‑AUC (reported in `reports/`).
- **Video demo** (`video/demo.mp4`) and **presentation** (`presentation/presentation.pdf`).
- **GitHub repo** ready for submission.
