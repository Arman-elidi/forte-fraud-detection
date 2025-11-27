# Project Structure Overview

This document describes the folder layout for the hackathon repository.

```
forte/
├─ data/                 # raw & processed data (git‑ignored)
├─ src/                  # source code
│   ├─ __init__.py
│   ├─ data_loader.py    # Load & clean transaction & behavioral data
│   ├─ feature_engineering.py  # Feature pipeline (30+ features)
│   └─ models/
│       ├─ __init__.py
│       ├─ fraud_model.py # CatBoost wrapper, training, inference
│       └─ explainer.py   # SHAP explanations
├─ api/                  # FastAPI service
│   └─ api.py
├─ app/                  # Streamlit demo UI
│   └─ app.py
├─ notebooks/            # EDA & experiments
├─ demo/                 # Demo CSV files (transactions, behavioral, merged)
├─ reports/              # Metrics, SHAP plots, evaluation tables
├─ scripts/              # Helper scripts (run_all.sh, generate_presentation.py)
├─ video/                # Demo video (demo.mp4, ≤3 min)
├─ presentation/         # PDF/PPT presentation (presentation.pdf)
├─ walkthrough.md        # Summary of work performed
├─ README.md             # Project overview (generated earlier)
├─ requirements.txt      # Python dependencies
└─ project_structure.md # **This file** – detailed description
```

Each folder contains a `README.md` with usage instructions (optional).
