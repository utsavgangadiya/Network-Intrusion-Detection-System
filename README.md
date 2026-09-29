# Network Intrusion Detection System

A Streamlit application for classifying network-flow records as normal traffic or one of six attack categories. The project uses a trained XGBoost model and the CICIDS2017 dataset, with tools for bulk scanning, manual tests, model comparison, and dataset exploration.

## Recent Updates

- Added a separate **Dataset Insights** sidebar page immediately after Model Performance.
- Added a gallery for the 11 PNG visualizations in `CICIDS2017_Charts/`.
- Added a 10-row data preview, full-sample totals, attack-class distribution, and column details.
- Added a hosted fallback using the small preview CSV and JSON summary because the full 130 MB CSV is excluded from Git.
- Documented the model/encoder compatibility checks and the corrected training order for fitting the label encoder.

## Features

- **Dashboard:** model summary, evaluation accuracy, workflow, and supported classes.
- **Threat Scanner:** upload CSV, Excel, or JSON traffic records, review the input, run predictions, inspect severity summaries, and download results.
- **Manual Prediction:** test a single flow using editable values or example traffic presets.
- **Model Performance:** compare saved evaluation metrics for Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- **Dataset Insights:** browse a sample of the CICIDS2017 data, see dataset totals and class distribution, inspect column details, and view the project's generated charts.
- **Prediction integrity:** decode model outputs with the saved label encoder, validate model/encoder compatibility at startup, and align uploaded features to the model's expected feature order.

## Supported Classes

- Normal Traffic
- DoS
- DDoS
- Port Scanning
- Bots
- Brute Force
- Web Attacks

## Model Results

The following results are recorded in the project's evaluation materials. Performance can vary with the training data and evaluation setup.

| Model | Accuracy | Precision | Recall | F1 Score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 97.51% | 97.70% | 97.51% | 97.51% |
| Decision Tree | 99.83% | 99.83% | 99.83% | 99.83% |
| Random Forest | 99.82% | 99.82% | 99.82% | 99.82% |
| XGBoost | 99.90% | 99.90% | 99.90% | 99.90% |

## Dataset and Deployment Assets

The CICIDS2017 data contains labeled network-flow records. The full local file, `cicids2017_sample.csv`, is approximately 130 MB and is intentionally excluded from Git.

The Dataset Insights page uses the full CSV when it is available locally. If it is not present, including on a hosted deployment, the app uses:

- `cicids2017_preview.csv`: a lightweight, 10-row preview.
- `cicids2017_overview.json`: summary totals, label counts, and column details derived from the full sample.

The fallback summary represents the full 500,000-row sample even though only ten rows are bundled for display. Keep both fallback files in the repository when deploying. If the source dataset changes, regenerate the preview and summary before publishing the update; do not add the large full CSV to Git.

## Run Locally

Python 3.11 is specified in `runtime.txt`.

### Windows

```powershell
git clone https://github.com/utsavgangadiya/Network-Intrusion-Detection-System.git
cd Network-Intrusion-Detection-System
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS or Linux

```bash
git clone https://github.com/utsavgangadiya/Network-Intrusion-Detection-System.git
cd Network-Intrusion-Detection-System
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app requires the model and encoder artifacts in `models/`. A local copy of the full sample CSV is optional; the bundled preview assets are sufficient for Dataset Insights.

## Deploy on Streamlit Community Cloud

1. Push the application code, `models/`, `CICIDS2017_Charts/`, `cicids2017_preview.csv`, and `cicids2017_overview.json` to GitHub.
2. Create an app in Streamlit Community Cloud and select this repository and branch.
3. Set the app's main file to `app.py` and deploy.

The full `cicids2017_sample.csv` is ignored by Git, so the hosted app uses the bundled preview and overview files automatically. The chart images are loaded from `CICIDS2017_Charts/`.

## Training

`train_model.py` accepts a full labeled CSV containing an `Attack Type` column. It cleans labels, filters rare classes before fitting the label encoder, trains XGBoost, validates that the model and encoder use the same class mapping, and saves the artifacts under `models/`.

```bash
python train_model.py path/to/full_training_dataset.csv
```

Do not use the 10-row preview or a small validation file to train the model. `repair_label_encoder.py` is a separate utility for restoring the authoritative class mapping recorded in `nids.ipynb`.

## Project Structure

```text
.
├── app.py
├── train_model.py
├── repair_label_encoder.py
├── requirements.txt
├── runtime.txt
├── README.md
├── cicids2017_preview.csv
├── cicids2017_overview.json
├── CICIDS2017_Charts/        # Generated dataset visualizations
├── models/                   # Trained models, encoder, and comparison metrics
├── nids.ipynb                # Analysis and label-mapping reference
└── train_models.ipynb        # Model experiments
```

The full `cicids2017_sample.csv` may also exist in a local working copy, but it is not tracked by Git.

## Technology

Python, Streamlit, Pandas, NumPy, scikit-learn, XGBoost, Joblib, Matplotlib, and OpenPyXL.

## Author

Utsav Gangadiya

[GitHub](https://github.com/utsavgangadiya)
