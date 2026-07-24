# 🛡️ Network Intrusion Detection System (NIDS)

A **Machine Learning-based Network Intrusion Detection System (NIDS)** that detects and classifies malicious network traffic using multiple Machine Learning algorithms. The best-performing model (**XGBoost**) is deployed through an interactive **Streamlit** web application for easy prediction and testing.

## 🌐 Live Demo

🔗 **https://network-intrusion-detection-system-utsav.streamlit.app/**

---

## 📌 Features

- 📂 Upload a CSV file for batch prediction
- 📝 Manual prediction using network traffic features
- 📊 Interactive prediction summary and visualization
- 📥 Download prediction results as CSV
- 🤖 Detects multiple cyber attack types
- ⚡ Fast predictions using a trained XGBoost model
- 🌐 Live Streamlit web application

---

## 🚀 Supported Attack Types

- ✅ Normal Traffic
- 🚨 DoS
- 🚨 DDoS
- 🚨 Port Scanning
- 🚨 Brute Force
- 🚨 Bots
- 🚨 Web Attacks

---

## 📊 Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- **XGBoost (Selected Model)**

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 97.51% | 97.70% | 97.51% | 97.51% |
| Decision Tree | 99.83% | 99.83% | 99.83% | 99.83% |
| Random Forest | 99.82% | 99.82% | 99.82% | 99.82% |
| **XGBoost** | **99.90%** | **99.90%** | **99.90%** | **99.90%** |

---

# 📂 Project Structure

```text
NIDS/
│
├── app.py
├── README.md
├── requirements.txt
├── runtime.txt
├── .gitignore
│
├── .streamlit/
├── charts/
├── img/
│
└── models/
    ├── network_intrusion_detector.pkl
    ├── label_encoder.pkl
    ├── decision_tree.pkl
    ├── logistic_regression.pkl
    ├── random_forest.pkl
    ├── scaler.pkl
    ├── xgboost.pkl
    └── model_comparison.csv
```

---

# ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/utsavgangadiya/Network-Intrusion-Detection-System.git
```

### Navigate to the project

```bash
cd Network-Intrusion-Detection-System
```

### Create a virtual environment (Optional)

```bash
python -m venv venv
```

### Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
streamlit run app.py
```

---

# 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**.

**Live Application**

🔗 https://network-intrusion-detection-system-utsav.streamlit.app/

---

# 📈 Project Workflow

```text
CICIDS2017 Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Exploratory Data Analysis (EDA)
        │
        ▼
Data Preprocessing
        │
        ▼
Train-Test Split
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
        │
        ▼
Best Model Selection (XGBoost)
        │
        ▼
Model Serialization (Joblib)
        │
        ▼
Streamlit Deployment
```

---

# 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Matplotlib

---

# 📂 Dataset

This project uses the **CICIDS2017** dataset, which contains labeled network traffic representing both normal activity and multiple intrusion types. It is widely used for training and evaluating machine learning models for intrusion detection.

---

# 📋 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Main libraries:

- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Matplotlib

---

# 📌 Notes

- The trained model is loaded directly from the **models/** folder, so retraining is not required.
- The uploaded CSV file must follow the same feature format used during model training.
- Both CSV upload and manual prediction are supported.

---

# 🔮 Future Improvements

- Real-time packet capture
- Live network monitoring dashboard
- Deep Learning-based intrusion detection
- Explainable AI (SHAP)
- Docker containerization
- Cloud deployment with AWS or Azure

---

# 👨‍💻 Author

**Utsav Gangadiya**

**📊 Data Analyst | 🤖 Machine Learning & AI | Python • SQL • Power BI | Turning Data into Insights**

- 🔗 GitHub: https://github.com/utsavgangadiya
- 💼 LinkedIn: https://www.linkedin.com/in/utsav-gangadiya/

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!
