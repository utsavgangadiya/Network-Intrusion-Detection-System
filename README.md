# 🛡️ Network Intrusion Detection System (NIDS)

A **Machine Learning-based Network Intrusion Detection System (NIDS)** developed using the **CICIDS2017** dataset. This project detects malicious network traffic and classifies different types of cyber attacks using multiple machine learning algorithms. The final deployed model is **XGBoost**, integrated into an interactive **Streamlit** web application.

---

## 📌 Features

- 📂 Upload a CSV file for batch network traffic prediction
- 📝 Manual prediction using sample feature values
- 📊 Interactive prediction summary and visualization
- 📥 Download prediction results as CSV
- 🤖 Detects multiple cyber attack types
- ⚡ Fast predictions using a trained XGBoost model
- 🌐 Ready for deployment on Streamlit Cloud

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
- **XGBoost (Best Model)**

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
├── nids.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── network_intrusion_detector.pkl
│   ├── label_encoder.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── model_comparison.csv
│
├── charts/
│
├── cicids2017_cleaned.csv
├── cicids2017_final_cleaned.csv
└── cicids2017_sample.csv
```

---

# ⚙️ Setup Locally

### 1. Clone the repository

```bash
git clone https://github.com/utsavgangadiya/Network-Intrusion-Detection-System.git
```

### 2. Navigate to the project

```bash
cd Network-Intrusion-Detection-System
```

### 3. (Optional) Create a virtual environment

```bash
python -m venv venv
```

Activate it

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

---

# 🌐 Deploy on Streamlit Cloud

1. Push the project to GitHub.
2. Open **Streamlit Community Cloud**.
3. Click **New App**.
4. Select your GitHub repository.
5. Set the main file as:

```text
app.py
```

6. Click **Deploy**.

---

# 📈 Project Workflow

```text
Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Preprocessing
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

**Dataset:** CICIDS2017

The CICIDS2017 dataset contains labeled network traffic flows used for intrusion detection research. It includes multiple attack categories and normal traffic records, making it suitable for training and evaluating machine learning models for network security.

---

# 📋 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Main libraries used:

- Streamlit
- Pandas
- NumPy
- scikit-learn
- XGBoost
- Joblib
- Matplotlib

---

# 📌 Notes

- The trained model files are stored in the **models/** folder so the application can make predictions without retraining.
- The application expects input data to follow the same feature format used during training (CICIDS2017).
- Sample CSV files are included for testing and demonstration purposes.

---

# 🔮 Future Improvements

- Real-time packet capture
- Live intrusion monitoring dashboard
- Deep Learning-based intrusion detection
- SHAP model explainability
- Docker support
- Cloud deployment (AWS / Azure)

---

# 👨‍💻 Author

**Utsav Gangadiya**

**📊 Data Analyst | 🤖 Machine Learning & AI | Python • SQL • Power BI | Turning data into decisions**

- 🔗 GitHub: https://github.com/utsavgangadiya
- 💼 LinkedIn: https://www.linkedin.com/in/utsav-gangadiya/

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!
