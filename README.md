# Network Intrusion Detection System (NIDS)

This project implements a Streamlit-based Network Intrusion Detection System that uses a trained machine learning model to classify network traffic as normal or suspicious.

## Features
- Upload a CSV file of network traffic data
- Make manual predictions using sample feature values
- View prediction results and summaries directly in the browser
- Ready for deployment on Streamlit Cloud

## Project Structure
- app.py - Main Streamlit application
- models/ - Trained model artifacts
- charts/ - Visuals and evaluation outputs
- data files - CICIDS2017 related CSV files used for training and testing

## Setup Locally
1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deploy on Streamlit Cloud
1. Push this repository to GitHub
2. Open Streamlit Cloud
3. Click New app
4. Select the GitHub repository and set the main file to app.py
5. Deploy

## Requirements
The app uses:
- Streamlit
- Pandas
- NumPy
- scikit-learn
- joblib
- xgboost

## Notes
The trained model files are stored in the models folder so that the app can run without retraining.
