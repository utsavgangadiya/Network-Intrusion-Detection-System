from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ===========================================
# Page configuration
# ===========================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STYLE_FILE = BASE_DIR / "style.css"

st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if STYLE_FILE.exists():
    with STYLE_FILE.open(encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ===========================================
# Load model artifacts
# ===========================================

@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "network_intrusion_detector.pkl"
    encoder_path = MODELS_DIR / "label_encoder.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Missing label encoder file: {encoder_path}")

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder


try:
    model, label_encoder = load_model()
except Exception as exc:
    st.error(f"Unable to load the trained model files. Error: {exc}")
    st.stop()


# ===========================================
# Sidebar
# ===========================================

st.sidebar.image(
    "https://img.icons8.com/color/96/security-shield-green.png",
    width=80,
)
st.sidebar.title("Network IDS")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📂 Upload CSV", "📝 Manual Prediction", "ℹ About"],
)


# ===========================================
# Helpers
# ===========================================


def prepare_input_df(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [col for col in model.feature_names_in_ if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    return df[model.feature_names_in_]


# ===========================================
# Pages
# ===========================================

if menu == "🏠 Home":
    st.title("🛡️ Network Intrusion Detection System")
    st.markdown(
        """
This application detects malicious network traffic using an **XGBoost machine learning model** trained on the **CICIDS2017** dataset.

### Supported Attack Types
- ✅ Normal Traffic
- 🚨 DoS
- 🚨 DDoS
- 🚨 Port Scanning
- 🚨 Bots
- 🚨 Brute Force
- 🚨 Web Attacks

---

### Workflow
Use the sidebar to upload a CSV file or make a manual prediction.
"""
    )
    st.info("Choose a page from the left sidebar to begin.")

elif menu == "📂 Upload CSV":
    st.title("📂 Upload Network Traffic")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Predict"):
            try:
                input_df = prepare_input_df(df)
                predictions = model.predict(input_df)
                labels = label_encoder.inverse_transform(predictions)

                result_df = df.copy()
                result_df["Prediction"] = labels

                st.success("Prediction completed successfully!")
                st.subheader("Prediction Result")
                st.dataframe(result_df, use_container_width=True)

                total = len(result_df)
                attacks = (result_df["Prediction"] != "Normal Traffic").sum()
                normal = total - attacks

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Records", total)
                col2.metric("Normal Traffic", normal)
                col3.metric("Attacks", attacks)

                st.subheader("Prediction Summary")
                st.bar_chart(result_df["Prediction"].value_counts())

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Result",
                    csv_bytes,
                    file_name="prediction.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

elif menu == "📝 Manual Prediction":
    st.title("📝 Manual Prediction")
    st.write("Enter the network traffic values below and click **Predict**.")

    manual_data = {feature: 0.0 for feature in model.feature_names_in_}

    col1, col2 = st.columns(2)
    with col1:
        manual_data["Destination Port"] = st.number_input("Destination Port", min_value=0.0, value=80.0)
        manual_data["Flow Duration"] = st.number_input("Flow Duration", min_value=0.0, value=1000.0)
        manual_data["Total Fwd Packets"] = st.number_input("Total Forward Packets", min_value=0.0, value=10.0)
        manual_data["Total Length of Fwd Packets"] = st.number_input("Total Length of Forward Packets", min_value=0.0, value=500.0)
        manual_data["Fwd Packet Length Max"] = st.number_input("Forward Packet Length Max", min_value=0.0, value=250.0)
        manual_data["Flow Bytes/s"] = st.number_input("Flow Bytes/s", min_value=0.0, value=1000.0)
        manual_data["Flow Packets/s"] = st.number_input("Flow Packets/s", min_value=0.0, value=100.0)
        manual_data["Packet Length Mean"] = st.number_input("Packet Length Mean", min_value=0.0, value=150.0)
        manual_data["Average Packet Size"] = st.number_input("Average Packet Size", min_value=0.0, value=200.0)
        manual_data["Init_Win_bytes_forward"] = st.number_input("Init Window Bytes Forward", min_value=0.0, value=8192.0)

    with col2:
        manual_data["Bwd Packet Length Mean"] = st.number_input("Backward Packet Length Mean", min_value=0.0, value=100.0)
        manual_data["Flow IAT Mean"] = st.number_input("Flow IAT Mean", min_value=0.0, value=500.0)
        manual_data["Fwd IAT Mean"] = st.number_input("Forward IAT Mean", min_value=0.0, value=400.0)
        manual_data["Bwd IAT Mean"] = st.number_input("Backward IAT Mean", min_value=0.0, value=400.0)
        manual_data["Min Packet Length"] = st.number_input("Minimum Packet Length", min_value=0.0, value=20.0)
        manual_data["Max Packet Length"] = st.number_input("Maximum Packet Length", min_value=0.0, value=500.0)
        manual_data["Packet Length Std"] = st.number_input("Packet Length Std", min_value=0.0, value=50.0)
        manual_data["Active Mean"] = st.number_input("Active Mean", min_value=0.0, value=200.0)
        manual_data["Idle Mean"] = st.number_input("Idle Mean", min_value=0.0, value=1000.0)
        manual_data["act_data_pkt_fwd"] = st.number_input("Active Data Packets Forward", min_value=0.0, value=5.0)

    st.divider()

    if st.button("🚀 Predict"):
        input_df = pd.DataFrame([manual_data])
        input_df = prepare_input_df(input_df)

        prediction = model.predict(input_df)
        prediction_label = label_encoder.inverse_transform(prediction)[0]

        st.success(f"Prediction: {prediction_label}")

        if prediction_label == "Normal Traffic":
            st.balloons()
        else:
            st.error("🚨 Suspicious Network Traffic Detected!")

elif menu == "ℹ About":
    st.title("ℹ About")
    st.markdown(
        """
This app is a Streamlit-based Network Intrusion Detection System that uses a trained XGBoost classifier to predict whether a network flow looks normal or suspicious.

Built for the CICIDS2017 dataset and designed for simple deployment and testing.
"""
    )