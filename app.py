from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "network_intrusion_detector.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"

st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

KNOWN_ATTACK_CLASSES = [
    "Normal Traffic", "DoS", "DDoS", "Port Scanning",
    "Bots", "Brute Force", "Web Attacks",
]

NORMAL_LABELS = {
    "normal traffic", "normal", "benign", "benign traffic",
    "benign_traffic", "benign-traffic",
}

def load_css() -> None:
    st.markdown(
        """

        <style>
        /* Hide Streamlit top header */
[data-testid="stHeader"] {
    display: none;
}

/* Remove the extra top space left by the header */
.block-container {
    padding-top: 2rem;
}

/* Optional: hide the bottom deploy/manage bar */
[data-testid="stStatusWidget"] {
    display: none;
}
        :root {
            --primary-color: #2dd4bf;
            color-scheme: dark;
        }

        .stApp {
            background: linear-gradient(135deg, #07111f 0%, #0f172a 100%);
            color: #f1f5f9;
        }
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1200px; }

        .stApp, .stApp p, .stApp span, .stApp label,
        .stApp li, .stApp div, .stApp small { color: #f1f5f9 !important; }

        h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }

        .section-header { font-size: 1.15rem; font-weight: 700; color: #f8fafc !important; margin-bottom: 0.3rem; }
        .subtle { color: #cbd5e1 !important; font-size: 0.97rem; margin-bottom: 0.6rem; line-height: 1.4; }

        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
            color: #b9c4d6 !important;
        }

        /* REMOVED: div:empty { display: none !important; } 
           Broad empty div selectors hide Streamlit's asynchronously rendered 
           components like charts, dataframes, and metrics. */

        .metric-card {
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 1rem;
            height: 100%;
        }
        .metric-title { font-size: 0.75rem; color: #cbd5e1 !important; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; font-weight: 600; }
        .metric-value { font-size: 1.5rem; font-weight: 800; color: #f8fafc !important; word-break: break-word; line-height: 1.2; }
        .metric-caption { font-size: 0.85rem; color: #cbd5e1 !important; margin-top: 0.5rem; line-height: 1.3; }

        .badge { display: inline-block; padding: 0.24rem 0.6rem; border-radius: 999px; font-size: 0.82rem; font-weight: 700; color: #ffffff !important; }
        .badge-low { background: #0f766e; }
        .badge-medium { background: #d97706; }
        .badge-high { background: #b45309; }
        .badge-critical { background: #dc2626; }

        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button,
        [data-testid="stFormSubmitButton"] button {
            border-radius: 10px;
            border: 1px solid rgba(45, 212, 191, 0.35);
            background: linear-gradient(135deg, #0f766e, #2563eb);
            font-weight: 600;
        }
        [data-testid="stButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            border-color: rgba(45, 212, 191, 0.6);
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.28);
        }
        [data-testid="stButton"] button *,
        [data-testid="stDownloadButton"] button *,
        [data-testid="stFormSubmitButton"] button *,
        [data-testid="stButton"] button p,
        [data-testid="stDownloadButton"] button p,
        [data-testid="stFormSubmitButton"] button p {
            color: #ffffff !important;
        }
        [data-testid="stButton"] button:focus,
        [data-testid="stDownloadButton"] button:focus,
        [data-testid="stFormSubmitButton"] button:focus,
        [data-testid="stButton"] button:focus-visible,
        [data-testid="stDownloadButton"] button:focus-visible,
        [data-testid="stFormSubmitButton"] button:focus-visible {
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.45) !important;
            border-color: rgba(45, 212, 191, 0.7) !important;
        }

        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] * { color: #dbe4f2 !important; }
        [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: #b7c2d4 !important; }
        [data-testid="stThumbValue"] { color: #f8fafc !important; font-weight: 700; }

        [data-testid="stSlider"] [role="slider"],
        [data-testid="stSlider"] [data-testid="stThumbValue"] ~ div,
        .stSlider [role="slider"] {
            background-color: #2dd4bf !important;
            border-color: #2dd4bf !important;
        }
        [data-testid="stSlider"] div[data-baseweb="slider"] div[style*="background-color"] {
            background-color: #2dd4bf !important;
        }
        [data-testid="stProgress"] div[role="progressbar"] > div {
            background-color: #2dd4bf !important;
        }

        [data-testid="stMetric"] * { color: #f8fafc !important; }
        [data-testid="stMetricLabel"] * { color: #cbd5e1 !important; }
        [data-testid="stMetricValue"] * { color: #f8fafc !important; }
        [data-testid="stMetricDelta"] * { color: #4ade80 !important; }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(15, 23, 42, 0.92);
            border-radius: 14px;
        }

        [data-testid="stAlert"] {
            background: rgba(226, 232, 240, 0.96) !important;
            border-radius: 10px;
        }
        [data-testid="stAlert"] * { color: #0f172a !important; }

        [data-testid="stFileUploaderDropzone"] {
            background: #eef2f7 !important;
            border-radius: 10px;
        }
        [data-testid="stFileUploaderDropzone"] *,
        [data-testid="stFileUploaderDropzone"] button * {
            color: #1e293b !important;
        }
        [data-testid="stFileUploaderDropzone"] svg { fill: #1e293b !important; }
        [data-testid="stFileUploaderDropzone"] button {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
        }

        section[data-testid="stSidebar"] { background: linear-gradient(180deg, #020617 0%, #0f172a 100%); border-right: 1px solid rgba(148, 163, 184, 0.15); }
        section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding: 0.35rem 0.6rem;
            border-radius: 8px;
            margin-bottom: 0.1rem;
            transition: background 0.15s ease;
        }
        section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(45, 212, 191, 0.12);
        }
        .sidebar-brand { font-size: 1.3rem; font-weight: 800; color: #f8fafc !important; margin-bottom: 0.2rem; }
        .sidebar-tagline { font-size: 0.82rem; color: #94a3b8 !important; margin-bottom: 1rem; }
        .sidebar-info-card {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 12px;
            padding: 0.7rem 0.9rem;
            margin-bottom: 0.9rem;
        }

        .slider-card-title { font-size: 0.92rem; font-weight: 700; color: #f1f5f9 !important; margin-bottom: 0.1rem; }
        .slider-card-desc { font-size: 0.8rem; color: #97a4ba !important; margin-bottom: 0.4rem; }

        .result-title { font-size: 1.6rem; font-weight: 800; color: #f8fafc !important; margin-bottom: 0.3rem; }
        .confidence-value { font-size: 1.4rem; font-weight: 800; color: #2dd4bf !important; }
        .recommendation-box {
            background: rgba(148, 163, 184, 0.08);
            border-left: 3px solid #2563eb;
            border-radius: 8px;
            padding: 0.7rem 0.9rem;
            margin-top: 0.6rem;
            font-size: 0.92rem;
            color: #e2e8f0 !important;
        }

        @media (max-width: 900px) {
            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                row-gap: 1rem;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }
            .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
            .metric-value { font-size: 1.2rem; }
            .section-header { font-size: 1.02rem; }
        }

        .footer {
            text-align: center;
            padding: 30px 0 15px 0;
            margin-top: 50px;
            border-top: 1px solid #334155;
            color: #94a3b8;
            font-size: 13px;
        }
        .footer p {
            margin-bottom: 6px;
            color: #e2e8f0;
            font-size: 15px;
            font-weight: 600;
        }
        .footer span {
            color: #94a3b8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_resource
def load_model_artifacts():
    if not MODEL_PATH.exists(): raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not ENCODER_PATH.exists(): raise FileNotFoundError(f"Missing encoder file: {ENCODER_PATH}")
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, encoder

@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    if not MODEL_COMPARISON_PATH.exists(): raise FileNotFoundError(f"Missing performance file: {MODEL_COMPARISON_PATH}")
    return pd.read_csv(MODEL_COMPARISON_PATH)

def render_section_title(title: str, subtitle: str = "") -> None:
    if subtitle:
        st.markdown(f"<div class='section-header'>{title}</div><div class='subtle'>{subtitle}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)

def render_metric_card(title: str, value: str, caption: str) -> None:
    st.markdown(
        f"<div class='metric-card'><div class='metric-title'>{title}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-caption'>{caption}</div></div>",
        unsafe_allow_html=True,
    )

def normalize_prediction_label(label) -> str:
    label_text = str(label).strip()
    if label_text.lower() in NORMAL_LABELS:
        return "Normal Traffic"

    try:
        numeric_value = float(label_text)
        if numeric_value.is_integer():
            idx = int(numeric_value)
            if 0 <= idx < len(KNOWN_ATTACK_CLASSES):
                return KNOWN_ATTACK_CLASSES[idx]
    except ValueError:
        pass

    return label_text

def _decode_single_label(raw_value, encoder, encoder_classes: list) -> str:
    try:
        decoded = encoder.inverse_transform([raw_value])[0]
        return normalize_prediction_label(decoded)
    except Exception:
        pass

    try: idx = int(raw_value)
    except (TypeError, ValueError): return normalize_prediction_label(raw_value)

    if 0 <= idx < len(encoder_classes): return normalize_prediction_label(encoder_classes[idx])
    if 0 <= idx < len(KNOWN_ATTACK_CLASSES): return KNOWN_ATTACK_CLASSES[idx]
    return f"Unknown Class {idx}"

def safe_inverse_transform(predictions, encoder):
    predictions = np.asarray(predictions)
    if predictions.size == 0: return predictions
    encoder_classes = list(getattr(encoder, "classes_", []))
    return np.array([_decode_single_label(p, encoder, encoder_classes) for p in predictions])

def get_severity(prediction: str) -> str:
    label_norm = str(prediction).strip().lower()
    if label_norm in NORMAL_LABELS: return "Low"
    if label_norm in {"bots", "bot"}: return "Medium"
    if label_norm in {"port scanning", "portscan", "port scan"}: return "Medium"
    if label_norm in {"brute force", "bruteforce", "bruteforce attack"}: return "High"
    if label_norm in {"web attacks", "web attack", "web"}: return "High"
    if label_norm in {"dos", "ddos", "denial of service"}: return "Critical"
    return "Medium"

def render_severity_badge(prediction: str) -> str:
    severity = get_severity(prediction)
    color_class = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high", "Critical": "badge-critical"}.get(severity, "badge-medium")
    return f"<span class='badge {color_class}'>{severity}</span>"

RECOMMENDATIONS = {
    "Low": "This traffic pattern looks consistent with normal, benign activity. No action is required - continue routine monitoring.",
    "Medium": "This traffic shows characteristics of reconnaissance or automated activity (e.g. scanning or bot behavior). Review the source IP and consider rate-limiting or watch-listing it.",
    "High": "This traffic matches patterns associated with active exploitation attempts (e.g. brute force or web application attacks). Investigate the source promptly and consider blocking it at the firewall.",
    "Critical": "This traffic matches a denial-of-service pattern. Escalate immediately to the on-call security team and consider engaging upstream rate-limiting or scrubbing.",
}

def get_recommendation(severity: str) -> str:
    return RECOMMENDATIONS.get(severity, "Review this flow manually to confirm whether it requires action.")

def render_slider_card(title: str, description: str, slider_fn):
    with st.container(border=True):
        st.markdown(f"<div class='slider-card-title'>{title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='slider-card-desc'>{description}</div>", unsafe_allow_html=True)
        return slider_fn()

def build_manual_feature_frame(values: dict, feature_names: list) -> pd.DataFrame:
    base_data = {feature: 0.0 for feature in feature_names}

    destination_port = float(values["destination_port"])
    protocol = values["protocol"]
    connection_duration = max(0.001, float(values["connection_duration"]))
    total_packets = max(1.0, float(values["total_packets"]))
    average_packet_size = max(1.0, float(values["average_packet_size"]))
    flow_bytes_per_second = float(values["flow_bytes_per_second"])
    flow_packets_per_second = float(values["flow_packets_per_second"])
    max_packet_size = max(average_packet_size, float(values["max_packet_size"]))

    total_fwd_packets = max(1, int(round(total_packets * 0.6)))
    total_bwd_packets = max(1, int(round(total_packets * 0.4)))
    total_length_fwd_packets = max(1.0, total_fwd_packets * average_packet_size)
    total_length_bwd_packets = max(1.0, total_bwd_packets * (average_packet_size * 0.7))

    fwd_packet_length_min = max(1.0, average_packet_size * 0.5)
    fwd_packet_length_mean = average_packet_size
    fwd_packet_length_std = max(1.0, (max_packet_size - fwd_packet_length_min) / 6.0)

    bwd_packet_length_mean = max(1.0, average_packet_size * 0.7)
    bwd_packet_length_max = max(1.0, max_packet_size * 0.75)
    bwd_packet_length_min = max(1.0, average_packet_size * 0.4)
    bwd_packet_length_std = max(1.0, (bwd_packet_length_max - bwd_packet_length_min) / 6.0)

    flow_duration = connection_duration
    flow_iat_mean = max(1.0, flow_duration / total_packets)
    flow_iat_std = max(1.0, flow_iat_mean * 0.3)
    flow_iat_max = flow_iat_mean * 1.5
    flow_iat_min = max(1.0, flow_iat_mean * 0.5)

    fwd_iat_total = flow_iat_mean * total_fwd_packets * 0.8
    fwd_iat_mean = max(1.0, flow_iat_mean * 0.9)
    fwd_iat_std = max(1.0, flow_iat_mean * 0.25)
    fwd_iat_max = fwd_iat_mean * 1.4
    fwd_iat_min = max(1.0, fwd_iat_mean * 0.6)

    bwd_iat_total = flow_iat_mean * total_bwd_packets * 0.8
    bwd_iat_mean = max(1.0, flow_iat_mean * 1.05)
    bwd_iat_std = max(1.0, flow_iat_mean * 0.25)
    bwd_iat_max = bwd_iat_mean * 1.4
    bwd_iat_min = max(1.0, bwd_iat_mean * 0.6)

    is_tcp = protocol == "TCP"
    ack_flag_count = total_packets if is_tcp else 0.0
    psh_flag_count = max(1.0, total_fwd_packets * 0.5) if is_tcp else 0.0
    fin_flag_count = 1.0 if is_tcp else 0.0
    syn_flag_count = 2.0 if is_tcp else 0.0

    base_data.update({
        "Destination Port": destination_port, "Flow Duration": flow_duration,
        "Total Fwd Packets": float(total_fwd_packets), "Total Backward Packets": float(total_bwd_packets),
        "Total Bwd Packets": float(total_bwd_packets), "Total Length of Fwd Packets": total_length_fwd_packets,
        "Total Length of Bwd Packets": total_length_bwd_packets, "Total Length of Backward Packets": total_length_bwd_packets,
        "Fwd Packet Length Max": max_packet_size, "Fwd Packet Length Min": fwd_packet_length_min,
        "Fwd Packet Length Mean": fwd_packet_length_mean, "Fwd Packet Length Std": fwd_packet_length_std,
        "Bwd Packet Length Max": bwd_packet_length_max, "Bwd Packet Length Min": bwd_packet_length_min,
        "Bwd Packet Length Mean": bwd_packet_length_mean, "Bwd Packet Length Std": bwd_packet_length_std,
        "Flow Bytes/s": flow_bytes_per_second, "Flow Packets/s": flow_packets_per_second,
        "Flow IAT Mean": flow_iat_mean, "Flow IAT Std": flow_iat_std, "Flow IAT Max": flow_iat_max, "Flow IAT Min": flow_iat_min,
        "Fwd IAT Total": fwd_iat_total, "Fwd IAT Mean": fwd_iat_mean, "Fwd IAT Std": fwd_iat_std, "Fwd IAT Max": fwd_iat_max, "Fwd IAT Min": fwd_iat_min,
        "Bwd IAT Total": bwd_iat_total, "Bwd IAT Mean": bwd_iat_mean, "Bwd IAT Std": bwd_iat_std, "Bwd IAT Max": bwd_iat_max, "Bwd IAT Min": bwd_iat_min,
        "Fwd Header Length": float(total_fwd_packets * 20), "Bwd Header Length": float(total_bwd_packets * 20),
        "Fwd Packets/s": flow_packets_per_second * 0.6, "Bwd Packets/s": flow_packets_per_second * 0.4,
        "Min Packet Length": max(1.0, average_packet_size * 0.4), "Max Packet Length": max_packet_size,
        "Packet Length Mean": average_packet_size, "Packet Length Std": max(1.0, (max_packet_size - average_packet_size) / 6.0),
        "Packet Length Variance": max(1.0, ((max_packet_size - average_packet_size) / 6.0) ** 2),
        "FIN Flag Count": fin_flag_count, "SYN Flag Count": syn_flag_count, "RST Flag Count": 0.0,
        "PSH Flag Count": psh_flag_count, "ACK Flag Count": ack_flag_count, "URG Flag Count": 0.0,
        "CWE Flag Count": 0.0, "ECE Flag Count": 0.0, "Average Packet Size": average_packet_size,
        "Subflow Fwd Packets": float(total_fwd_packets), "Subflow Fwd Bytes": total_length_fwd_packets,
        "Subflow Bwd Packets": float(total_bwd_packets), "Subflow Bwd Bytes": total_length_bwd_packets,
        "Init_Win_bytes_forward": 29200.0 if is_tcp else 8192.0, "Init_Win_bytes_backward": 29200.0 if is_tcp else 8192.0,
        "act_data_pkt_fwd": max(1, int(round(total_fwd_packets * 0.5))), "min_seg_size_forward": 32.0,
        "Active Mean": connection_duration * 0.6, "Active Max": connection_duration * 0.8, "Active Min": connection_duration * 0.3, "Active Std": connection_duration * 0.15,
        "Idle Mean": connection_duration * 0.1, "Idle Max": connection_duration * 0.2, "Idle Min": connection_duration * 0.02, "Idle Std": connection_duration * 0.05,
    })

    return pd.DataFrame([base_data])[feature_names]

def prepare_uploaded_dataframe(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    if df.empty: raise ValueError("The uploaded file is empty.")
    missing_columns = [column for column in feature_names if column not in df.columns]
    if missing_columns:
        raise ValueError("The uploaded file is missing required traffic features: " + ", ".join(missing_columns))
    return df[feature_names].copy()

def run_prediction(df: pd.DataFrame, model, encoder) -> pd.DataFrame:
    feature_names = [str(column) for column in model.feature_names_in_]
    input_df = prepare_uploaded_dataframe(df, feature_names)
    predictions = model.predict(input_df)
    labels = safe_inverse_transform(predictions, encoder)

    result_df = df.copy()
    result_df["Prediction"] = [normalize_prediction_label(label) for label in labels]
    result_df["Severity"] = result_df["Prediction"].apply(get_severity)
    result_df["Record Number"] = range(1, len(result_df) + 1)
    ordered_cols = ["Record Number", "Prediction", "Severity"] + [
        col for col in result_df.columns if col not in {"Record Number", "Prediction", "Severity"}
    ]
    return result_df[ordered_cols]

def build_summary_stats(result_df: pd.DataFrame) -> dict:
    total_records = len(result_df)
    normal_count = int(result_df["Prediction"].apply(lambda x: str(x).strip().lower() in NORMAL_LABELS).sum())
    attack_count = total_records - normal_count
    attack_rate = round((attack_count / total_records) * 100, 2) if total_records else 0.0

    if attack_rate < 5: health, recommendation = "Healthy", "No malicious traffic detected. Monitoring remains stable."
    elif attack_rate < 20: health, recommendation = "Low Risk", "A small number of suspicious flows were identified. Continue monitoring."
    elif attack_rate < 40: health, recommendation = "Medium Risk", "Suspicious traffic is elevated. Review the affected flows promptly."
    else: health, recommendation = "High Risk", "A high number of malicious flows detected. Immediate investigation is recommended."

    return {
        "total_records": total_records, "attack_count": attack_count, "normal_count": normal_count,
        "attack_rate": attack_rate, "health": health, "recommendation": recommendation,
    }

def display_results(result_df: pd.DataFrame) -> None:
    stats = build_summary_stats(result_df)

    st.subheader("Prediction Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", stats["total_records"])
    col2.metric("Normal Traffic", stats["normal_count"])
    col3.metric("Attack Traffic", stats["attack_count"])
    col4.metric("Attack Rate", f"{stats['attack_rate']}%")

    st.download_button(
        "Download Prediction Results (CSV)", result_df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_results.csv", mime="text/csv", use_container_width=True,
    )

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Prediction Distribution")
        counts = result_df["Prediction"].astype(str).apply(normalize_prediction_label).value_counts().head(10).reset_index()
        counts.columns = ["Prediction", "Count"]
        counts["Count"] = counts["Count"].astype(int)
        st.bar_chart(counts.set_index("Prediction"))

    with chart_col2:
        st.subheader("Severity Distribution")
        severity_counts = result_df["Severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]
        severity_counts["Count"] = severity_counts["Count"].astype(int)
        st.bar_chart(severity_counts.set_index("Severity"))

    attack_breakdown = result_df[result_df["Prediction"] != "Normal Traffic"]["Prediction"].value_counts().head(10).reset_index()
    attack_breakdown.columns = ["Attack Type", "Count"]
    st.subheader("Top Attack Types")
    if not attack_breakdown.empty:
        attack_breakdown["Count"] = attack_breakdown["Count"].astype(int)
        st.bar_chart(attack_breakdown.set_index("Attack Type"))
    else:
        st.info("No attack traffic detected.")

    st.markdown("---")

    st.subheader("Network Health Summary")
    with st.container(border=True):
        st.markdown(f"**Status:** {stats['health']}")
        st.markdown(f"**Recommendation:** {stats['recommendation']}")

def show_dashboard(model, encoder) -> None:
    render_section_title("Security Operations Overview", "Operational intelligence for the deployed intrusion detection workflow.")

    comparison_df = load_model_comparison()
    best_model_row = comparison_df.loc[comparison_df["Accuracy"].idxmax()]
    accuracy = float(best_model_row["Accuracy"])

    col1, col2, col3, col4 = st.columns(4)
    with col1: render_metric_card("Best Model", "XGBoost", "Highest accuracy on the evaluation set")
    with col2: render_metric_card("Accuracy", f"{accuracy:.2%}", "Model accuracy")
    with col3: render_metric_card("Dataset", "CICIDS2017", "Benchmark network intrusion data")
    with col4: render_metric_card("Supported Attack Types", "7", "Normal traffic plus major intrusion classes")

    st.markdown("")
    with st.container(border=True):
        st.subheader("Quick Actions")
        st.markdown("Use the sidebar to move between CSV ingestion, manual testing, and model evaluation views.")

    col1, col2 = st.columns(2)
    with col1:
        render_section_title("Project Overview")
        st.markdown("This platform combines a trained XGBoost classifier with an analyst-oriented interface for reviewing network traffic behavior.")
    with col2:
        render_section_title("Machine Learning Pipeline")
        st.markdown(
            "1. Load the trained model and label encoder  \n"
            "2. Validate incoming traffic data or synthetic samples  \n"
            "3. Generate predictions with severity interpretation  \n"
            "4. Deliver operational reporting and export support"
        )

    col1, col2 = st.columns(2)
    with col1:
        render_section_title("Supported Attack Types")
        st.markdown("- Normal Traffic\n- DoS\n- DDoS\n- Port Scanning\n- Bots\n- Brute Force\n- Web Attacks")
    with col2:
        render_section_title("Application Features")
        st.markdown("- CSV prediction workflow\n- Simplified manual input form\n- Severity mapping and health scoring\n- Analyst-ready reporting and download support")

def show_csv_prediction(model, encoder) -> None:
    render_section_title("Predict from CSV", "Upload a network flow dataset and review the resulting classification report.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="Expected columns should align with the trained XGBoost model")
    if uploaded_file is None:
        st.info("Select a CSV file to begin the analysis workflow.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read the uploaded file: {exc}")
        return

    st.caption(f"{len(df)} records loaded and ready for prediction.")

    if st.button("Run Prediction", use_container_width=True):
        try:
            result_df = run_prediction(df, model, encoder)
            display_results(result_df)
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

def show_manual_prediction(model, encoder) -> None:
    render_section_title("Manual Prediction", "Provide a compact set of traffic indicators and the application synthesizes the remaining model inputs.")

    with st.form("manual_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            destination_port = render_slider_card("🎯 Destination Port", "The TCP/UDP port the traffic is heading to (e.g. 80 = HTTP, 443 = HTTPS).", lambda: st.slider("Destination Port", 0, 65535, 80, step=1, label_visibility="collapsed"))
            protocol = render_slider_card("🔌 Protocol", "The transport-layer protocol used for this connection.", lambda: st.selectbox("Protocol", ["TCP", "UDP", "ICMP"], label_visibility="collapsed"))
            connection_duration = render_slider_card("⏱️ Connection Duration", "How long the flow stayed open, in seconds.", lambda: st.slider("Connection Duration", 0.0, 60.0, 2.5, step=0.1, label_visibility="collapsed"))
            total_packets = render_slider_card("📦 Total Packets", "Total number of packets exchanged in the flow.", lambda: st.slider("Total Packets", 1.0, 1000.0, 12.0, step=1.0, label_visibility="collapsed"))

        with col2:
            average_packet_size = render_slider_card("📏 Average Packet Size", "Mean packet size across the flow, in bytes.", lambda: st.slider("Average Packet Size", 1.0, 1500.0, 150.0, step=1.0, label_visibility="collapsed"))
            max_packet_size = render_slider_card("📐 Maximum Packet Size", "The largest single packet observed in the flow, in bytes.", lambda: st.slider("Maximum Packet Size", 1.0, 1500.0, 500.0, step=1.0, label_visibility="collapsed"))
            flow_bytes_per_second = render_slider_card("🌊 Flow Bytes / Second", "Throughput of the flow, in bytes per second.", lambda: st.slider("Flow Bytes Per Second", 0.0, 100000.0, 800.0, step=10.0, label_visibility="collapsed"))
            flow_packets_per_second = render_slider_card("⚡ Flow Packets / Second", "Packet rate of the flow, in packets per second.", lambda: st.slider("Flow Packets Per Second", 0.0, 1000.0, 6.0, step=0.5, label_visibility="collapsed"))

        submitted = st.form_submit_button("Generate Prediction", use_container_width=True)

    if not submitted: return

    try:
        feature_names = [str(column) for column in model.feature_names_in_]
        input_df = build_manual_feature_frame({
            "destination_port": destination_port, "protocol": protocol, "connection_duration": connection_duration,
            "total_packets": total_packets, "average_packet_size": average_packet_size,
            "flow_bytes_per_second": flow_bytes_per_second, "flow_packets_per_second": flow_packets_per_second,
            "max_packet_size": max_packet_size,
        }, feature_names)

        prediction = model.predict(input_df)[0]
        label = safe_inverse_transform([prediction], encoder)[0]
        severity = get_severity(label)

        confidence = None
        proba_df = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            encoder_classes = list(getattr(encoder, "classes_", []))
            if len(encoder_classes) == len(proba):
                class_labels = [normalize_prediction_label(c) for c in encoder_classes]
            elif len(KNOWN_ATTACK_CLASSES) == len(proba):
                class_labels = KNOWN_ATTACK_CLASSES
            else:
                class_labels = [f"Class {i}" for i in range(len(proba))]

            proba_df = pd.DataFrame({"Class": class_labels, "Confidence": proba}).sort_values("Confidence", ascending=False)
            confidence = float(proba_df.iloc[0]["Confidence"])

        st.subheader("Prediction Outcome")
        with st.container(border=True):
            if confidence is not None: title_col, conf_col = st.columns([2, 1])
            else: title_col, conf_col = st.container(), None

            with title_col:
                st.markdown(f"<div class='result-title'>{label}</div>", unsafe_allow_html=True)
                st.markdown(f"Severity: {render_severity_badge(label)}", unsafe_allow_html=True)

            if conf_col is not None:
                with conf_col:
                    st.markdown("<div class='slider-card-desc'>Model Confidence</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='confidence-value'>{confidence:.1%}</div>", unsafe_allow_html=True)
                    st.progress(min(max(confidence, 0.0), 1.0))

            st.markdown(f"<div class='recommendation-box'><strong>Recommended action:</strong> {get_recommendation(severity)}</div>", unsafe_allow_html=True)

            if proba_df is not None:
                with st.expander("View confidence by class"):
                    st.bar_chart(proba_df.set_index("Class"))
                    st.caption("This form generates synthetic features based on your input. The model is highly sensitive to bidirectional flow characteristics (backward packets/bytes). Adjusting total packets and connection duration will significantly impact whether the model classifies the traffic as Normal vs Attack.")
    except Exception as exc:
        st.error(f"Manual prediction failed: {exc}")

def show_model_performance() -> None:
    render_section_title("Model Performance", "Compare the evaluated classifiers and review their operational metrics.")

    try:
        comparison_df = load_model_comparison()
        required_cols = {"Model", "Accuracy", "Precision", "Recall", "F1 Score"}
        missing = required_cols - set(comparison_df.columns)
        if missing:
            st.error("model_comparison.csv is missing required column(s): " + ", ".join(sorted(missing)) + f". Columns found: {', '.join(comparison_df.columns)}.")
            return

        best_idx = comparison_df["Accuracy"].idxmax()
        best_model = comparison_df.loc[best_idx, "Model"]
        best_accuracy = comparison_df.loc[best_idx, "Accuracy"]

        st.info(f"{best_model} is currently the best-performing model with an accuracy of {best_accuracy:.2%}.")

        metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
        display_df = comparison_df[["Model", *metric_cols]].copy()

        # Ensure metrics are strictly numeric for formatting and charting
        for col in metric_cols:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

        display_df_fmt = display_df.copy()
        for col in metric_cols:
            display_df_fmt[col] = display_df_fmt[col].apply(lambda value: f"{value:.5f}" if pd.notnull(value) else "N/A")

        st.dataframe(display_df_fmt, use_container_width=True, hide_index=True)

        chart_data = display_df.set_index("Model")[metric_cols]
        st.subheader("Metric Comparison")
        st.bar_chart(chart_data)
    except FileNotFoundError as exc:
        st.error(f"Could not load model performance data: {exc}")
    except Exception as exc:
        st.error(f"Model Performance page failed to load: {exc}")

def show_about() -> None:
    render_section_title("About the Platform", "A professional, analyst-focused deployment for network intrusion detection.")

    with st.container(border=True):
        st.subheader("Application")
        st.markdown("This application delivers a secure interface for reviewing network traffic classification results using the trained XGBoost model.")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Dataset")
            st.markdown("**Dataset:** CICIDS2017")
            st.markdown("**Focus:** Network intrusion and benign traffic classification")
    with col2:
        with st.container(border=True):
            st.subheader("Machine Learning Model")
            st.markdown("**Model:** XGBoost")
            st.markdown("**Purpose:** Multi-class network traffic classification")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("Framework")
            st.markdown("**Platform:** Streamlit")
            st.markdown("**Programming Language:** Python")
    with col2:
        with st.container(border=True):
            st.subheader("Application Features")
            st.markdown("Pandas, Joblib, Scikit-learn, XGBoost")

    with st.container(border=True):
        st.subheader("Project Workflow")
        st.markdown("The workflow begins with model loading, continues through traffic validation and prediction, and concludes with analyst-ready reporting and export.")

def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            <p>Network Intrusion Detection System</p>
            <span>Built with Python • XGBoost • Streamlit</span>
            <br>
            <span>© 2026 Utsav Gangadiya</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def main() -> None:
    load_css()

    try:
        model, encoder = load_model_artifacts()
    except Exception as exc:
        st.error(f"Unable to load the trained model artifacts. Error: {exc}")
        st.stop()

    st.sidebar.markdown(
        "<div class='sidebar-brand'>🛡️ Network Security Center</div>"
        "<div class='sidebar-tagline'>Intrusion detection & traffic analysis</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div class='sidebar-info-card'>"
        "<div class='metric-title'>Model</div><div class='metric-value'>XGBoost</div>"
        "</div>"
        "<div class='sidebar-info-card'>"
        "<div class='metric-title'>Dataset</div><div class='metric-value'>CICIDS2017</div>"
        "</div>"
        "<div class='sidebar-info-card'>"
        "<div class='metric-title'>Framework</div><div class='metric-value'>Streamlit</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    menu = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Predict From CSV", "Manual Prediction", "Model Performance", "About"],
        index=0,
        label_visibility="collapsed",
    )

    if menu == "Dashboard": show_dashboard(model, encoder)
    elif menu == "Predict From CSV": show_csv_prediction(model, encoder)
    elif menu == "Manual Prediction": show_manual_prediction(model, encoder)
    elif menu == "Model Performance": show_model_performance()
    else: show_about()

    render_footer()

if __name__ == "__main__":
    main()