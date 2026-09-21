from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import openpyxl

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "network_intrusion_detector.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"

st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="",
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

# Sidebar navigation: (key, icon, label)
NAV_ITEMS = [
    ("dashboard", "", "Dashboard"),
    ("scanner", "", "Threat Scanner"),
    ("manual", "", "Manual Prediction"),
    ("performance", "", "Model Performance"),
    ("about", "", "About"),
]

# Quick-start scenarios for the manual prediction form
TRAFFIC_PRESETS = {
    "Custom": None,
    " Typical Web Browsing": dict(
        destination_port=443, protocol="TCP", connection_duration=3.2,
        total_packets=18, average_packet_size=420, max_packet_size=1200,
        flow_bytes_per_second=2500, flow_packets_per_second=8,
    ),
    " Port Scan Pattern": dict(
        destination_port=22, protocol="TCP", connection_duration=0.05,
        total_packets=2, average_packet_size=44, max_packet_size=64,
        flow_bytes_per_second=900, flow_packets_per_second=40,
    ),
    " DoS-style Burst": dict(
        destination_port=80, protocol="UDP", connection_duration=0.8,
        total_packets=950, average_packet_size=1400, max_packet_size=1500,
        flow_bytes_per_second=95000, flow_packets_per_second=850,
    ),
}


def load_css() -> None:
    st.markdown(
        """
        <style>

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

        /* =========================================================
           DESIGN TOKENS
        ========================================================= */

        :root {
            --c-bg: #f7f9fc;
            --c-surface: #ffffff;
            --c-border: #e2e8f0;
            --c-text: #1e293b;
            --c-text-strong: #0f172a;
            --c-text-muted: #64748b;

            --c-cyan: #0891b2;      --c-cyan-light: #ecfeff;    --c-cyan-border: #a5f3fc;
            --c-violet: #7c3aed;    --c-violet-light: #f5f3ff;  --c-violet-border: #ddd6fe;
            --c-amber: #d97706;     --c-amber-light: #fffbeb;   --c-amber-border: #fde68a;
            --c-teal: #0d9488;      --c-teal-light: #f0fdfa;    --c-teal-border: #99f6e4;
            --c-rose: #e11d48;      --c-rose-light: #fff1f2;    --c-rose-border: #fecdd3;

            --sev-low: #16a34a;
            --sev-medium: #d97706;
            --sev-high: #ea580c;
            --sev-critical: #dc2626;

            --sidebar-bg: #0b1220;
            --sidebar-bg-raised: #131c2e;
            --sidebar-border: #1f2a3d;
            --sidebar-text: #cbd5e1;
            --sidebar-text-muted: #7c8aa5;

            --radius-sm: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --ease: cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* =========================================================
           GLOBAL
        ========================================================= */

        [data-testid="stHeader"] { display: none; }
        [data-testid="stStatusWidget"] { display: none; }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.5rem;
            max-width: 1220px;
        }

        body {
            background-color: var(--c-bg);
            color: var(--c-text);
            font-family: "Inter", "Segoe UI", system-ui, -apple-system, Arial, sans-serif;
        }

        .stApp { background: var(--c-bg); color: var(--c-text); }

        h1, h2, h3, h4, h5, h6 {
            color: var(--c-text-strong) !important;
            font-weight: 700;
            letter-spacing: -0.01em;
        }

        p, label { color: var(--c-text); }

        code, .mono {
            font-family: "JetBrains Mono", ui-monospace, monospace !important;
        }


        /* =========================================================
           SECTION HEADERS
        ========================================================= */

        .section-header {
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--c-text-strong);
            margin-top: 0.6rem;
            margin-bottom: 0.3rem;
            padding-bottom: 0.4rem;
            border-bottom: 2px solid var(--c-border);
            position: relative;
        }

        .section-header::after {
            content: "";
            position: absolute;
            bottom: -2px; left: 0;
            width: 46px; height: 2px;
            background: linear-gradient(90deg, var(--c-cyan), var(--c-violet));
            border-radius: 999px;
        }

        .subtle {
            color: var(--c-text-muted);
            font-size: 0.95rem;
            margin-bottom: 0.7rem;
            line-height: 1.5;
        }


        /* =========================================================
           METRIC CARDS (with colored accents)
        ========================================================= */

        .metric-card {
            background: var(--c-surface);
            border: 1px solid var(--c-border);
            border-radius: var(--radius-lg);
            border-top: 3px solid var(--c-border);
            padding: 1.05rem 1.2rem;
            height: 100%;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            transition: transform 0.2s var(--ease), box-shadow 0.2s var(--ease), border-color 0.2s var(--ease);
        }

        .metric-card:hover {
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.10);
            transform: translateY(-2px);
        }

        .metric-card.accent-cyan   { border-top-color: var(--c-cyan); }
        .metric-card.accent-violet { border-top-color: var(--c-violet); }
        .metric-card.accent-amber  { border-top-color: var(--c-amber); }
        .metric-card.accent-teal   { border-top-color: var(--c-teal); }
        .metric-card.accent-rose   { border-top-color: var(--c-rose); }

        .metric-title {
            font-size: 0.72rem;
            color: var(--c-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 0.4rem;
            font-weight: 700;
        }

        .metric-value {
            font-size: 1.55rem;
            font-weight: 800;
            color: var(--c-text-strong);
            word-break: break-word;
            line-height: 1.2;
            font-family: "JetBrains Mono", ui-monospace, monospace;
        }

        .metric-caption {
            font-size: 0.82rem;
            color: var(--c-text-muted);
            margin-top: 0.45rem;
            line-height: 1.35;
        }


        /* =========================================================
           SEVERITY BADGES
        ========================================================= */

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            color: #ffffff !important;
            letter-spacing: 0.02em;
        }

        .badge::before {
            content: "";
            width: 6px; height: 6px;
            border-radius: 50%;
            background: rgba(255,255,255,0.85);
        }

        .badge-low      { background: var(--sev-low); }
        .badge-medium   { background: var(--sev-medium); }
        .badge-high     { background: var(--sev-high); }
        .badge-critical { background: var(--sev-critical); }

        /* Generic colored chip, used for attack-type tags */
        .tag {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.83rem;
            font-weight: 600;
            margin: 0.2rem 0.3rem 0.2rem 0;
            border: 1px solid transparent;
        }

        .tag-cyan   { background: var(--c-cyan-light);   color: var(--c-cyan);   border-color: var(--c-cyan-border); }
        .tag-violet { background: var(--c-violet-light); color: var(--c-violet); border-color: var(--c-violet-border); }
        .tag-amber  { background: var(--c-amber-light);  color: var(--c-amber);  border-color: var(--c-amber-border); }
        .tag-teal   { background: var(--c-teal-light);   color: var(--c-teal);   border-color: var(--c-teal-border); }
        .tag-rose   { background: var(--c-rose-light);   color: var(--c-rose);   border-color: var(--c-rose-border); }


        /* =========================================================
           BUTTONS
        ========================================================= */

        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button,
        [data-testid="stFormSubmitButton"] button {
            border-radius: var(--radius-sm);
            border: 1px solid #d1d5db;
            background: var(--c-surface);
            color: var(--c-text) !important;
            font-weight: 600;
            padding: 0.55rem 1rem;
            transition: all 0.2s var(--ease);
        }

        [data-testid="stButton"] button:hover,
        [data-testid="stDownloadButton"] button:hover,
        [data-testid="stFormSubmitButton"] button:hover {
            border-color: var(--c-cyan);
            color: var(--c-cyan) !important;
            background: var(--c-cyan-light);
            box-shadow: 0 4px 12px rgba(8, 145, 178, 0.14);
        }

        [data-testid="stButton"] button[kind="primary"],
        [data-testid="stFormSubmitButton"] button[kind="primary"] {
            background: linear-gradient(135deg, var(--c-cyan), #0e7490) !important;
            border-color: var(--c-cyan) !important;
            color: #ffffff !important;
            box-shadow: 0 6px 18px rgba(8, 145, 178, 0.25);
        }

        [data-testid="stButton"] button[kind="primary"]:hover,
        [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
            background: linear-gradient(135deg, #0e7490, #155e75) !important;
            border-color: #0e7490 !important;
            color: #ffffff !important;
            transform: translateY(-1px);
        }


        /* =========================================================
           INPUTS / SELECTBOX / TEXT INPUT
        ========================================================= */

        div[data-baseweb="select"] > div {
            background: var(--c-surface) !important;
            border: 1px solid #d1d5db !important;
            border-radius: var(--radius-sm) !important;
        }

        div[data-baseweb="select"] > div:hover { border-color: var(--c-cyan) !important; }

        input, textarea {
            background-color: var(--c-surface) !important;
            color: var(--c-text) !important;
            border-radius: var(--radius-sm) !important;
        }

        input:focus, textarea:focus {
            border-color: var(--c-cyan) !important;
            box-shadow: 0 0 0 2px var(--c-cyan-light) !important;
        }

        /* Radio pills used inside forms (e.g. Protocol selector) */
        div[role="radiogroup"] label {
            border: 1px solid var(--c-border);
            background: var(--c-surface);
            border-radius: 999px;
            padding: 0.3rem 0.85rem !important;
            margin-right: 0.4rem;
            transition: all 0.2s var(--ease);
        }

        div[role="radiogroup"] label:hover {
            border-color: var(--c-cyan);
            background: var(--c-cyan-light);
        }


        /* =========================================================
           FILE UPLOADER
        ========================================================= */

        [data-testid="stFileUploaderDropzone"] {
            background: var(--c-bg);
            border: 1.5px dashed #cbd5e1;
            border-radius: var(--radius-lg);
            padding: 1rem;
            transition: all 0.2s var(--ease);
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            background: var(--c-cyan-light);
            border-color: var(--c-cyan);
        }

        [data-testid="stFileUploaderDropzone"] * { color: var(--c-text) !important; }
        [data-testid="stFileUploaderDropzone"] svg { fill: var(--c-cyan) !important; }

        [data-testid="stFileUploaderDropzone"] button {
            background: var(--c-surface) !important;
            border: 1px solid #d1d5db !important;
            border-radius: 7px !important;
            color: var(--c-text) !important;
        }

        [data-testid="stFileUploaderDropzone"] button:hover {
            border-color: var(--c-cyan) !important;
            color: var(--c-cyan) !important;
        }


        /* =========================================================
           SIDEBAR (dark "operations center" theme)
        ========================================================= */

        section[data-testid="stSidebar"] {
            background: var(--sidebar-bg);
            border-right: 1px solid var(--sidebar-border);
        }

        section[data-testid="stSidebar"] * { color: var(--sidebar-text); }

        section[data-testid="stSidebar"] hr {
            border-top: 1px solid var(--sidebar-border);
            margin: 1rem 0;
        }

        /* Navigation radio, restyled as a stacked nav list */
        section[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.35rem;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            display: flex;
            align-items: center;
            width: 100%;
            background: transparent;
            border: 1px solid transparent;
            border-radius: var(--radius-md);
            padding: 0.6rem 0.8rem !important;
            margin-right: 0;
            font-weight: 600;
            font-size: 0.95rem;
            transition: all 0.18s var(--ease);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: var(--sidebar-bg-raised);
            border-color: var(--sidebar-border);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, rgba(8,145,178,0.22), rgba(124,58,237,0.16));
            border-color: var(--c-cyan);
            box-shadow: inset 0 0 0 1px rgba(8,145,178,0.25);
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
            color: #ffffff !important;
        }

        /* Brand block at the bottom of the sidebar */
        .sidebar-brand {
            font-size: 1.2rem;
            font-weight: 800;
            color: #ffffff !important;
            margin-bottom: 0.1rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }

        .sidebar-tagline {
            font-size: 0.78rem;
            color: var(--sidebar-text-muted) !important;
            margin-bottom: 0.85rem;
            line-height: 1.4;
        }

        /* Compact colored stat chips (Model / Dataset / Framework) */
        .sb-chip-row {
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }

        .sb-chip {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--sidebar-bg-raised);
            border: 1px solid var(--sidebar-border);
            border-left: 3px solid var(--c-cyan);
            border-radius: 8px;
            padding: 0.4rem 0.65rem;
            font-size: 0.78rem;
        }

        .sb-chip.chip-cyan   { border-left-color: var(--c-cyan); }
        .sb-chip.chip-violet { border-left-color: var(--c-violet); }
        .sb-chip.chip-amber  { border-left-color: var(--c-amber); }

        .sb-chip-label {
            color: var(--sidebar-text-muted);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.66rem;
            font-weight: 700;
        }

        .sb-chip-value {
            color: #ffffff;
            font-weight: 700;
            font-family: "JetBrains Mono", ui-monospace, monospace;
            font-size: 0.78rem;
        }


        /* =========================================================
           ALERT / INFO CARDS
        ========================================================= */

        .info-card {
            background: var(--c-cyan-light); border: 1px solid var(--c-cyan-border);
            border-left: 4px solid var(--c-cyan); border-radius: 8px;
            padding: 0.85rem 1rem; margin: 0.7rem 0; color: #155e75;
        }

        .success-card {
            background: #f0fdf4; border: 1px solid #bbf7d0;
            border-left: 4px solid var(--sev-low); border-radius: 8px;
            padding: 0.85rem 1rem; margin: 0.7rem 0; color: #166534;
        }

        .warning-card {
            background: var(--c-amber-light); border: 1px solid var(--c-amber-border);
            border-left: 4px solid var(--c-amber); border-radius: 8px;
            padding: 0.85rem 1rem; margin: 0.7rem 0; color: #92400e;
        }

        .danger-card {
            background: var(--c-rose-light); border: 1px solid var(--c-rose-border);
            border-left: 4px solid var(--c-rose); border-radius: 8px;
            padding: 0.85rem 1rem; margin: 0.7rem 0; color: #9f1239;
        }


        /* =========================================================
           DATAFRAME / TABLE
        ========================================================= */

        [data-testid="stDataFrame"] {
            border: 1px solid var(--c-border);
            border-radius: var(--radius-md);
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
        }


        /* =========================================================
           TABS
        ========================================================= */

        button[data-baseweb="tab"] { color: var(--c-text-muted) !important; font-weight: 600; }
        button[data-baseweb="tab"][aria-selected="true"] { color: var(--c-cyan) !important; }
        [data-baseweb="tab-highlight"] { background-color: var(--c-cyan) !important; }


        /* =========================================================
           EXPANDERS
        ========================================================= */

        [data-testid="stExpander"] {
            background: var(--c-surface);
            border: 1px solid var(--c-border);
            border-radius: var(--radius-md);
            margin-bottom: 0.7rem;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
        }


        /* =========================================================
           SLIDER / INPUT CARDS (manual prediction)
        ========================================================= */

        .slider-card-title {
            font-weight: 700;
            color: var(--c-text-strong);
            font-size: 0.95rem;
            margin-bottom: 0.15rem;
        }

        .slider-card-desc {
            color: var(--c-text-muted);
            font-size: 0.8rem;
            margin-bottom: 0.55rem;
            line-height: 1.4;
        }

        .result-title {
            font-size: 1.7rem;
            font-weight: 800;
            color: var(--c-text-strong);
            margin-bottom: 0.35rem;
        }

        .confidence-value {
            font-size: 1.6rem;
            font-weight: 800;
            font-family: "JetBrains Mono", ui-monospace, monospace;
            color: var(--c-cyan);
            margin: 0.2rem 0 0.5rem 0;
        }

        .recommendation-box {
            background: var(--c-bg);
            border: 1px solid var(--c-border);
            border-radius: var(--radius-sm);
            padding: 0.8rem 1rem;
            margin-top: 0.9rem;
            font-size: 0.92rem;
            line-height: 1.5;
        }


        /* =========================================================
           DIVIDERS / FOOTER
        ========================================================= */

        hr { border: none; border-top: 1px solid var(--c-border); margin: 1.3rem 0; }

        .footer {
            text-align: center;
            padding: 30px 0 15px 0;
            margin-top: 50px;
            border-top: 1px solid var(--c-border);
            color: #9ca3af;
            font-size: 13px;
        }

        .footer p { margin-bottom: 6px; color: #4b5563; font-size: 14px; font-weight: 600; }
        .footer span { color: #9ca3af; }

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
    validate_artifacts(model, encoder)
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


def render_metric_card(title: str, value: str, caption: str, accent: str = "cyan") -> None:
    st.markdown(
        f"<div class='metric-card accent-{accent}'><div class='metric-title'>{title}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-caption'>{caption}</div></div>",
        unsafe_allow_html=True,
    )


def render_tag(text: str, color: str = "cyan") -> str:
    return f"<span class='tag tag-{color}'>{text}</span>"


def normalize_prediction_label(label) -> str:
    label_text = str(label).strip()
    if label_text.lower() in NORMAL_LABELS:
        return "Normal Traffic"
    return label_text


def validate_artifacts(model, encoder) -> None:
    model_classes = np.asarray(getattr(model, "classes_", []))
    encoder_classes = np.asarray(getattr(encoder, "classes_", []))
    model_count = int(getattr(model, "n_classes_", len(model_classes)))
    encoder_count = len(encoder_classes)
    errors = []

    if model_count != encoder_count:
        errors.append(f"Class-count mismatch: model={model_count}, encoder={encoder_count}")
    if model_classes.size and not np.array_equal(model_classes, np.arange(model_count)):
        errors.append("Unexpected model class indices; verify the training pipeline.")
    if not hasattr(model, "feature_names_in_"):
        errors.append("Model does not expose feature_names_in_.")
    if errors:
        raise RuntimeError("Invalid model artifacts:\n- " + "\n- ".join(errors))


def safe_inverse_transform(predictions, encoder):
    predictions = np.asarray(predictions)
    if predictions.size == 0:
        return predictions
    try:
        decoded = encoder.inverse_transform(predictions)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Unable to decode model predictions with the saved label encoder.") from exc
    return np.asarray([normalize_prediction_label(label) for label in decoded])


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


def render_slider_card(title: str, description: str, widget_fn):
    with st.container(border=True):
        st.markdown(f"<div class='slider-card-title'>{title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='slider-card-desc'>{description}</div>", unsafe_allow_html=True)
        return widget_fn()


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
    with col1: render_metric_card("Total Records", f"{stats['total_records']:,}", "Flows analyzed", accent="cyan")
    with col2: render_metric_card("Normal Traffic", f"{stats['normal_count']:,}", "Classified as benign", accent="teal")
    with col3: render_metric_card("Attack Traffic", f"{stats['attack_count']:,}", "Flagged as malicious", accent="rose")
    with col4: render_metric_card("Attack Rate", f"{stats['attack_rate']}%", "Share of flagged flows", accent="amber")

    st.markdown("")
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
        st.markdown("<div class='success-card'>No attack traffic detected in this batch.</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("Network Health Summary")
    health_color = {"Healthy": "success-card", "Low Risk": "info-card", "Medium Risk": "warning-card", "High Risk": "danger-card"}.get(stats["health"], "info-card")
    st.markdown(
        f"<div class='{health_color}'><strong>Status:</strong> {stats['health']}<br>"
        f"<strong>Recommendation:</strong> {stats['recommendation']}</div>",
        unsafe_allow_html=True,
    )


def show_dashboard(model, encoder) -> None:
    render_section_title("Security Operations Overview", "Operational intelligence for the deployed intrusion detection workflow.")

    comparison_df = load_model_comparison()
    best_model_row = comparison_df.loc[comparison_df["Accuracy"].idxmax()]
    accuracy = float(best_model_row["Accuracy"])

    col1, col2, col3, col4 = st.columns(4)
    with col1: render_metric_card("Best Model", "XGBoost", "Highest accuracy on the evaluation set", accent="cyan")
    with col2: render_metric_card("Accuracy", f"{accuracy:.2%}", "On the held-out evaluation set", accent="teal")
    with col3: render_metric_card("Dataset", "CICIDS2017", "Benchmark network intrusion data", accent="violet")
    with col4: render_metric_card("Attack Types", "7", "Normal traffic + 6 intrusion classes", accent="amber")

    st.markdown("")

    tab_overview, tab_pipeline, tab_coverage = st.tabs(["Overview", "Pipeline", "Coverage & Features"])

    with tab_overview:
        st.markdown(
            "<div class='info-card'>Use the sidebar to move between the Threat Scanner (bulk CSV/Excel/JSON), "
            "Manual Prediction (single-flow testing), and Model Performance views.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "This platform pairs a trained **XGBoost** classifier with an analyst-oriented interface for "
            "reviewing network traffic behavior — upload a dataset or test a single flow, and get an "
            "instant classification with a severity rating and recommended action."
        )

    with tab_pipeline:
        steps = [
            ("", "Load", "Load the trained model and label encoder."),
            ("", "Validate", "Validate incoming traffic data or synthetic samples."),
            ("", "Predict", "Generate predictions with severity interpretation."),
            ("", "Report", "Deliver analyst-ready reporting and export support."),
        ]
        cols = st.columns(4)
        for col, (icon, title, desc) in zip(cols, steps):
            with col:
                st.markdown(
                    f"<div class='metric-card accent-cyan' style='text-align:center;'>"
                    f"<div style='font-size:1.6rem;'>{icon}</div>"
                    f"<div style='font-weight:700; margin:0.3rem 0;'>{title}</div>"
                    f"<div class='metric-caption'>{desc}</div></div>",
                    unsafe_allow_html=True,
                )

    with tab_coverage:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Supported Attack Types**")
            attack_tags = [
                ("Normal Traffic", "teal"), ("DoS", "rose"), ("DDoS", "rose"),
                ("Port Scanning", "amber"), ("Bots", "amber"),
                ("Brute Force", "violet"), ("Web Attacks", "violet"),
            ]
            st.markdown("".join(render_tag(name, color) for name, color in attack_tags), unsafe_allow_html=True)
        with col2:
            st.markdown("**Application Features**")
            st.markdown(
                "- Bulk CSV / Excel / JSON scanning\n"
                "- Single-flow manual testing with presets\n"
                "- Severity mapping and health scoring\n"
                "- Analyst-ready reporting and export"
            )


def render_data_preview(df: pd.DataFrame, uploaded_file, file_type: str) -> None:
    render_section_title("File Preview", "A quick look at what was uploaded before running detection.")

    missing_cells = int(df.isna().sum().sum())
    size_label = "—"
    if hasattr(uploaded_file, "size") and uploaded_file.size is not None:
        size_kb = uploaded_file.size / 1024
        size_label = f"{size_kb:,.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:,.2f} MB"

    col1, col2, col3, col4 = st.columns(4)
    with col1: render_metric_card("Rows", f"{len(df):,}", "Flow records detected", accent="cyan")
    with col2: render_metric_card("Columns", f"{df.shape[1]:,}", "Feature columns detected", accent="violet")
    with col3: render_metric_card("Missing Values", f"{missing_cells:,}", "Empty cells across the file", accent="amber")
    with col4: render_metric_card("File Size", size_label, f"Format: {file_type.upper()}", accent="teal")

    with st.expander("Preview first rows", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Column overview"):
        col_info = pd.DataFrame({
            "Column": df.columns.astype(str),
            "Data Type": df.dtypes.astype(str).values,
            "Missing": df.isna().sum().values,
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)


def show_csv_prediction(model, encoder) -> None:
    render_section_title("Threat Scanner", "Upload a network flow dataset (CSV, Excel, or JSON) and scan it in bulk.")

    uploaded_file = st.file_uploader("Upload CSV, Excel (.xlsx), or JSON file", type=["csv", "xlsx", "json"])

    if uploaded_file is None:
        st.markdown(
            "<div class='info-card'>Drop a file above to begin the analysis workflow. "
            "Once uploaded, you'll see a preview of the data before running detection.</div>",
            unsafe_allow_html=True,
        )
        return

    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif file_type == "json":
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return
    except Exception as exc:
        st.error(f"Unable to read the uploaded file: {exc}")
        return

    render_data_preview(df, uploaded_file, file_type)

    st.markdown("---")
    if st.button("Run Prediction", use_container_width=True, type="primary"):
        try:
            result_df = run_prediction(df, model, encoder)
            display_results(result_df)
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def show_manual_prediction(model, encoder) -> None:
    render_section_title("Manual Prediction", "Provide a compact set of traffic indicators and the application synthesizes the remaining model inputs.")

    defaults = dict(
        destination_port=80, protocol="TCP", connection_duration=2.5, total_packets=12,
        average_packet_size=150, max_packet_size=500, flow_bytes_per_second=800, flow_packets_per_second=6,
    )
    for field, value in defaults.items():
        st.session_state.setdefault(f"mp_{field}", value)

    preset_col, apply_col = st.columns([3, 1])
    with preset_col:
        preset_choice = st.selectbox("Start from a traffic profile", list(TRAFFIC_PRESETS.keys()), label_visibility="collapsed")
    with apply_col:
        apply_preset = st.button("Apply Profile", use_container_width=True, disabled=(preset_choice == "Custom"))

    if apply_preset and TRAFFIC_PRESETS[preset_choice]:
        for field, value in TRAFFIC_PRESETS[preset_choice].items():
            st.session_state[f"mp_{field}"] = value
        st.rerun()

    with st.form("manual_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            destination_port = render_slider_card(
                "Destination Port", "The TCP/UDP port the traffic is heading to (e.g. 80 = HTTP, 443 = HTTPS).",
                lambda: st.number_input("Destination Port", 0, 65535, step=1, key="mp_destination_port", label_visibility="collapsed"),
            )
            protocol = render_slider_card(
                "Protocol", "The transport-layer protocol used for this connection.",
                lambda: st.radio("Protocol", ["TCP", "UDP", "ICMP"], key="mp_protocol", horizontal=True, label_visibility="collapsed"),
            )
            connection_duration = render_slider_card(
                "Connection Duration", "How long the flow stayed open, in seconds.",
                lambda: st.slider("Connection Duration", 0.0, 60.0, step=0.1, key="mp_connection_duration", label_visibility="collapsed"),
            )
            total_packets = render_slider_card(
                "Total Packets", "Total number of packets exchanged in the flow.",
                lambda: st.number_input("Total Packets", 1, 5000, step=1, key="mp_total_packets", label_visibility="collapsed"),
            )

        with col2:
            average_packet_size = render_slider_card(
                "Average Packet Size", "Mean packet size across the flow, in bytes.",
                lambda: st.slider("Average Packet Size", 1, 1500, key="mp_average_packet_size", label_visibility="collapsed"),
            )
            max_packet_size = render_slider_card(
                "Maximum Packet Size", "The largest single packet observed in the flow, in bytes.",
                lambda: st.select_slider("Maximum Packet Size", options=list(range(64, 1501, 32)), key="mp_max_packet_size", label_visibility="collapsed"),
            )
            flow_bytes_per_second = render_slider_card(
                "Flow Bytes / Second", "Throughput of the flow, in bytes per second.",
                lambda: st.number_input("Flow Bytes Per Second", 0.0, 200000.0, step=10.0, key="mp_flow_bytes_per_second", label_visibility="collapsed"),
            )
            flow_packets_per_second = render_slider_card(
                "Flow Packets / Second", "Packet rate of the flow, in packets per second.",
                lambda: st.slider("Flow Packets Per Second", 0.0, 1000.0, step=0.5, key="mp_flow_packets_per_second", label_visibility="collapsed"),
            )

        submitted = st.form_submit_button("Generate Prediction", use_container_width=True, type="primary")

    if not submitted: return

    try:
        feature_names = [str(column) for column in model.feature_names_in_]
        input_values = {
            "destination_port": destination_port, "protocol": protocol, "connection_duration": connection_duration,
            "total_packets": total_packets, "average_packet_size": average_packet_size,
            "flow_bytes_per_second": flow_bytes_per_second, "flow_packets_per_second": flow_packets_per_second,
            "max_packet_size": max_packet_size,
        }
        input_df = build_manual_feature_frame(input_values, feature_names)

        prediction = model.predict(input_df)[0]
        label = safe_inverse_transform([prediction], encoder)[0]
        severity = get_severity(label)

        confidence = None
        proba_df = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            encoder_classes = list(getattr(encoder, "classes_", []))
            if len(encoder_classes) != len(proba):
                raise RuntimeError(
                    "Prediction probabilities do not match the saved encoder classes."
                )
            class_labels = [normalize_prediction_label(c) for c in encoder_classes]

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

            result_row = pd.DataFrame([{**input_values, "Prediction": label, "Severity": severity,
                                         "Confidence": f"{confidence:.4f}" if confidence is not None else ""}])
            st.download_button(
                "Download This Result (CSV)", result_row.to_csv(index=False).encode("utf-8"),
                file_name="manual_prediction_result.csv", mime="text/csv", use_container_width=True,
            )

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

        st.markdown(
            f"<div class='success-card'><strong>{best_model}</strong> is currently the best-performing model "
            f"with an accuracy of <strong>{best_accuracy:.2%}</strong>.</div>",
            unsafe_allow_html=True,
        )

        metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
        display_df = comparison_df[["Model", *metric_cols]].copy()

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

    # --- Navigation (top of sidebar) ---
    nav_labels = [f"{icon}  {label}" for _, icon, label in NAV_ITEMS]
    selected_label = st.sidebar.radio("Navigation", nav_labels, index=0, label_visibility="collapsed")
    selected_key = NAV_ITEMS[nav_labels.index(selected_label)][0]

    st.sidebar.markdown("---")

    # --- Brand & system info (bottom of sidebar, compact) ---
    st.sidebar.markdown(
        "<div class='sidebar-brand'>Network Security Center</div>"
        "<div class='sidebar-tagline'>Intrusion detection & traffic analysis</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div class='sb-chip-row'>"
        "<div class='sb-chip chip-cyan'><span class='sb-chip-label'>Model</span><span class='sb-chip-value'>XGBoost</span></div>"
        "<div class='sb-chip chip-violet'><span class='sb-chip-label'>Dataset</span><span class='sb-chip-value'>CICIDS2017</span></div>"
        "<div class='sb-chip chip-amber'><span class='sb-chip-label'>Framework</span><span class='sb-chip-value'>Streamlit</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if selected_key == "dashboard": show_dashboard(model, encoder)
    elif selected_key == "scanner": show_csv_prediction(model, encoder)
    elif selected_key == "manual": show_manual_prediction(model, encoder)
    elif selected_key == "performance": show_model_performance()
    else: show_about()

    render_footer()


if __name__ == "__main__":
    main()