import streamlit as st
import subprocess
import os
import sys
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# === Constants ===
DATA_PROCESSED = "data/processed"
DATA_OUTPUTS = "data/outputs"
PROCESSED_FILE = os.path.join(DATA_PROCESSED, "processed_feedback_dataset.csv")
MODEL_PATH = "data/outputs/bert_sentiment_model"
LABEL_MAP = {i: label for i, label in enumerate(range(-3, 4))}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_script(script_name):
    python_executable = sys.executable
    cmd = f'"{python_executable}" scripts/{script_name}'
    with st.spinner(f"Running {script_name}..."):
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        st.success(f"{script_name} completed successfully.")
    else:
        st.error(f"Error running {script_name}:\n{result.stderr}")

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

def predict_sentiments(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confs, preds = torch.max(probs, dim=1)
    return [LABEL_MAP[p.item()] for p in preds], [c.item() for c in confs]

# === Enhanced JD-Style Streamlit UI ===
st.set_page_config(page_title="Sentiment Active Learning Dashboard", layout="centered")
st.markdown(
    """
    <style>
        .main {background-color: #f6fafd;}
        .st-bb {background-color: #f2f5f7;}
        .stButton>button {background-color: #2e7eef; color: white;}
        .stDownloadButton>button {background-color: #2e7eef; color: white;}
        .stCheckbox>div>div {color: #2e7eef;}
        h1 {color: #2e7eef;}
        .stTabs [data-baseweb="tab-list"] {background: #e5efff;}
        .stTabs [data-baseweb="tab"] {font-weight: bold;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Active Learning Workflow Dashboard")
st.caption("Streamlined, colorful & interactive for next-generation feedback curation and sentiment modeling.")

with st.container():
    st.markdown("### <span style='color:#2e7eef'>Step 1: Upload Raw Excel File</span>", unsafe_allow_html=True)
    uploaded_excel = st.file_uploader("📤 Upload Excel file with `text*` and (optional) `label*` columns:", type=["xlsx"], key="upload_xlsx")
    if uploaded_excel:
        try:
            df = pd.read_excel(uploaded_excel, engine="openpyxl")
            st.success(f"✅ Loaded {len(df)} rows.")
            st.write("Detected columns:", df.columns.tolist())
            text_columns = [col for col in df.columns if col.startswith("text")]
            label_columns = [col for col in df.columns if col.startswith("label")]

            if not text_columns:
                st.error("❌ No 'text*' columns found.")
            else:
                df["text"] = df[text_columns].fillna('').astype(str).agg(" ".join, axis=1).str.strip()
                df["id"] = range(1, len(df) + 1)
                df["true_label"] = df[label_columns[0]] if label_columns else pd.NA
                df["predicted_label"] = pd.NA

                final_df = df[["id", "text", "true_label", "predicted_label"]]
                os.makedirs(DATA_PROCESSED, exist_ok=True)
                final_df.to_csv(PROCESSED_FILE, index=False, encoding="utf-8")
                st.success(f"✅ Processed and saved to: {PROCESSED_FILE}")
                st.dataframe(final_df.head(), use_container_width=True)

                # === Pipeline with vibrant accent boxes ===
                st.markdown('<div style="background-color:#e5efff;padding:12px;border-radius:8px;">'
                            "### <span style='color:#2e7eef'>🔁 Active Learning Pipeline</span>"
                            '</div>', unsafe_allow_html=True)

                ann_path = os.path.join(DATA_PROCESSED, "manual_annotations.csv")
                if os.path.exists(ann_path):
                    run_script("merge_and_sample.py")
                else:
                    st.info("ℹ️ No manual annotations found — skipping merge.")

                run_script("model_train.py")

                df_pred = pd.read_csv(PROCESSED_FILE)
                texts = df_pred['text'].astype(str).tolist()
                preds, confs = predict_sentiments(texts)
                df_pred['predicted_label'] = preds
                df_pred['confidence'] = confs

                os.makedirs(DATA_OUTPUTS, exist_ok=True)
                output_path = os.path.join(DATA_OUTPUTS, "model_sentiment_predictions.csv")
                df_pred.to_csv(output_path, index=False, encoding="utf-8")
                st.success("✅ Predictions saved to: " + output_path)
                run_script("select_hardest_samples.py")

                st.success("✅ Workflow complete! Open the annotation app below.")
                with st.expander("How to Annotate", expanded=False):
                    st.code("streamlit run scripts/annotation_app.py")

        except Exception as e:
            st.error(f"❌ Failed to read uploaded file: {e}")

with st.container():
    st.markdown('<div style="background-color:#e5f3e3;padding:10px;border-radius:8px;">'
                "### <span style='color:#33aa55'>Step 2: Visualize Results</span>"
                '</div>', unsafe_allow_html=True)
    st.info("Run the visualization UI in another tab:")
    st.code("streamlit run scripts/visualization_app.py")

st.markdown("---")
st.write("🔁 Continue with annotation, retraining, and monitoring performance...")

# === Innovative Sentiment Testing Tabs (colorful JD style) ===
tab1, tab2, tab3 = st.tabs([
    "🔤 Predict Single Feedback",
    "🎭 Emoji Style (-3 to 3)",
    "📥 Batch Upload"
])

with tab1:
    st.markdown('<div style="background-color:#e5efff;padding:8px;border-radius:8px;">'
                "<span style='color:#2e7eef;font-size:18px;font-weight:bold;'>🔍 Predict Sentiment</span>"
                '</div>', unsafe_allow_html=True)
    input_text = st.text_area(
        "Type feedback here:",
        placeholder="e.g. I loved the way the instructor explained...",
        key="single_input"
    )
    if st.button("Predict Sentiment", key="single_pred_btn"):
        if input_text.strip():
            preds, confs = predict_sentiments([input_text])
            sentiment_color = "#33aa55" if preds[0] > 0 else ("#c83240" if preds[0] < 0 else "#faad14")
            st.markdown(
                f"<div style='background:{sentiment_color};color:white;padding:8px 12px;border-radius:8px;display:inline-block;margin-top:8px;'>"
                f"**Predicted Sentiment:** {preds[0]} &nbsp;|&nbsp; Confidence: {confs[0]:.2f}"
                "</div>", unsafe_allow_html=True
            )
        else:
            st.warning("Please enter some feedback text.")

with tab2:
    st.markdown('<div style="background-color:#fff1e1;padding:8px;border-radius:8px;">'
                "<span style='color:#eb7157;font-size:18px;font-weight:bold;'>🎭 Emoji Sentiment Detector</span>"
                '</div>', unsafe_allow_html=True)
    emo_input = st.text_input("Enter feedback (emojis optional):", placeholder="That class was 🔥🔥🔥", key="emoji_input")
    if st.button("🧠 Predict Sentiment (Emoji)", key="emoji_pred_btn"):
        if emo_input.strip():
            preds, confs = predict_sentiments([emo_input])
            label = preds[0]
            emojis = {-3: "😡", -2: "😠", -1: "😕", 0: "😐", 1: "🙂", 2: "😃", 3: "🤩"}
            sentiment_color = "#33aa55" if label > 0 else ("#c83240" if label < 0 else "#faad14")
            st.markdown(
                f"<div style='background:{sentiment_color};color:white;padding:8px 12px;border-radius:8px;display:inline-block;margin-top:8px;'>"
                f"**Prediction:** {label} {emojis.get(label, '')} | Confidence: {confs[0]:.2f}"
                "</div>", unsafe_allow_html=True
            )
        else:
            st.warning("Type something to predict!")

with tab3:
    st.markdown('<div style="background-color:#e5efff;padding:8px;border-radius:8px;">'
                "<span style='color:#2e7eef;font-size:18px;font-weight:bold;'>📥 Batch Upload for Predictions</span>"
                '</div>', unsafe_allow_html=True)
    batch_file = st.file_uploader("Upload CSV with a 'text' column:", type=["csv"], key="batch_upload")
    if batch_file:
        try:
            batch_df = pd.read_csv(batch_file)
            if "text" not in batch_df.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                texts = batch_df["text"].astype(str).tolist()
                preds, confs = predict_sentiments(texts)
                batch_df["predicted_label"] = preds
                batch_df["confidence"] = confs
                st.success("✅ Batch prediction complete!")
                st.dataframe(batch_df.head(), use_container_width=True)
                st.download_button("📥 Download Predictions", batch_df.to_csv(index=False).encode("utf-8"), "batch_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")
