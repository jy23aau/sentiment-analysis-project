import os
import base64
import subprocess
import pandas as pd
import streamlit as st
from predict_sentiment import predict_sentiment  # your prediction function

# File paths
DATA_DIR = "data/processed"
UNLABELED_FILE = os.path.join(DATA_DIR, "unlabeled_data.csv")
ANNOTATION_FILE = os.path.join(DATA_DIR, "manual_annotations.csv")
TRAINING_DATA_FILE = os.path.join(DATA_DIR, "training_data_labeled.csv")
SAMPLES_TO_ANNOTATE_FILE = os.path.join(DATA_DIR, "samples_to_annotate.csv")
PREDICTIONS_FILE = os.path.join("data/outputs", "model_sentiment_predictions.csv")
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sentiment emojis mapping
emoji_map = {
    -3: "😡 Very Negative",
    -2: "😠 Negative",
    -1: "🙁 Slightly Negative",
     0: "😐 Neutral",
     1: "🙂 Slightly Positive",
     2: "😃 Positive",
     3: "🤩 Very Positive"
}

# Save uploaded files with utf-8 encoding
def save_uploaded_file(uploaded_file, save_path):
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding='utf-8')
    return df

# Run shell commands and show output/errors
def run_cmd(cmd):
    with st.spinner(f"Running {cmd}..."):
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"❌ Error running {cmd}:\n{result.stderr}")
            return False
        else:
            st.success(f"✅ Completed {cmd}.")
            st.text(result.stdout)
            return True

# Initialize Streamlit session state variables
if 'loop_step' not in st.session_state:
    st.session_state.loop_step = 1

if 'labeling_index' not in st.session_state:
    st.session_state.labeling_index = 0

def display_loop_status():
    steps = [
        "Upload Raw Feedback",
        "Batch Prediction",
        "Manual Annotation",
        "Train / Retrain Model",
        "Visualizations"
    ]
    st.sidebar.markdown("## 🔄 Active Learning Loop Status")
    for i, step in enumerate(steps, 1):
        if i < st.session_state.loop_step:
            st.sidebar.markdown(f"✅ {step}")
        elif i == st.session_state.loop_step:
            st.sidebar.markdown(f"➡️ **{step}**")
        else:
            st.sidebar.markdown(f"⬜ {step}")

# Sidebar navigation for steps and tools
st.sidebar.markdown("### Steps")
page = st.sidebar.radio("", [
    "Upload Raw Feedback",
    "Batch Prediction",
    "Manual Annotation",
    "Train / Retrain Model",
    "Visualizations"
])

st.sidebar.markdown("### Tools")
tool_page = st.sidebar.radio("", [
    "Sentiment Legend",
    "Single Feedback Prediction"
])

display_loop_status()

def manual_annotation_ui():
    # Load samples to annotate
    try:
        samples_df = pd.read_csv(SAMPLES_TO_ANNOTATE_FILE)
    except FileNotFoundError:
        st.error("No samples to annotate found. Run batch prediction and sample selection first.")
        return False

    total_samples = len(samples_df)
    current_idx = st.session_state.labeling_index

    if current_idx >= total_samples:
        st.success("🎉 All samples annotated!")
        if st.button("Continue Active Learning Loop"):
            st.session_state.loop_step += 1
            st.session_state.labeling_index = 0
        return False

    sample = samples_df.iloc[current_idx]
    st.markdown(f"Sample {current_idx + 1} of {total_samples}")
    st.write(sample['text'])

    sentiment = st.radio("Select Sentiment:", options=list(emoji_map.keys()),
                         format_func=lambda x: f"{emoji_map.get(x, 'Unknown')} ({x})")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Label"):
            if not os.path.exists(ANNOTATION_FILE):
                pd.DataFrame(columns=['id', 'text', 'label']).to_csv(ANNOTATION_FILE, index=False)
            ann_df = pd.read_csv(ANNOTATION_FILE)
            new_row = pd.DataFrame([{'id': sample['id'], 'text': sample['text'], 'label': sentiment}])
            ann_df = pd.concat([ann_df, new_row], ignore_index=True)
            ann_df.to_csv(ANNOTATION_FILE, index=False)
            st.session_state.labeling_index += 1
            st.experimental_rerun()
    with col2:
        if st.button("Skip"):
            st.session_state.labeling_index += 1
            st.experimental_rerun()

    st.progress((current_idx + 1) / total_samples)
    return True

def active_learning_loop():
    if st.session_state.loop_step == 1:
        st.header("Step 1: Upload Raw Feedback")
        uploaded_file = st.file_uploader("Upload Excel or CSV of raw feedback")
        if uploaded_file:
            df = save_uploaded_file(uploaded_file, os.path.join(DATA_DIR, "raw_feedback.csv"))
            st.success(f"Uploaded {len(df)} samples.")
            st.dataframe(df.head())
            if st.button("Prepare Dataset"):
                if run_cmd("python scripts/prepare_dataset.py"):
                    st.session_state.loop_step = 2

    elif st.session_state.loop_step == 2:
        st.header("Step 2: Batch Prediction")
        uploaded_file = st.file_uploader("Upload Excel or CSV file with 'text' column for prediction")
        if uploaded_file:
            df = save_uploaded_file(uploaded_file, UNLABELED_FILE)
            st.success(f"Saved uploaded file as unlabeled data pool with {len(df)} samples.")
            st.dataframe(df.head())
            if st.button("Run Batch Prediction"):
                if run_cmd("python scripts/predict_sentiment.py"):
                    st.session_state.loop_step = 3
                    pred_df = pd.read_csv(PREDICTIONS_FILE)
                    st.dataframe(pred_df.head())
                    csv = pred_df.to_csv(index=False).encode()
                    b64 = base64.b64encode(csv).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)

    elif st.session_state.loop_step == 3:
        st.header("Step 3: Manual Annotation")
        manual_annotation_ui()

    elif st.session_state.loop_step == 4:
        st.header("Step 4: Train / Retrain Model")
        st.write("Trigger model training or retraining using the labeled dataset.")
        if st.button("Start Training"):
            if run_cmd("python scripts/model_train.py"):
                st.session_state.loop_step = 5

    elif st.session_state.loop_step == 5:
        st.header("Step 5: Visualizations")
        run_cmd("streamlit run scripts/visualization_app.py")

def main():
    st.title("Active Learning Sentiment Analysis System")

    # Show the loop or tools pages
    if page in ["Upload Raw Feedback", "Batch Prediction", "Manual Annotation", "Train / Retrain Model", "Visualizations"]:
        active_learning_loop()
    elif tool_page == "Sentiment Legend":
        st.title("Sentiment Class Legend")
        for cls, desc in emoji_map.items():
            st.write(f"Class {cls}: {desc}")
    elif tool_page == "Single Feedback Prediction":
        st.title("Single Feedback Sentiment Prediction")
        user_input = st.text_area("Enter your feedback text here:", height=150)
        if st.button("Predict Sentiment"):
            if user_input.strip():
                label, confidence = predict_sentiment(user_input)
                sentiment_text = emoji_map.get(label, f"Label {label}")
                st.markdown(f"<h2>Prediction: Class {label} ({sentiment_text})</h2>", unsafe_allow_html=True)
                st.write(f"Confidence Score: **{confidence:.2f}**")
            else:
                st.warning("Please enter some feedback text to predict.") 