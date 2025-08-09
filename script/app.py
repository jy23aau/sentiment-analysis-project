import os
import base64
import subprocess
import pandas as pd
import streamlit as st
from predict_sentiment import predict_sentiment  # your prediction function

DATA_DIR = "data/processed"
RAW_DATA_FILE = os.path.join(DATA_DIR, "raw_feedback.csv")
UNLABELED_FILE = os.path.join(DATA_DIR, "unlabeled_data.csv")
ANNOTATION_FILE = os.path.join(DATA_DIR, "manual_annotations.csv")
SAMPLES_TO_ANNOTATE_FILE = os.path.join(DATA_DIR, "samples_to_annotate.csv")
PREDICTIONS_FILE = os.path.join("data/outputs", "model_sentiment_predictions.csv")
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

emoji_map = {
    -3: "😡 Very Negative",
    -2: "😠 Negative",
    -1: "🙁 Slightly Negative",
     0: "😐 Neutral",
     1: "🙂 Slightly Positive",
     2: "😃 Positive",
     3: "🤩 Very Positive"
}

def save_uploaded_file(uploaded_file, save_path):
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding='utf-8')
    return df

def run_cmd(cmd):
    with st.spinner(f"Running `{cmd}`..."):
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"❌ Error running `{cmd}`:\n{result.stderr}")
            return False
        else:
            st.success(f"✅ Completed `{cmd}`.")
            st.text(result.stdout)
            return True

if 'loop_step' not in st.session_state:
    st.session_state.loop_step = 0  # 0 = not started

if 'labeling_index' not in st.session_state:
    st.session_state.labeling_index = 0

if 'running_loop' not in st.session_state:
    st.session_state.running_loop = False

if st.sidebar.button("Run Active Learning Loop"):
    st.session_state.loop_step = 1
    st.session_state.labeling_index = 0
    st.session_state.running_loop = True
    st.experimental_rerun()

st.sidebar.markdown("### Tools")
tool_page = st.sidebar.radio("", [
    "Sentiment Legend",
    "Single Feedback Prediction"
])

def manual_annotation_ui():
    try:
        samples_df = pd.read_csv(SAMPLES_TO_ANNOTATE_FILE)
    except FileNotFoundError:
        st.error("No samples to annotate found. Please run batch prediction and sample selection first.")
        return False

    total = len(samples_df)
    idx = st.session_state.labeling_index

    if idx >= total:
        st.success("🎉 All samples annotated!")
        if st.button("Continue Active Learning Loop"):
            st.session_state.loop_step = 6
            st.session_state.labeling_index = 0
        return False

    sample = samples_df.iloc[idx]
    st.markdown(f"Sample {idx + 1} of {total}")
    st.write(sample['text'])

    sentiment = st.radio(
        "Select Sentiment:",
        options=list(emoji_map.keys()),
        format_func=lambda x: f"{emoji_map.get(x, 'Unknown')} ({x})"
    )

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

    st.progress((idx + 1) / total)
    return True

def active_learning_loop():
    if st.session_state.loop_step == 0:
        st.info("Click 'Run Active Learning Loop' button in sidebar to start.")

    elif st.session_state.loop_step == 1:
        st.header("Step 1: Upload Raw Feedback")
        uploaded_file = st.file_uploader("Upload Excel or CSV of raw feedback")
        if uploaded_file:
            save_uploaded_file(uploaded_file, RAW_DATA_FILE)
            st.success("Raw feedback uploaded successfully!")
            if st.button("Continue"):
                st.session_state.loop_step = 2
                st.experimental_rerun()
        else:
            st.info("Please upload your raw feedback data to begin.")

    elif st.session_state.loop_step == 2:
        st.header("Step 2: Prepare Dataset")
        if run_cmd("python scripts/prepare_dataset.py"):
            st.session_state.loop_step = 3

    elif st.session_state.loop_step == 3:
        st.header("Step 3: Batch Prediction")
        if run_cmd("python scripts/predict_sentiment.py"):
            st.session_state.loop_step = 4

    elif st.session_state.loop_step == 4:
        st.header("Step 4: Select Hardest Samples")
        if run_cmd("python scripts/merge_and_sample.py"):
            st.session_state.loop_step = 5

    elif st.session_state.loop_step == 5:
        st.header("Step 5: Manual Annotation")
        annotated = manual_annotation_ui()
        if not annotated:
            st.stop()

    elif st.session_state.loop_step == 6:
        st.header("Step 6: Train / Retrain Model")
        if st.button("Start Training"):
            if run_cmd("python scripts/model_train.py"):
                st.session_state.loop_step = 7

    elif st.session_state.loop_step == 7:
        st.header("Step 7: Visualizations")
        st.info("Run visualization app separately or refresh this page after running visualizations.")
        if st.button("Restart Active Learning Loop"):
            st.session_state.loop_step = 1
            st.session_state.labeling_index = 0
            st.experimental_rerun()

def main():
    st.title("Active Learning Sentiment Analysis System")
    if st.session_state.running_loop:
        active_learning_loop()
    else:
        if tool_page == "Sentiment Legend":
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

if __name__ == "__main__":
    main()
