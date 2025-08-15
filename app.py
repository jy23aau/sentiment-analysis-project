import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt

# ---- Load your trained model here ----
MODEL_DIR = "models/saved_models/best_model"  # Make sure this path matches your folder
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

id_to_label = {0: -1, 1: 0, 2: 1}

def predict_sentiment(texts):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encodings).logits
    preds = torch.argmax(logits, dim=-1).cpu().tolist()
    return [id_to_label[p] for p in preds]

st.markdown(
    "<h1 style='text-align: center; color: #42b6f5;'>🌟 Student Sentiment Detector 🌟</h1>", unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align: center;'>Classify feedback as Positive (1), Neutral (0), or Negative (-1)</h4>", unsafe_allow_html=True)

# Single feedback input
st.subheader("🔹 Try Single Feedback")
user_text = st.text_area("Type your feedback here:", "")
if st.button("👉 Detect Sentiment (Single Text)"):
    if user_text.strip():
        sentiment = predict_sentiment([user_text])[0]
        st.success(f"Detected Sentiment: **{sentiment}** (1=Positive, 0=Neutral, -1=Negative)")
    else:
        st.warning("Please enter your feedback.")

# Batch upload input
st.subheader("🔹 Batch Feedback (Upload CSV file)")
csv_file = st.file_uploader("Upload CSV with a column named 'text'", type="csv")

if csv_file:
    df = pd.read_csv(csv_file)
    if "text" not in df.columns:
        st.error("Your CSV needs a column named 'text'.")
    else:
        if st.button("👉 Detect Sentiment (Batch File)"):
            preds = predict_sentiment(df["text"].fillna("").tolist())
            df["Predicted Sentiment"] = preds
            st.write("### Results Table", df)
            st.write("### Sentiment Distribution")
            count_df = df["Predicted Sentiment"].value_counts().sort_index()
            fig, ax = plt.subplots()
            colors = ['#e74c3c', '#f1c40f', '#2ecc71']  # red, yellow, green
            labels = ['Negative (-1)', 'Neutral (0)', 'Positive (1)']
            ax.bar(labels, count_df.values, color=colors)
            ax.set_ylabel('Count')
            st.pyplot(fig)
            st.download_button("Download results as CSV", df.to_csv(index=False), "sentiment_results.csv")

