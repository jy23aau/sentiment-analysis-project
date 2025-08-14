import gradio as gr
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# Load your trained model and tokenizer (adjust path if needed)
MODEL_DIR = 'models/saved_models/best_model'
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

id_to_label = {0: -1, 1: 0, 2: 1}

def predict_sentiment(texts):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    encodings = {k: v.to(device) for k, v in encodings.items()}
    with torch.no_grad():
        logits = model(**encodings).logits
    preds = torch.argmax(logits, dim=-1).cpu().tolist()
    return [id_to_label[p] for p in preds]

def single_predict(text):
    if not text.strip():
        return "Please enter some feedback text."
    pred = predict_sentiment([text])[0]
    return f"Predicted sentiment label: {pred}"

def batch_predict(file):
    df = pd.read_csv(file.name)
    if "text" not in df.columns:
        return "CSV must have a 'text' column."
    texts = df["text"].dropna().tolist()
    preds = predict_sentiment(texts)
    df["predicted_sentiment"] = preds
    
    # Plot distribution
    counts = df["predicted_sentiment"].value_counts().sort_index()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax, color=['red', 'gray', 'green'])
    ax.set_xlabel('Sentiment Label')
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Distribution')
    plt.tight_layout()

    results_csv = df.to_csv(index=False)
    return df.head(10), fig, (results_csv, "predictions.csv")

with gr.Blocks() as demo:
    gr.Markdown("# Sentiment Prediction with DistilBERT")
    
    with gr.Tab("Single Feedback"):
        text_input = gr.Textbox(lines=4, label="Enter feedback text")
        single_output = gr.Textbox(label="Prediction")
        btn1 = gr.Button("Predict")
        btn1.click(single_predict, inputs=text_input, outputs=single_output)
    
    with gr.Tab("Batch Feedback"):
        file_input = gr.File(label="Upload CSV with a 'text' column")
        batch_table = gr.Dataframe(headers="auto", datatype=["str"]*10, label="Predictions")
        batch_plot = gr.Plot(label="Sentiment Distribution")
        download_btn = gr.File(label="Download CSV with Predictions")
        btn2 = gr.Button("Predict Batch")
        btn2.click(batch_predict, inputs=file_input, outputs=[batch_table, batch_plot, download_btn])

demo.launch()
