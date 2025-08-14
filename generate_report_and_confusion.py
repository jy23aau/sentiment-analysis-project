import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():
    # Paths
    test_csv_path = "data/processed/test_data.csv"
    model_dir = "models/saved_models/best_model"
    output_metrics_dir = "results/metrics"
    os.makedirs(output_metrics_dir, exist_ok=True)

    # Load test dataset with true labels
    test_df = pd.read_csv(test_csv_path)
    y_true = test_df["model_label"].values
    texts = test_df["text"].tolist()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Batch prediction setup
    batch_size = 32
    all_preds = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)

    y_pred = np.array(all_preds)

    # Define label names consistent with training
    target_names = ['Negative (-1)', 'Neutral (0)', 'Positive (1)']

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("Classification Report:\n", report)

    # Save classification report text
    with open(os.path.join(output_metrics_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print confusion matrix
    print("Confusion Matrix:\n", cm)

    # Save confusion matrix CSV
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(os.path.join(output_metrics_dir, "confusion_matrix.csv"))

    # Visualize confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues", colorbar=True)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_metrics_dir, "confusion_matrix.png"))
    plt.show()

if __name__ == "__main__":
    main()
