import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# === Config ===
MODEL_DIR = "bert_sentiment_model_7_classes_active_learning/iter_1"  # Adjust if needed
LABEL_MAP = {-3: 0, -2: 1, -1: 2, 0: 3, 1: 4, 2: 5, 3: 6}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# === Load Model ===
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# === Step 1: Load Excel file ===
df = pd.read_excel("finalDataset0.2.xlsx")  # make sure file is in same folder
df.dropna(subset=['text', 'label'], inplace=True)
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(int)

# === Step 2: Predict ===
def predict_sentiment(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        predictions = torch.argmax(probs, dim=1).tolist()
        return predictions

# Batch prediction
predictions = []
batch_size = 16
for i in range(0, len(df), batch_size):
    batch_texts = df['text'][i:i+batch_size].tolist()
    preds = predict_sentiment(batch_texts)
    predictions.extend(preds)

df['predicted_label'] = [REVERSE_LABEL_MAP[p] for p in predictions]

# === Step 3: Visualizations ===

# Sentiment Distribution
plt.figure(figsize=(8, 4))
sns.countplot(x='label', data=df, order=sorted(df['label'].unique()))
plt.title("True Sentiment Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()

# Confusion Matrix
true_labels = df['label'].map(LABEL_MAP).tolist()
pred_labels = [LABEL_MAP[l] for l in df['predicted_label']]
cm = confusion_matrix(true_labels, pred_labels, labels=range(7))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=REVERSE_LABEL_MAP.values(), yticklabels=REVERSE_LABEL_MAP.values(), cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Save predictions for reference
df.to_csv("predicted_output.csv", index=False)
print("✅ Predictions saved to predicted_output.csv")
