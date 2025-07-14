import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# === Configuration ===
MODEL_DIR = "bert_sentiment_model_7_classes_active_learning/iter_1"  # Use latest iteration folder
LABEL_MAP = {0: -3, 1: -2, 2: -1, 3: 0, 4: 1, 5: 2, 6: 3}  # Reverse mapping

# === Load Model & Tokenizer ===
print("Loading model...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# === Prediction Function ===
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        predicted_label = LABEL_MAP[predicted_class]
        confidence = probs[0][predicted_class].item()
    return predicted_label, confidence

# === Try it out ===
while True:
    text = input("\nEnter student feedback (or type 'exit' to quit):\n> ")
    if text.lower() == 'exit':
        break
    label, conf = predict_sentiment(text)
    print(f"Predicted Sentiment: {label}  (Confidence: {conf:.2f})")
