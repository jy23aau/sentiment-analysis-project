# train_distilbert_3class.py

import os
import random
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)


# -----------------------
# Config parameters
# -----------------------
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "results"
BEST_MODEL_DIR = os.path.join("models", "saved_models", "best_model")
SEED = 42
TEST_SIZE = 0.1
VAL_SIZE = 0.1  # fraction of remaining after test split
MAX_LENGTH = 256
BATCH_TRAIN = 16
BATCH_EVAL = 32
LEARNING_RATE = 2e-5
EPOCHS = 4
WEIGHT_DECAY = 0.01
FP16 = torch.cuda.is_available()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
set_seed(SEED)
torch.manual_seed(SEED)


# -----------------------
# Load data (CSV or Excel)
# -----------------------
def load_data():
    """
    Load consolidated CSV 'id-text-label.csv' with columns ['text', 'label'],
    or fallback to 'EXCEL.xlsx' to extract feedback-sentiment pairs.
    """
    if os.path.exists("id-text-label.csv"):
        df = pd.read_csv("id-text-label.csv")
        df.columns = [c.lower().strip() for c in df.columns]
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns")
        df = df[["text", "label"]].dropna()
        return df

    if not os.path.exists("EXCEL.xlsx"):
        raise FileNotFoundError(
            "Neither 'id-text-label.csv' nor 'EXCEL.xlsx' found in working directory"
        )

    # Read Excel, pair "... Feedback" with "... Sentiment" columns
    xls = pd.ExcelFile("EXCEL.xlsx")
    df_raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    feedback_cols = [c for c in df_raw.columns if "feedback" in str(c).lower()]
    rows = []
    for fcol in feedback_cols:
        candidates = [c for c in df_raw.columns if "sentiment" in str(c).lower()]
        key = str(fcol).split()[0].lower()
        match = next(
            (scol for scol in candidates if str(scol).split()[0].lower() == key), None
        )
        if match is None:
            continue
        sub = df_raw[[fcol, match]].dropna()
        sub.columns = ["text", "label"]
        sub = sub[pd.to_numeric(sub["label"], errors="coerce").notnull()]
        sub["text"] = sub["text"].astype(str).str.strip()
        sub = sub[sub["text"] != ""]
        rows.append(sub)
    if not rows:
        raise ValueError("No Feedback/Sentiment column pairs found in Excel")
    df = pd.concat(rows, ignore_index=True)
    return df[["text", "label"]]


# -----------------------
# Map labels from [-3..3] to {-1,0,1}
# -----------------------
def map_label_to_3class(label):
    label = float(label)
    if label <= -1:
        return -1
    elif label == 0:
        return 0
    else:
        return 1


# -----------------------
# Encode labels for model training {0,1,2}
# -----------------------
LABEL_TO_ID = {-1: 0, 0: 1, 1: 2}
ID_TO_LABEL = {0: -1, 1: 0, 2: 1}


# -----------------------
# Prepare dataset with mapped and encoded labels
# -----------------------
def prepare_dataset(df):
    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].notna()]
    df["label_3class"] = df["label"].apply(map_label_to_3class)
    df["model_label"] = df["label_3class"].map(LABEL_TO_ID)
    df = df[["text", "label_3class", "model_label"]].dropna()
    df = df[df["text"].astype(str).str.strip() != ""]
    return df


# -----------------------
# Stratified train/val/test split
# -----------------------
def create_splits(df):
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["model_label"]
    )
    val_size_adj = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_size_adj, random_state=SEED, stratify=train_val["model_label"]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# -----------------------
# Dataset class for HuggingFace Trainer
# -----------------------
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


# -----------------------
# Metrics computation
# -----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


# -----------------------
# Main function for training and evaluation
# -----------------------
def main():
    print("Loading data...")
    df = load_data()
    df = prepare_dataset(df)
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['label_3class'].value_counts().sort_index()}")

    train_df, val_df, test_df = create_splits(df)
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")

    # Save splits
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train_data.csv", index=False)
    val_df.to_csv("data/processed/val_data.csv", index=False)
    test_df.to_csv("data/processed/test_data.csv", index=False)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={i: str(ID_TO_LABEL[i]) for i in range(3)},
        label2id={str(k): v for k, v in LABEL_TO_ID.items()},
    )

    # Tokenize
    def tokenize_texts(texts):
        return tokenizer(texts, truncation=True, padding=False, max_length=MAX_LENGTH)

    train_encodings = tokenize_texts(train_df["text"].tolist())
    val_encodings = tokenize_texts(val_df["text"].tolist())
    test_encodings = tokenize_texts(test_df["text"].tolist())

    train_dataset = SimpleDataset(train_encodings, train_df["model_label"].tolist())
    val_dataset = SimpleDataset(val_encodings, val_df["model_label"].tolist())
    test_dataset = SimpleDataset(test_encodings, test_df["model_label"].tolist())

    # Training arguments (note: use eval_strategy for transformers v4.55.2+)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        eval_strategy="epoch",  # Changed from 'evaluation_strategy' to 'eval_strategy'
        save_strategy="epoch",
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        save_total_limit=2,
        fp16=FP16,
        report_to=[],  # disable reporting to wandb etc.
        seed=SEED,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print(val_metrics)

    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(test_metrics)

    # Save classification report and confusion matrix
    test_pred_logits = trainer.predict(test_dataset).predictions
    test_preds = np.argmax(test_pred_logits, axis=-1)
    y_true = np.array(test_df["model_label"])

    target_names = [str(ID_TO_LABEL[i]) for i in range(3)]
    cls_report = classification_report(y_true, test_preds, target_names=target_names, digits=4)
    cm = confusion_matrix(y_true, test_preds, labels=[0, 1, 2])
    os.makedirs(os.path.join(OUTPUT_DIR, "metrics"), exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "metrics", "classification_report.txt"), "w") as f:
        f.write(cls_report)
    pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in target_names],
        columns=[f"pred_{label}" for label in target_names],
    ).to_csv(os.path.join(OUTPUT_DIR, "metrics", "confusion_matrix.csv"))

    summary = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "seed": SEED,
        "epochs": EPOCHS,
        "max_length": MAX_LENGTH,
        "batch_train": BATCH_TRAIN,
        "batch_eval": BATCH_EVAL,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    }
    with open(os.path.join(OUTPUT_DIR, "metrics", "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save best model and tokenizer
    trainer.save_model(BEST_MODEL_DIR)
    tokenizer.save_pretrained(BEST_MODEL_DIR)
    print(f"Best model saved to {BEST_MODEL_DIR}")

    # Demo inference function
    def predict_sentiment(texts):
        enc = tokenizer(
            texts, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        enc = {k: v.to(trainer.model.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = trainer.model(**enc).logits
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        return [ID_TO_LABEL[i] for i in pred_ids]

    demo_texts = [
        "Lectures are clear and helpful.",
        "The paper checking seems unfair and biased.",
        "It was okay, not good or bad.",
    ]
    demo_preds = predict_sentiment(demo_texts)
    print("Demo predictions:")
    for text, pred in zip(demo_texts, demo_preds):
        print(f"Text: {text}\nPredicted sentiment label: {pred}\n")


if __name__ == "__main__":
    main()
