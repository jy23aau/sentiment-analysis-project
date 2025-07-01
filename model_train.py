# model_train.py
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
import pandas as pd

df = pd.read_csv("data/cleaned.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(df["cleaned"], df["class"], test_size=0.2)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels): self.encodings = encodings; self.labels = labels
    def __getitem__(self, idx): return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {"labels": torch.tensor(self.labels[idx])}
    def __len__(self): return len(self.labels)

train_dataset = Dataset(train_encodings, list(train_labels))
val_dataset = Dataset(val_encodings, list(val_labels))

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(output_dir='./results', evaluation_strategy="epoch", per_device_train_batch_size=16, per_device_eval_batch_size=64, num_train_epochs=3)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset)

trainer.train()
