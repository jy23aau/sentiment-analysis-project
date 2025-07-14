import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset
import os
import shutil # For moving files/folders

# --- Configuration ---
INITIAL_LABELED_DATA_FILE = 'dataset.csv' # Your starting labeledpip install -r requirements.txt
 data
UNLABELED_POOL_FILE = 'unlabeled_data.csv' # Your starting unlabeled data pool
NEWLY_LABELED_DATA_FILE = 'newly_labeled_data.csv' # Output from annotation app
MODEL_NAME = 'bert-base-uncased'
NUM_LABELS = 7 # Labels: -3, -2, -1, 0, 1, 2, 3 -> mapped to 0, 1, 2, 3, 4, 5, 6
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
OUTPUT_DIR = './results_active_learning' # Output directory for models/logs
MODEL_SAVE_PATH = './bert_sentiment_model_7_classes_active_learning' # Path for current model
NUM_SAMPLES_TO_SELECT_PER_ITER = 5 # Number of samples to query for labeling per iteration
MAX_ACTIVE_LEARNING_ITERATIONS = 2 # Number of active learning cycles to run

# Updated label mapping for 7 classes
LABEL_MAPPING = {-3: 0, -2: 1, -1: 2, 0: 3, 1: 4, 2: 5, 3: 6}
REVERSE_LABEL_MAPPING = {0: -3, 1: -2, 2: -1, 3: 0, 4: 1, 5: 2, 6: 3} # For displaying original labels

# --- Utility Functions ---

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.encodings.input_ids) # Use length of input_ids

def load_and_preprocess_data(file_path, is_labeled=True):
    """Loads and preprocesses data for training or prediction."""
    df = pd.read_csv(file_path)
    df.dropna(subset=['text'], inplace=True)
    df['text'] = df['text'].astype(str).str.lower()

    if is_labeled:
        if 'label' not in df.columns:
            raise ValueError(f"Labeled CSV '{file_path}' must contain 'text' and 'label' columns.")
        df['label'] = df['label'].astype(int)
        valid_labels = list(LABEL_MAPPING.keys())
        df_filtered = df[df['label'].isin(valid_labels)].copy()
        if df_filtered.empty:
            raise ValueError(f"No data with labels {valid_labels} found in '{file_path}'.")
        df_filtered['mapped_label'] = df_filtered['label'].map(LABEL_MAPPING)
        return df_filtered
    else: # Unlabeled data
        if 'label' in df.columns:
            print(f"Warning: Unlabeled data file '{file_path}' contains a 'label' column. It will be ignored.")
        return df

def train_model(model, tokenizer, train_df, eval_df, iteration=0):
    """Fine-tunes the BERT model."""
    print(f"\n--- Training Model for Iteration {iteration} ---")
    train_texts = train_df['text'].tolist()
    train_labels = train_df['mapped_label'].tolist()

    eval_texts = eval_df['text'].tolist()
    eval_labels = eval_df['mapped_label'].tolist()

    train_encodings = tokenizer(train_texts, padding="max_length", truncation=True, return_tensors="pt", max_length=128)
    eval_encodings = tokenizer(eval_texts, padding="max_length", truncation=True, return_tensors="pt", max_length=128)

    train_dataset = SentimentDataset(train_encodings, train_labels)
    eval_dataset = SentimentDataset(eval_encodings, eval_labels)

    # Clean up previous results to avoid conflicts if output_dir is reused
    current_output_dir = os.path.join(OUTPUT_DIR, f"iter_{iteration}")
    if os.path.exists(current_output_dir):
        shutil.rmtree(current_output_dir)
    os.makedirs(current_output_dir, exist_ok=True)


    training_args = TrainingArguments(
        output_dir=current_output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(current_output_dir, 'logs'),
        logging_steps=10,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    # Save the trained model for this iteration
    current_model_path = os.path.join(MODEL_SAVE_PATH, f"iter_{iteration}")
    os.makedirs(current_model_path, exist_ok=True)
    model.save_pretrained(current_model_path)
    tokenizer.save_pretrained(current_model_path)
    print(f"Model for iteration {iteration} saved to {current_model_path}")
    return model

def get_uncertain_samples(model, tokenizer, unlabeled_df):
    """Predicts on unlabeled data and selects the most uncertain samples."""
    print("\n--- Identifying Uncertain Samples ---")
    if unlabeled_df.empty:
        print("No unlabeled data remaining.")
        return pd.DataFrame()

    unlabeled_texts = unlabeled_df['text'].tolist()
    unlabeled_encodings = tokenizer(unlabeled_texts, padding="max_length", truncation=True, return_tensors="pt", max_length=128)
    unlabeled_dataset = SentimentDataset(unlabeled_encodings)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE)

    probabilities = []
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for batch in unlabeled_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_probs = softmax(logits, dim=1)
            probabilities.extend(batch_probs.cpu().numpy())

    uncertainty_scores = [1 - max(p) for p in probabilities]

    unlabeled_df['uncertainty'] = uncertainty_scores
    unlabeled_df['predicted_label'] = [REVERSE_LABEL_MAPPING[torch.argmax(torch.tensor(p)).item()] for p in probabilities] # Add predicted label for context

    # Select the most uncertain samples
    most_uncertain_samples = unlabeled_df.sort_values(by='uncertainty', ascending=False).head(NUM_SAMPLES_TO_SELECT_PER_ITER)

    # Save selected samples for manual annotation via Streamlit app
    samples_to_annotate_df = most_uncertain_samples[['text']].copy()
    samples_to_annotate_df['label'] = '' # Placeholder for manual annotation
    samples_to_annotate_df.to_csv('samples_to_annotate.csv', index=True, index_label='original_unlabeled_index') # Save with original index
    print(f"Saved {len(samples_to_annotate_df)} most uncertain samples to 'samples_to_annotate.csv' for manual labeling.")
    print("These are the samples to prioritize for annotation:")
    print(samples_to_annotate_df)
    return most_uncertain_samples

def incorporate_new_labels(main_labeled_df, unlabeled_pool_df):
    """Reads newly labeled data, adds to main labeled set, removes from unlabeled pool."""
    print("\n--- Incorporating Newly Labeled Data ---")
    if not os.path.exists(NEWLY_LABELED_DATA_FILE):
        print(f"No new labeled data file found at '{NEWLY_LABELED_DATA_FILE}'. Skipping this step.")
        return main_labeled_df, unlabeled_pool_df

    newly_labeled_df = pd.read_csv(NEWLY_LABELED_DATA_FILE)
    if newly_labeled_df.empty:
        print("Newly labeled data file is empty. Skipping this step.")
        return main_labeled_df, unlabeled_pool_df

    # Basic validation for newly labeled data
    if 'text' not in newly_labeled_df.columns or 'label' not in newly_labeled_df.columns:
        print(f"Error: '{NEWLY_LABELED_DATA_FILE}' must contain 'text' and 'label' columns.")
        return main_labeled_df, unlabeled_pool_df

    newly_labeled_df['text'] = newly_labeled_df['text'].astype(str).str.lower()
    newly_labeled_df['label'] = newly_labeled_df['label'].astype(int)

    # Filter and map new labels
    valid_labels = list(LABEL_MAPPING.keys())
    newly_labeled_df_filtered = newly_labeled_df[newly_labeled_df['label'].isin(valid_labels)].copy()
    if newly_labeled_df_filtered.empty:
        print(f"No valid labels ({valid_labels}) found in '{NEWLY_LABELED_DATA_FILE}'. Skipping.")
        return main_labeled_df, unlabeled_pool_df

    newly_labeled_df_filtered['mapped_label'] = newly_labeled_df_filtered['label'].map(LABEL_MAPPING)

    # Identify samples to remove from unlabeled pool using text content
    texts_to_remove = newly_labeled_df_filtered['text'].tolist()
    unlabeled_pool_df = unlabeled_pool_df[~unlabeled_pool_df['text'].isin(texts_to_remove)].copy()

    # Append to main labeled dataset
    # Ensure columns match for concatenation
    cols_to_keep = ['text', 'label', 'mapped_label']
    newly_labeled_for_append = newly_labeled_df_filtered[cols_to_keep]
    updated_main_labeled_df = pd.concat([main_labeled_df, newly_labeled_for_append], ignore_index=True)

    print(f"Added {len(newly_labeled_df_filtered)} new labeled samples.")
    print(f"Remaining unlabeled samples: {len(unlabeled_pool_df)}")

    # Clear the newly_labeled_data.csv for the next iteration
    pd.DataFrame(columns=['text', 'label']).to_csv(NEWLY_LABELED_DATA_FILE, index=False)
    print(f"Cleared '{NEWLY_LABELED_DATA_FILE}' for next iteration.")

    return updated_main_labeled_df, unlabeled_pool_df

# --- Main Active Learning Loop ---
def run_active_learning():
    print("--- Starting Full Active Learning Loop ---")

    # Initialize tokenizer and model (from scratch or last saved checkpoint)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load initial labeled and unlabeled data
    try:
        current_labeled_df = load_and_preprocess_data(INITIAL_LABELED_DATA_FILE, is_labeled=True)
        initial_unlabeled_df = load_and_preprocess_data(UNLABELED_POOL_FILE, is_labeled=False)

        # Remove any initial labeled data from the unlabeled pool to avoid overlap
        initial_unlabeled_df = initial_unlabeled_df[~initial_unlabeled_df['text'].isin(current_labeled_df['text'])].copy()
        current_unlabeled_df = initial_unlabeled_df
        print(f"Initial labeled samples: {len(current_labeled_df)}")
        print(f"Initial unlabeled pool samples: {len(current_unlabeled_df)}")

    except Exception as e:
        print(f"Error initializing data: {e}")
        print("Please ensure your 'dataset.csv' and 'unlabeled_data.csv' are correctly formatted and exist.")
        return

    # Split initial labeled data for train/test (test set remains constant for evaluation)
    train_df, test_df = train_test_split(
        current_labeled_df,
        test_size=0.2,
        random_state=42,
        stratify=current_labeled_df['mapped_label'].tolist()
    )
    print(f"Initial training set size: {len(train_df)}")
    print(f"Fixed test set size: {len(test_df)}")


    # --- Active Learning Loop ---
    for i in range(MAX_ACTIVE_LEARNING_ITERATIONS):
        print(f"\n======== Active Learning Iteration {i+1}/{MAX_ACTIVE_LEARNING_ITERATIONS} ========")

        # Step 1: Train/Retrain the model
        # Use the accumulated labeled data for training
        print(f"Total labeled data for training: {len(train_df)} samples.")
        model = train_model(model, tokenizer, train_df, test_df, iteration=i)

        # Step 2: Select uncertain samples
        if current_unlabeled_df.empty:
            print("Unlabeled pool is empty. Stopping active learning.")
            break
        
        # Get uncertain samples from the *remaining* unlabeled pool
        selected_for_annotation_df = get_uncertain_samples(model, tokenizer, current_unlabeled_df)

        if selected_for_annotation_df.empty:
            print("No uncertain samples found to select. Stopping active learning.")
            break

        # --- Manual Annotation Step (simulated for this script) ---
        print("\n*** MANUAL ANNOTATION REQUIRED ***")
        print(f"Please run the 'annotate_app.py' Streamlit app now.")
        print(f"Label the {len(selected_for_annotation_df)} samples saved in 'samples_to_annotate.csv'.")
        print(f"Once finished, ensure 'newly_labeled_data.csv' contains your new annotations.")
        input("Press Enter after you have finished labeling these samples and 'newly_labeled_data.csv' is ready...")

        # Step 3: Incorporate newly labeled data
        current_labeled_df_updated, current_unlabeled_df_updated = incorporate_new_labels(current_labeled_df, current_unlabeled_df)

        # Update dataframes for next iteration
        current_labeled_df = current_labeled_df_updated
        current_unlabeled_df = current_unlabeled_df_updated

        # Re-split training and testing set with the updated labeled data
        # Note: The 'test_df' here remains the initial test_df as specified in the project
        # In a real-world scenario, you might have a separate held-out test set.
        # For simplicity and to reflect project steps, train_df grows.
        train_df = current_labeled_df # For the next iteration, all current_labeled_df is considered training data

        if current_unlabeled_df.empty:
            print("\nUnlabeled pool exhausted. Active learning loop finished.")
            break

    print("\n--- Active Learning Process Completed ---")
    print(f"Final total labeled samples: {len(current_labeled_df)}")
    # Final evaluation step would typically go here
    # (e.g., call a separate evaluation function using the final model and test_df)

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Ensure newly_labeled_data.csv exists with headers
    if not os.path.exists(NEWLY_LABELED_DATA_FILE):
        pd.DataFrame(columns=['text', 'label']).to_csv(NEWLY_LABELED_DATA_FILE, index=False)
        print(f"Created empty '{NEWLY_LABELED_DATA_FILE}' for annotations.")

    run_active_learning()