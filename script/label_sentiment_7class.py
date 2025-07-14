import pandas as pd
from transformers import pipeline

# Load Excel file from input_files
df = pd.read_excel("input_files/finalDataset0.2.xlsx", engine="openpyxl")

# Initialize model for sentiment from text
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Feedback text columns
text_columns = ["teaching.1", "coursecontent.1", "Examination",
                "labwork.1", "library_facilities", "extracurricular.1"]

# Original numeric label columns (3-class)
label_columns = ["teaching", "coursecontent", "examination",
                 "labwork", "library_facilities", "extracurricular"]

# Map stars (from model) to 7-class
def map_star_to_7class(star):
    return {
        1: -3,
        2: -2,
        3: 0,
        4: 2,
        5: 3
    }.get(star, 0)

# Map original 3-class labels to 7-class
def map_3class_to_7class(val):
    return {
        -1: -2,
        0: 0,
        1: 2
    }.get(val, 0)

# 1️⃣ Create model-based 7-class labels from text
for col in text_columns:
    print(f"Model labeling: {col}")
    results = classifier(df[col].astype(str).tolist(), batch_size=8, truncation=True)
    df[col + "_label_7class_model"] = [map_star_to_7class(int(r['label'][0])) for r in results]

# 2️⃣ Create mapped 7-class labels from original 3-class numeric values
for col in label_columns:
    df[col + "_label_7class_mapped"] = df[col].apply(map_3class_to_7class)

# Save result
output_path = "input_files/finalDataset0.2_labeled_combined.xlsx"
df.to_excel(output_path, index=False)
print(f"✅ Final labeled file saved: {output_path}")
