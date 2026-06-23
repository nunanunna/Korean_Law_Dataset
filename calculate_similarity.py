from sentence_transformers import SentenceTransformer, util
import torch
import json
import os
import sys

# Reconfigure stdout to print Korean correctly on Windows
sys.stdout.reconfigure(encoding='utf-8')

# 1. Device configuration: use CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device configuration: using {device} for computation")

# 2. Load the SBERT model
print("Loading SBERT model...")
model_name = 'woong0322/ko-legal-sbert-finetuned'
model = SentenceTransformer(model_name, device=device)
print("Model loaded successfully!")

# 3. Load the dataset
dataset_path = os.path.join("test_dataset", "full_dataset.json")
if not os.path.exists(dataset_path):
    print(f"Error: Dataset not found at {dataset_path}")
    sys.exit(1)

with open(dataset_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

bills = dataset.get("bills", [])
num_bills = len(bills)
print(f"Loaded {num_bills} bills from full_dataset.json")

# 4. Prepare text inputs and run SBERT encoding
print("Extracting summaries and encoding embeddings...")
texts = []
for bill in bills:
    summary = bill.get("summary")
    # Fallback to bill_name if summary is empty/missing
    if not summary or not summary.strip():
        summary = bill.get("bill_name", "")
    texts.append(summary)

# Encode all texts in a single batch
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_tensor=True)
print(f"Generated embeddings shape: {embeddings.shape}")

# 5. Compute pairwise cosine similarities
print("Computing pairwise cosine similarities...")
similarity_matrix = util.cos_sim(embeddings, embeddings)

# 6. Compare categories and collect similarity results for all unique pairs
results = []
print("Processing pairwise bill comparisons and category matching...")
for i in range(num_bills):
    for j in range(i + 1, num_bills):
        bill1 = bills[i]
        bill2 = bills[j]
        
        # Calculate similarity score
        similarity_score = similarity_matrix[i][j].item()
        
        # Category comparison
        cats1 = bill1.get("categories", [])
        cats2 = bill2.get("categories", [])
        
        set1 = set(cats1)
        set2 = set(cats2)
        
        intersection = sorted(list(set1.intersection(set2)))
        union = sorted(list(set1.union(set2)))
        
        jaccard_sim = len(intersection) / len(union) if len(union) > 0 else 0.0
        
        results.append({
            "bill_id_1": bill1["bill_id"],
            "bill_name_1": bill1["bill_name"],
            "categories_1": cats1,
            "bill_id_2": bill2["bill_id"],
            "bill_name_2": bill2["bill_name"],
            "categories_2": cats2,
            "similarity": round(similarity_score, 6),
            "category_intersection": intersection,
            "category_intersection_count": len(intersection),
            "category_union_count": len(union),
            "jaccard_similarity": round(jaccard_sim, 6)
        })

# 7. Write results to output directory
output_dir = "Sbert_output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "similarity_results.json")

print(f"Writing {len(results)} pairwise similarity results to {output_path}...")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Calculation complete! Results stored successfully.")
