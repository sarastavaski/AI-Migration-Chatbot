import os
import json
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = "/Users/sara.stavaski/Library/CloudStorage/OneDrive-Slalom/AppDev/AI-Final-Project/data/migration_data.csv"
INDEX_PATH = "/Users/sara.stavaski/Library/CloudStorage/OneDrive-Slalom/AppDev/AI-Final-Project/data/migration_index.faiss"
MAPPING_PATH = "/Users/sara.stavaski/Library/CloudStorage/OneDrive-Slalom/AppDev/AI-Final-Project/data/migration_mapping.json"
MODEL_NAME = "all-MiniLM-L6-v2"

def embed_csv():
    df = pd.read_csv(DATA_PATH)
    # Strip BOM and trim whitespace in column names
    df.columns = [col.replace('\ufeff', '').strip() for col in df.columns]

    rows = []
    for _, row in df.iterrows():
        content = (
            f"App Name: {row['App Name']}, Server: {row['Server']}, "
            f"DC Exit Strategy: {row['DC Exit Strategy']}, Migration Path: {row['Migration Path']}, "
            f"DC Exit Status: {row['DC Exit Status']}, Environment: {row['Environment']}, "
            f"Destination Account: {row['Destination Account']}"
        )
        rows.append(content)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(rows, convert_to_tensor=True)

    # Save index and mapping
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    print("Shape of embeddings:", embeddings.shape)
    index.add(embeddings.cpu().numpy())

    faiss.write_index(index, INDEX_PATH)

    with open(MAPPING_PATH, "w") as f:
        json.dump(rows, f)

    print("Embeddings and index saved.")

def load_index_and_mapping():
    index = faiss.read_index(INDEX_PATH)
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    return index, mapping

def query_embeddings(query, top_k=5):
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query], convert_to_tensor=False)

    index, mapping = load_index_and_mapping()
    distances, indices = index.search(query_embedding, top_k)
    print(f"Top matches (indices): {indices}")
    print(f"Distances (D): {distances}")

    results = [mapping[i] for i in indices[0]]
    return results


# Test the embedding process
if __name__ == "__main__":
    embed_csv()


