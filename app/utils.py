import pandas as pd
import faiss
import numpy as np

def embed_csv(df):
    """
    Takes a pandas DataFrame and generates a basic embedding for the CSV data.
    In this example, we use the 'App Name' column to simulate embeddings.
    You can replace this with a more complex method (e.g., OpenAI embeddings).
    
    Args:
    - df (pandas.DataFrame): The CSV data loaded into a DataFrame.
    
    Returns:
    - index (faiss.Index): FAISS index created for the CSV data.
    - rows (list): List of rows (in this case, App Names).
    """
    # Example: Create simple embeddings based on the "App Name" column
    # In a real scenario, you'd use a more sophisticated embedding model
    app_names = df['App Name'].tolist()

    # Simulate simple embeddings (this is a dummy example using ASCII values for each character in App Name)
    embeddings = []
    for app in app_names:
        embeddings.append([sum(ord(c) for c in app)])  # Convert each app name to a sum of ASCII values

    # Convert to a numpy array (FAISS requires numpy arrays)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Initialize FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for the index

    # Add embeddings to FAISS index
    index.add(embeddings)

    # Return both the FAISS index and rows (for query purposes)
    return index, app_names
