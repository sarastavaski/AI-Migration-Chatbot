import json
import numpy as np
import faiss
import os
import re
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai
from openai import AzureOpenAI 

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Azure OpenAI API client
api_key = os.getenv("AZURE_OPENAI_KEY")
deployment_name = os.getenv("DEPLOYMENT_NAME")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Create a new client object
client = AzureOpenAI(  
    api_key=api_key,  
    api_version="2024-02-01",  
    azure_endpoint=azure_endpoint,  
)  

# Global variables to store index and mapping
index = None
mapping = None
structured_mapping = []

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def parse_mapping(mapping_list):
    parsed = []
    for entry in mapping_list:
        fields = {}
        for part in entry.split(", "):
            if ": " in part:
                key, value = part.split(": ", 1)
                fields[key.strip()] = value.strip()
        parsed.append(fields)
    return parsed

def load_existing_data():
    global index, mapping, structured_mapping
    index = faiss.read_index('/Users/sara.stavaski/Library/CloudStorage/OneDrive-Slalom/AppDev/AI-Final-Project/data/migration_index.faiss')
    with open('/Users/sara.stavaski/Library/CloudStorage/OneDrive-Slalom/AppDev/AI-Final-Project/data/migration_mapping.json', 'r') as f:
        mapping = json.load(f)
    structured_mapping = parse_mapping(mapping)

def load_default_data():
    global index, mapping
    # Load the FAISS index and mapping data
    load_existing_data()

# Load data at the start
with app.app_context():
    load_default_data()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Process the uploaded CSV
        # Process CSV and generate embeddings
        embed_csv(filepath)

        return jsonify({"message": "File uploaded and processed successfully!"})

def embed_csv(csv_file_path):
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    import faiss
    import json

    # Load CSV data
    df = pd.read_csv(csv_file_path)

    # Initialize sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Prepare data for embeddings
    rows = []
    for _, row in df.iterrows():
        content = f"App Name: {row['App Name']}, Server: {row['Server']}, DC Exit Strategy: {row['DC Exit Strategy']}, " \
                  f"Migration Path: {row['Migration Path']}, DC Exit Status: {row['DC Exit Status']}, " \
                  f"Environment: {row['Environment']}, Destination Account: {row['Destination Account']}"
        rows.append(content)

    # Generate embeddings for the CSV data
    embeddings = model.encode(rows, convert_to_tensor=False)

    # Convert embeddings to numpy array for FAISS
    embeddings = np.array(embeddings, dtype=np.float32)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, 'data/migration_index.faiss')

    # Save the mappings (for later reference in routes)
    mapping = {i: row.to_dict() for i, row in df.iterrows()}
    with open('data/migration_mapping.json', 'w') as f:
        json.dump(mapping, f)

@app.route("/ask", methods=["POST"])
def ask_question():
    question = request.json.get("question", "")
    
    # 🧠 Try structured logic first
   # structured_answer = handle_structured_question(question)
  #  if structured_answer:
   #     return jsonify({"answer": structured_answer})
    # 🤖 Fallback: use embeddings to find relevant rows and feed to Azure OpenAI
    context = get_relevant_context(question, index, mapping)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions about application migrations from data center exit planning. "
                "Use the following migration context to answer the user's question. "
                "Each line is a summary of a server's migration data. "
                "Only use the information provided — do not make up answers or assume values not included in the context."
            )
        },
        {
            "role": "user",
            "content": f"Migration Context:\n{context}\n\nQuestion: {question}"
        }
    ]


    # Call Azure OpenAI API to get the response
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages
    )

    answer = response.choices[0].message.content.strip()
    return jsonify({"answer": answer})

def get_relevant_context(question, index, mapping, top_k=5):
    question_embedding = model.encode([question], convert_to_tensor=True).cpu().numpy()
    D, I = index.search(question_embedding, top_k)

    results = [mapping[int(i)] for i in I[0] if int(i) < len(mapping)]
    return "\n".join(results)

def parse_mapping(mapping_list):
    parsed = []
    for entry in mapping_list:
        fields = {}
        for part in entry.split(", "):
            if ": " in part:
                key, value = part.split(": ", 1)
                fields[key.strip()] = value.strip()
        parsed.append(fields)
    return parsed


def get_applications():
    """
    Returns the list of unique applications from the structured mapping.
    """
    applications = set()  # Using a set to ensure uniqueness
    for entry in structured_mapping:
        applications.add(entry["App Name"])  # Assuming your structure has "App Name" field
    return applications

def get_servers():
    """
    Returns the list of unique servers from the structured mapping.
    """
    servers = set()
    for entry in structured_mapping:
        servers.add(entry["Server"])
    return servers

def get_environments():
    """
    Returns the list of unique environments from the structured mapping.
    """
    environments = set()
    for entry in structured_mapping:
        environments.add(entry["Environment"])
    return environments

def get_statuses():
    """
    Returns the list of unique statuses from the structured mapping.
    """
    statuses = set()
    for entry in structured_mapping:
        statuses.add(entry["DC Exit Status"])
    return statuses

def handle_structured_question(question):
    global structured_mapping
    response = ""

    # Check if the question asks about application names
    if "application" in question.lower():
        app_names = [entry['App Name'] for entry in structured_mapping]
        response = "The applications listed in the migration context are:\n" + "\n".join(app_names)

    # Check if the question asks about server names
    elif "server" in question.lower():
        #servers = [entry['Server'] for entry in structured_mapping]
        servers = get_servers()
        print(servers)
        response = "The servers listed in the migration context are:\n" + "\n".join(servers)

    # Check if the question asks about environment names
    elif "environment" in question.lower():
        environments = [entry['Environment'] for entry in structured_mapping]
        response = "The environments listed in the migration context are:\n" + "\n".join(environments)

    # Add more checks for other fields like "DC Exit Status", "Migration Path", etc.
    print(response)
    return response




if __name__ == '__main__':
    app.run(debug=True)


