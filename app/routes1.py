import os
import pandas as pd
import openai
from openai import AzureOpenAI 
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from app.utils import embed_csv
from app.embeddings import load_embeddings, query_embeddings

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Default CSV path
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "migration_data.csv")

# Global variables to store index and rows
index = None
rows = []

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
# 🔹 Load knowledge base once at startup
knowledge_base = load_embeddings("data/migration_data_embedded.json")

# Load default CSV data on startup
def load_default_data():
    global index, rows
    try:
        # Load CSV into a pandas DataFrame
        df = pd.read_csv(DEFAULT_CSV_PATH)
        # Embed CSV data (assuming embed_csv is a function that processes the data)
        index, rows = embed_csv(df)
        print("✅ Default migration data loaded from data/migration_data.csv")
    except Exception as e:
        print(f"⚠️ Could not load default CSV: {e}")

# Call load_default_data() when app starts
load_default_data()

# Home route that serves the index.html template
@app.route('/')
def index_page():
    return render_template('index.html')

# Route for file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        # Process the uploaded CSV
        df = pd.read_csv(file)
        global index, rows
        index, rows = embed_csv(df)  # Update index with new CSV data
        return jsonify({"message": "CSV file uploaded and processed successfully!"})
    return jsonify({"message": "Failed to upload CSV."})

# Route for asking questions based on embedded CSV data
@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 👇 Get the relevant migration context using embeddings
    context = query_embeddings(question)

    # 👇 Build the messages for the OpenAI chat completion
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
            "content": f"{context}\n\nQuestion: {question}"
        }
    ]

    # 👇 Use the updated messages with the context in your OpenAI call
    response = client.chat.completions.create(
        model=deployment_name,
        messages=messages
    )

    answer = response.choices[0].message.content.strip()
    return jsonify({"answer": answer})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
