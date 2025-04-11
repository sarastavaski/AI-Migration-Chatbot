# ☁️ Migration Chatbot with Azure OpenAI + Flask

A simple chatbot that answers questions about your application migration strategy using Azure OpenAI, CSV upload, and semantic search with embeddings.

---

## 📁 File Upload Format
Upload a CSV file with the following columns:

```
App Name,Server,DC Exit Strategy,Migration Path,DC Exit Status,Environment,Destination Account
```

Example:
```csv
App Name,Server,DC Exit Strategy,Migration Path,DC Exit Status,Environment,Destination Account
InventoryApp,inv01,Retire,Lift and Shift,Completed,Production,aws-prod-123
HRApp,hr01,Rehost,Refactor,In Progress,Staging,aws-staging-456
```

---

## 🚀 How to Run

1. **Clone the repo**
```bash
git clone https://github.com/your-username/migration-chatbot.git
cd migration-chatbot
```

2. **Set up environment variables**
Create a `.env` file:
```
AZURE_OPENAI_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
DEPLOYMENT_NAME=your-gpt-deployment
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the app**
```bash
python run.py
```

Visit: `http://localhost:5000`

---

## 🧠 How it Works
- Embeds your CSV rows using Azure OpenAI embeddings
- Uses FAISS for similarity search
- Sends top matches with your question to GPT for a smart answer

---

## ✨ Features
- Upload your migration planning CSV
- Ask natural language questions
- Cloud-themed chatbot UI