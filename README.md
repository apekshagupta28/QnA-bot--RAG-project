# QnA-bot--RAG-project

# 📄 Document Q&A Bot — RAG Pipeline

A web app that lets you upload any PDF and ask questions about it in plain English. Powered by a full RAG (Retrieval-Augmented Generation) pipeline — no reading the whole document required.

---

## 🚀 Demo

Upload a PDF → Ask a question → Get a grounded, accurate answer in seconds.

---

## 🧠 How It Works

| Step | What Happens |
|------|-------------|
| 1. Upload | User uploads a PDF via the Streamlit UI |
| 2. Extract | PyPDF2 reads every page and extracts raw text |
| 3. Chunk | LangChain splits text into 500-char chunks (50-char overlap) |
| 4. Embed | HuggingFace `all-MiniLM-L6-v2` converts each chunk into a vector |
| 5. Index | FAISS stores all vectors in an in-memory similarity index |
| 6. Retrieve | User's question is embedded and top 3 matching chunks are fetched |
| 7. Generate | Groq API (LLaMA 3.3 70B) reads only the retrieved chunks and answers |
| 8. Display | Answer appears in the Streamlit UI |

---

## 🛠️ Tech Stack

- **Python** — core language
- **Streamlit** — browser-based UI
- **PyPDF2** — PDF text extraction
- **LangChain** — pipeline orchestration (chunking, retrieval)
- **HuggingFace `all-MiniLM-L6-v2`** — free, open-source embedding model
- **FAISS** — vector similarity search (Meta)
- **Groq API (`llama-3.3-70b-versatile`)** — LLM for answer generation

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/document-qa-bot.git
cd document-qa-bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API key
Create a `.env` file in the root folder:
```
GROQ_API_KEY=your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
document-qa-bot/
│
├── app.py              # Full pipeline + Streamlit UI
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

---

## 💡 Key Concepts

- **RAG Architecture** — grounds the LLM in your document, preventing hallucinations
- **Vector Embeddings** — semantic search beyond simple keyword matching
- **FAISS** — fast in-memory similarity search without a cloud vector DB
- **Hallucination Prevention** — LLM only sees the top 3 retrieved chunks, nothing else

---

## 🆓 Completely Free to Run

- Groq API — free tier
- HuggingFace embedding model — open source, runs locally
- No OpenAI or paid APIs required
