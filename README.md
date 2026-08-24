# 🏥 Medical AI Chatbot

A Retrieval-Augmented Generation (RAG) based medical chatbot built with LangChain, FAISS, and Groq. It answers medical questions strictly based on **The Gale Encyclopedia of Medicine** — no hallucinations, no made-up answers.

---

## 🔍 How It Works

```
User Question
     ↓
FAISS Vector Search (finds relevant chunks from PDF)
     ↓
Retrieved Context sent to Groq LLM (GPT OSS 20B)
     ↓
Answer strictly based on the document
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq (OpenAI GPT OSS 20B) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| Framework | LangChain |
| Document | Gale Encyclopedia of Medicine |

---

## 📁 Project Structure

```
Medical_Chatbot/
├── data/                          # PDF documents
├── vectorstore/
│   └── db_faiss/
│       ├── index.faiss            # FAISS vector index
│       └── index.pkl              # FAISS metadata
├── app.py                         # Streamlit app
├── connect_memory_with_llm.py     # CLI version for testing
├── create_memory_for_llm.py       # Builds vectorstore from PDFs
├── requirements.txt               # Dependencies
├── .env.example                   # Environment variable template
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Priyanshav/Medical_AI_Bot.git
cd Medical_AI_Bot
```

### 2. Create and activate virtual environment
```bash
python -m venv medibot_env

# Windows
medibot_env\Scripts\activate

# Mac/Linux
source medibot_env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:
```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key
```

Get your free keys here:
- HuggingFace: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Groq: [console.groq.com](https://console.groq.com)

### 5. Build the vectorstore (only needed if adding new PDFs)
```bash
python create_memory_for_llm.py
```

### 6. Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 💬 Usage

- Type any medical question in the chat input
- The bot retrieves relevant information from the encyclopedia
- Answers are strictly based on the document — no outside knowledge
- If the answer isn't in the document, it will say so

---

## 🧪 Test via CLI (without Streamlit)

```bash
python connect_memory_with_llm.py
```

Type your query and press Enter. Type `exit` to quit.

---

## ☁️ Deployment

This app is deployed on **HuggingFace Spaces**:

👉 [Live Demo](https://huggingface.co/spaces/Priyanshav/medical-chatbot)

---

## ⚠️ Important Notes

- Never commit your `.env` file — it contains secret API keys
- The vectorstore is pre-built and included in the repo
- Re-run `create_memory_for_llm.py` only if you add new PDFs to `data/`

---

## 📄 License

This project is for educational purposes only. The medical information is sourced from The Gale Encyclopedia of Medicine and should not be used as a substitute for professional medical advice.
