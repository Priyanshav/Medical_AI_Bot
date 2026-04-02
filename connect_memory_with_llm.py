import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
You are a medical assistant. Use ONLY the context below to answer in detail.
Explain thoroughly and include all relevant information from the context.
If the answer is not in the context, say "I don't have that information in my documents."
Do NOT use any outside knowledge.

Context: {context}
Question: {question}

Detailed Answer:"""

# Step 1: Load LLM
def load_llm():
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Check your .env file.")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=1024,
        api_key=GROQ_API_KEY
    )

# Step 2: Build prompt
def set_custom_prompt(template):
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

# Step 3: Load vectorstore
def load_vectorstore():
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(
            f"Vectorstore not found at '{DB_FAISS_PATH}'. "
            "Run create_memory_for_llm.py first."
        )
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Step 4: Build QA chain
def build_qa_chain():
    db = load_vectorstore()
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 5}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    return qa_chain

if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not set. Check your .env file.")
        exit(1)

    print("🔗 Loading vectorstore and LLM...")
    qa_chain = build_qa_chain()

    print("✅ Ready! Type 'exit' to quit.\n")

    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not user_query:
            continue

        response = qa_chain.invoke({'query': user_query})
        print(f"\n🤖 Answer: {response['result']}")
        print("\n📚 Sources:")
        for i, doc in enumerate(response["source_documents"], 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            print(f"  [{i}] {source} — page {page}")
        print()