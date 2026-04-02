import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Step 1: Load raw PDF(s)
def load_pdf_files(data):
    if not os.path.exists(data):
        raise FileNotFoundError(f"Data directory '{data}' does not exist. Create it and add PDFs.")
    
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        raise ValueError(f"No PDF files found in '{data}'. Add at least one PDF.")
    
    print(f"✅ Loaded {len(documents)} page(s) from PDFs.")
    return documents

# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"✅ Created {len(text_chunks)} text chunks.")
    return text_chunks

# Step 3: Create Embedding Model
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model

# Step 4: Store embeddings in FAISS
def build_vectorstore(text_chunks, embedding_model):
    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Vectorstore saved to '{DB_FAISS_PATH}'.")

if __name__ == "__main__":
    print("📄 Loading PDFs...")
    documents = load_pdf_files(DATA_PATH)

    print("✂️  Splitting into chunks...")
    text_chunks = create_chunks(documents)

    print("🔢 Loading embedding model...")
    embedding_model = get_embedding_model()

    print("💾 Building and saving FAISS vectorstore...")
    build_vectorstore(text_chunks, embedding_model)

    print("\n🎉 Done! You can now run the app with: streamlit run app.py")