import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CUSTOM_PROMPT_TEMPLATE = """
You are a medical assistant. Use ONLY the context below to answer in detail.
Explain thoroughly and include all relevant information from the context.
If the answer is not in the context, say "I don't have that information in my documents."
Do NOT use any outside knowledge.

Context: {context}
Question: {question}

Detailed Answer:"""

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.5,
        max_tokens=1024,
        api_key=GROQ_API_KEY
    )

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def main():
    st.title("Ask Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store. Run create_memory_for_llm.py first.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=get_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]

            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()