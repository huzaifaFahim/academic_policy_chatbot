import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq

# Set Streamlit config
st.set_page_config(page_title="📘 Academic Policy Chatbot", layout="centered")

# Load environment variables
load_dotenv()
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Load and prepare documents
@st.cache_resource
def load_vectorstore():
    loader = PyPDFLoader("academic_policy.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

# Load vector DB
vectorstore = load_vectorstore()

# UI
st.title("📘 Academic Policy Chatbot (RAG)")
st.write("Ask me anything from the Academic Policy Manual.")

query = st.text_input("❓ Ask a question:")

if query:
    with st.spinner("🔍 Searching academic policy..."):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = (
            f"You are an expert academic assistant. Use the following academic policy document excerpts to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            f"Answer:"
        )

        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",  # You can also use llama3-8b-8192 for faster results
            messages=[
                {"role": "system", "content": "You are a helpful assistant trained to answer questions based only on the given academic policy context."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content.strip()

    st.subheader("📌 Answer:")
    st.write(answer)
