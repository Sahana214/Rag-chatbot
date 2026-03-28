import os
import streamlit as st
from dotenv import load_dotenv
import shutil
import json
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings

# ------------------ Load API key ------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ------------------ Gemini Setup ------------------
client = genai.Client(api_key=GEMINI_API_KEY)

# ------------------ Streamlit UI ------------------
st.title("📄 RAG PDF Chatbot with Gemini AI")

# ------------------ Upload PDFs ------------------
uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    # ------------------ Clear old Chroma DB ------------------
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db", ignore_errors=True)

    # Reset chat history
    st.session_state.chat_history = []

    # Load PDFs
    docs = []
    for file in uploaded_files:
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(temp_path)
        loaded_docs = loader.load()

        # store file name for source reference
        for d in loaded_docs:
            d.metadata["source_file"] = file.name

        docs += loaded_docs

    # ------------------ Split text ------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)

    # Add metadata
    for i, doc in enumerate(chunks):
        doc.metadata["source"] = doc.metadata.get("source_file", "Unknown")

    st.success(f"✅ PDF(s) loaded and split into {len(chunks)} chunks")

    # ------------------ Create vector DB ------------------
    embeddings = FakeEmbeddings(size=768)
    db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    db.persist()

# ------------------ Chat memory ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask a question about the PDF(s):")

if query and uploaded_files:
    # Retrieve relevant chunks
    results = db.similarity_search(query, k=3)

    context = "\n\n".join(
        f"{r.page_content}\n(Source: {r.metadata.get('source')})"
        for r in results
    )

    # Include chat history
    history_context = "\n".join(
        f"Q: {c['question']}\nA: {c['answer']}"
        for c in st.session_state.chat_history[-3:]
    )

    prompt = f"""
Answer ONLY using the context below. If the answer is not present, say 'I don't know.'

Conversation history:
{history_context}

Context from PDF(s):
{context}

Question:
{query}
"""

    # ------------------ Gemini response ------------------
    with st.spinner("Generating answer..."):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            answer = response.text
        except Exception as e:
            st.error(f"Error: {e}")
            answer = "Error generating answer."

    # Save chat
    st.session_state.chat_history.append({
        "question": query,
        "answer": answer
    })

# ------------------ Display chat ------------------
for chat in st.session_state.chat_history:
    st.markdown(f"**Q:** {chat['question']}")
    st.markdown(f"**A:** {chat['answer']}")

# ------------------ Download chat ------------------
if st.session_state.chat_history:
    st.download_button(
        "💾 Download Chat History",
        data=json.dumps(st.session_state.chat_history, indent=2),
        file_name="chat_history.json"
    )