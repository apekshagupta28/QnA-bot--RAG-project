import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# ---- Page Config ----
st.set_page_config(page_title="Document Q&A Bot", page_icon="🤖")
st.title("📄 Document Q&A Bot")
st.write("Upload a PDF and ask questions about it!")

# ---- API Key Input ----
api_key = st.text_input("Enter your Groq API Key", type="password")

# ---- PDF Upload ----
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and api_key:
    # Step 1: Extract text from PDF
    reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text()

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(raw_text)

    # Step 3: Create embeddings and store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    st.success("✅ PDF processed! Ask your question below.")

    # Step 4: Ask a question
    question = st.text_input("Ask a question about your document")

    if question:
        # Retrieve relevant chunks
        docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Send to Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": f"Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}"}
            ]
        )

        st.write("### Answer:")
        st.write(response.choices[0].message.content)