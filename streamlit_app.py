import streamlit as st
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
from langchain_groq import ChatGroq
import os

# ---- Set up API keys ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Ensure this is set in your environment or replace with a string

# ---- Streamlit UI ----
st.title("📄 RAG Document QA with Groq LLaMA")
st.markdown("Upload documents and ask questions. Compare LLM answers **with vs. without** retrieval augmentation.")

# Upload files
uploaded_files = st.file_uploader("Upload PDFs, TXT or CSVs:", type=["pdf", "txt", "csv"], accept_multiple_files=True)

# Fine-tuning parameters
st.markdown("### 🔧 Fine-Tuning Parameters")
chunk_size = st.slider("Chunk size (characters):", 200, 2000, 500, 100, help="Size of each text chunk for retrieval. Smaller sizes may improve precision.")
top_k = st.slider("Top K retrieved chunks:", 1, 10, 3, help="Number of most relevant document chunks retrieved for answering.")
temperature = st.slider("LLM Temperature:", 0.0, 1.0, 0.0, 0.1, help="Controls randomness in LLM output. Lower is more deterministic.")
chunk_overlap = st.slider("Chunk Overlap (characters):", 0, 500, 100, 10, help="Number of characters to overlap between adjacent chunks.")
max_tokens = st.slider("Max Tokens for Response:", 50, 2048, 512, 50, help="Maximum length of the generated LLM response.")
embedding_model = st.selectbox("Embedding Model:", ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/paraphrase-MiniLM-L3-v2", "BAAI/bge-large-en"], help="Choose the embedding model for vector generation.")
prompt_instruction = st.text_area("Custom Prompt Instruction:", value="Use the following context to answer the question.", help="Customize how the LLM is instructed to use the context.")
show_sources = st.checkbox("Show Retrieved Context in Output", value=False, help="Toggle to include retrieved document context in the RAG tab.")
query = st.text_input("Enter your query:")

# Submit button
if st.button("Submit"):
    if not uploaded_files or not query:
        st.warning("Upload at least one file and enter a query.")
    else:
        # ---- Load and read files ----
        all_text = ""
        for file in uploaded_files:
            name = file.name.lower()
            if name.endswith(".pdf"):
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                temp.write(file.read())
                temp.flush()
                loader = PyPDFLoader(temp.name)
                pages = loader.load()
                file_text = "\n".join([page.page_content for page in pages])
            else:
                file_text = file.read().decode("utf-8", errors="ignore")
            all_text += "\n" + file_text

        # ---- Split text into chunks ----
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = splitter.create_documents([all_text])

        # ---- Embed and index with FAISS ----
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        db = FAISS.from_documents(documents, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": top_k})

        # ---- Retrieve context ----
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])

        # ---- Query Groq LLaMA model ----
        llm = ChatGroq(model="llama3-70b-8192", temperature=temperature, groq_api_key=GROQ_API_KEY, max_tokens=max_tokens)

        rag_prompt = (
            f"{prompt_instruction}\n"
            f"Context:\n{context}\n"
            f"\nQuestion: {query}\nAnswer:"
        )

        with st.spinner("Generating answers..."):
            rag_response = llm.invoke(rag_prompt).content
            direct_response = llm.invoke(query).content

        # ---- Show answers ----
        tab1, tab2 = st.tabs(["RAG Answer", "Direct LLM Answer"])

        with tab1:
            st.subheader("Answer using Document Context")
            st.write(rag_response.strip())
            if show_sources:
                st.markdown("---")
                st.markdown("**Retrieved Context:**")
                st.text(context)

        with tab2:
            st.subheader("Answer using Model Only (No Context)")
            st.write(direct_response.strip())
