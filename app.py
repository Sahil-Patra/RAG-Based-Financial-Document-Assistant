import streamlit as st
import os
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="RAG Financial Assistant", layout="wide")
st.title("📊 Modern RAG Financial Assistant")
st.markdown("*Stable Version: Powered by Zephyr-7B & HF Cloud*")

# --- Sidebar ---
with st.sidebar:
    st.header("🔑 Authentication")
    hf_token = st.text_input("Hugging Face Token", type="password")
    
    st.header("📄 Upload")
    uploaded_file = st.file_uploader("Upload Financial PDF", type="pdf")
    
    st.markdown("---")
    st.write("🤖 **Model:** Zephyr-7B-Beta")
    st.write("🟢 **Status:** High Availability Mode")

# --- Core Logic ---
@st.cache_resource
def load_embedding_model():
    """Runs embeddings locally in CPU/GPU memory, avoiding network latency and rate limits."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdf(pdf_file):
    """Safe temp file lifecycle with local FAISS vector store creation."""
    embeddings = load_embedding_model()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        temp_path = tmp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        return FAISS.from_documents(chunks, embeddings)
    finally:
        # Guarantees file cleanup even if parsing or chunking throws an Exception
        if os.path.exists(temp_path):
            os.remove(temp_path)

@st.cache_resource(show_spinner=False)
def initialize_llm(api_token: str):
    """Instantiate and cache the LLM client once per session."""
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=api_token,
        temperature=0.1,
        max_new_tokens=512,
        streaming=True # Enabled for real-time streaming
    )
    return ChatHuggingFace(llm=llm_endpoint)

def build_rag_chain(vectorstore, chat_model):
    """Build lightweight chain without re-instantiating heavy models."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional Financial Data Scientist. Answer the question accurately using ONLY the provided context."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[Page {doc.metadata.get('page', 'N/A')}]: {doc.page_content}" 
            for doc in docs
        )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

# --- App Execution ---
if hf_token and uploaded_file:
    # Handle PDF indexing
    if "vector_store" not in st.session_state:
        with st.spinner("🛠️ Indexing PDF Content..."):
            vs = process_pdf(uploaded_file, hf_token)
            if vs:
                st.session_state.vector_store = vs
                st.success("✅ Document Ready!")

    # Handle Question/Answering
    if "vector_store" in st.session_state:
        user_q = st.text_input("Ask a question about the report:")
        if user_q:
            chat_model = initialize_llm(hf_token)
            rag_chain = build_rag_chain(st.session_state.vector_store, chat_model)
    
            st.markdown("### 📝 Answer:")
            try:
                # Stream output token-by-token for responsive UX
                response_stream = rag_chain.stream(user_q)
                st.write_stream(response_stream)
        
                # Display source metadata in expandable view
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                source_docs = retriever.invoke(user_q)
                with st.expander("📌 View Source Citations"):
                    for idx, doc in enumerate(source_docs, 1):
                        page_num = doc.metadata.get("page", "Unknown")
                        st.caption(f"**Source {idx} (Page {page_num}):**")
                        st.text(doc.page_content[:300] + "...")
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
                st.info("The Cloud API might be overloaded. Try again in a few seconds.")

elif not hf_token:
    st.warning("👈 Please enter your Hugging Face Token in the sidebar.")