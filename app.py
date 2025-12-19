import streamlit as st
import os
import tempfile
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings, ChatHuggingFace
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
def process_pdf(pdf_file, api_token):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            temp_path = tmp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Smaller chunks for better context window management
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            huggingfacehub_api_token=api_token
        )

        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        st.error(f"Processing Error: {str(e)}")
        return None

def get_rag_chain(vectorstore, api_token):
    # Using Zephyr-7B for better uptime and RAG performance
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=api_token,
        temperature=0.1,
        max_new_tokens=512,
    )
    chat_model = ChatHuggingFace(llm=llm_endpoint)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional Financial Data Scientist. Answer the question accurately using ONLY the provided context."),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

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
            with st.spinner("🤔 Zephyr is analyzing the data..."):
                try:
                    chain = get_rag_chain(st.session_state.vector_store, hf_token)
                    response = chain.invoke(user_q)
                    st.markdown("### 📝 Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Model Error: {str(e)}")
                    st.info("The Cloud API might be overloaded. Try again in a few seconds.")

elif not hf_token:
    st.warning("👈 Please enter your Hugging Face Token in the sidebar.")