# 📊 RAG-Based Financial Document Assistant

## 🚀 Overview
I developed this project to bridge the gap between traditional data analysis and Generative AI. It allows users to upload complex financial PDF documents and ask questions in plain English using **Retrieval-Augmented Generation (RAG)**.

**Note:** This project was specifically architected for resource-constrained environments (6GB RAM), utilizing Cloud-Native Inference to ensure high performance on low-end hardware.

## 🛠️ Tech Stack
- **Framework:** LangChain (LCEL)
- **Frontend:** Streamlit
- **LLM:** Zephyr-7B-Beta (via Hugging Face Inference API)
- **Vector Store:** FAISS
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Data Engineering:** PyPDFLoader, RecursiveCharacterTextSplitter

## 💡 Key Features
- **Cloud-Native:** Zero local GPU/RAM load for model inference.
- **Modern RAG:** Uses LangChain Expression Language (LCEL) for stable, production-ready pipelines.
- **Financial Intelligence:** Optimized chunking and prompts for financial report analysis.

## ⚙️ Setup Instructions
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/PROJECT_NAME.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Get a free API token from [Hugging Face](https://huggingface.co/settings/tokens).
4. Run the app: `streamlit run app.py`