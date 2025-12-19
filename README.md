# 📊 RAG-Based Financial Document Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)]([[share.streamlit.io](https://rag-based-financial-document-assistant-neae5oyytdnjyrx7casacl.streamlit.app/)])
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/Powered%20by-LangChain-green)](https://python.langchain.com/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Inference-HuggingFace-orange)](https://huggingface.co/)

## 🌟 Executive Summary
This project is an Advanced RAG (Retrieval-Augmented Generation) System designed to automate the extraction of high-signal insights from complex financial documents. While standard Large Language Models (LLMs) often suffer from "hallucinations" and lack access to specific, private data, this system bridges that gap by grounding every response in verifiable source material.
### 🎯 The Problem
Financial analysts often spend hours manually cross-referencing multi-page reports (10-Ks, 10-Qs, or audit summaries). Standard AI tools cannot handle these documents accurately because:
They lack access to the specific, non-public context of the uploaded file.
They have a limited "context window," meaning they "forget" earlier parts of a long report.
### 🛠️ The Solution
This application implements a state-of-the-art RAG pipeline that converts unstructured PDF data into a searchable vector database. By utilizing Semantic Search, the system retrieves only the most relevant financial data points and passes them to the LLM (Zephyr-7B) for synthesis.
### ⚡ Technical Achievement: Resource-Optimized Architecture
A key feature of this implementation is its Cloud-Native Design. I engineered the system to offload heavy model inference to Serverless GPU Endpoints via the Hugging Face Inference API. This ensures that the application remains high-performing and scalable without requiring high-end local hardware, making it a cost-effective solution for real-world enterprise deployments.
---

## 🚀 Live Demo
**Check out the live application here:** [👉 [[link](https://rag-based-financial-document-assistant-neae5oyytdnjyrx7casacl.streamlit.app/)]]

---

## 🛠️ System Architecture
The system follows a modern **Agentic workflow** using LangChain Expression Language (LCEL):

1.  **Ingestion:** PDF parsing using `PyPDFLoader`.
2.  **Chunking:** `RecursiveCharacterTextSplitter` optimized for financial tables.
3.  **Embedding:** Text vectorized via `all-MiniLM-L6-v2` (sentence-transformers).
4.  **Vector Store:** High-speed similarity search using **FAISS**.
5.  **Reasoning:** Instruction-following via **Zephyr-7B-Beta** (7-billion parameter model).

---

## 🧰 Tech Stack
| Category | Technology |
| :--- | :--- |
| **LLM Orchestration** | LangChain (LCEL) |
| **Generative Model** | Zephyr-7B-Beta (Mistral-based) |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **Frontend** | Streamlit |
| **Embeddings** | Hugging Face Inference API |
| **Deployment** | Streamlit Cloud |

---

## 🧠 Key Features & Portfolio Highlights
*   **LCEL Implementation:** Uses the latest LangChain Expression Language for faster, more transparent pipeline execution compared to legacy `RetrievalQA`.
*   **Memory Management:** Implemented `st.session_state` to prevent redundant document processing, optimizing API costs and speed.
*   **Context-Aware Prompting:** Engineered custom system prompts to ensure the model remains in a "Financial Analyst" persona and cites lack of information rather than hallucinating.
*   **Zero-Footprint Inference:** Demonstrates proficiency in using Serverless API endpoints for scalable AI deployment.

---

## 📦 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/[YOUR_USERNAME]/RAG-Financial-Assistant.git
   cd RAG-Financial-Assistant
   ```
2. **Install Dependencies**
```Bash
pip install -r requirements.txt
```
3. **Get your API Key**
Create a free account at Hugging Face.
Generate a 'Read' token in Settings.
4. **Run the App**
```Bash
streamlit run app.py
```

📈 Future Roadmap

Add support for multiple PDF uploads (Knowledge Graph).

Implement Table Extraction using Unstructured or Camelot.

Integrate Chat History for multi-turn financial consultations.
