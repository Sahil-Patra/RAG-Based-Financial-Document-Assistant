
# 📊 RAG-Based Financial Document Assistant

![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/Powered%20by-LangChain%20LCEL-green)](https://python.langchain.com/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Inference-Zephyr--7B-orange)](https://huggingface.co/)

## 🌟 Executive Summary
This project is an **Advanced Retrieval-Augmented Generation (RAG) System** designed to automate the extraction of high-signal financial insights from complex documents (e.g., 10-K filings, earnings call transcripts, and audit reports). Standard Large Language Models (LLMs) often hallucinate or struggle with long financial reports due to strict context windows. This application bridges that gap by grounding every LLM response directly in verifiable, page-indexed source context.

### 🎯 The Problem
Financial analysts spend hours cross-referencing dense financial PDFs. Off-the-shelf generative AI models fall short because:
1. They lack access to private, unindexed document data.
2. They suffer from context window truncation and loss of precision on numerical tables.
3. They generate unverified answers without source citations or page references.

### 🛠️ The Solution
This application converts unstructured PDF documents into high-density vector representations stored in a **FAISS** index. Using **LangChain Expression Language (LCEL)**, relevant text chunks are dynamically retrieved and passed to **Zephyr-7B-Beta**. The system streams responses token-by-token and provides explicit **page-level source attribution** for financial auditability.

### ⚡ Technical Achievement: Performance-Optimized Hybrid Architecture
To maximize performance without incurring API rate limits or latency penalties:
* **Local In-Memory Embeddings:** Document vectorization runs locally using Sentence-Transformers, completely eliminating remote network overhead during indexing.
* **Serverless LLM Inference:** Generative synthesis is offloaded to serverless GPU endpoints via Hugging Face Cloud.
* **Streamlit Resource Caching:** Pipeline clients and model instances are cached using `@st.cache_resource`, ensuring persistent execution across user UI interactions.

---

## 🚀 Live Demo
**Check out the live application here:** [👉 RAG Financial Document Assistant](https://rag-based-financial-document-assistant-neae5oyytdnjyrx7casacl.streamlit.app/)

---

## 🛠️ System Architecture
The system follows a modern **RAG workflow** engineered with LangChain Expression Language (LCEL):

```
[Uploaded PDF] ➔ [PyPDFLoader] ➔ [Recursive Splitter] ➔ [Local MiniLM Embeddings] ➔ [FAISS Vector Store]
                                                                                            │
[User Query] ───────────────────────────────────────────────────────────────────────────────┴➔ [Similarity Search (k=3)]
                                                                                                        │
[Token Stream Rendered in UI]  [Zephyr-7B LLM]  [Financial Persona System Prompt + Context] ──────────┘
```

1. **Ingestion & Safety:** PDF processing with deterministic `try...finally` temporary file lifecycle management.
2. **Chunking:** `RecursiveCharacterTextSplitter` tuned to 600-character chunks with 50-character overlap for precision table retention.
3. **Local Vectorization:** Fast in-memory vector indexing via `all-MiniLM-L6-v2`.
4. **Retrieval & Reranking:** Similarity search using **FAISS** with page-number metadata injection.
5. **Streaming Generation:** Token-by-token response rendering via `st.write_stream()`.

---

## 🧰 Tech Stack
| Category | Technology |
| :--- | :--- |
| **LLM Orchestration** | LangChain (LCEL) |
| **Generative LLM** | Zephyr-7B-Beta (HuggingFace Cloud Endpoint) |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (Local Execution) |
| **Frontend Framework** | Streamlit |
| **Document Parser** | PyPDFLoader |

---

## 🧠 Key Features & Portfolio Highlights
* **Token-by-Token Response Streaming:** Replaced blocking synchronous calls with `st.write_stream` and `chain.stream()`, delivering immediate visual feedback.
* **Page-Level Provenance & Citations:** Every retrieved chunk injects page metadata (`[Page X]`), allowing financial auditors to instantly trace figures back to source documents.
* **Optimized Execution Lifecycle:** Implemented `@st.cache_resource` decorators to decouple client instantiation from Streamlit's script rerun loop.
* **Local Vector Processing:** Eliminates HTTP rate-limiting bottlenecks during vector store generation by executing embedding math locally on CPU/GPU.
* **Strict Analyst System Prompting:** Engineered custom system prompts forcing the model to operate as a Financial Data Scientist and report missing data rather than hallucinating.

---

## 📦 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sahil_Patra/RAG-Based-Financial-Document-Assistant.git
   cd RAG-Financial-Assistant
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get your Hugging Face Token**
   - Create a free account at [Hugging Face](https://huggingface.co/).
   - Generate a `Read` token under **User Settings ➔ Access Tokens**.

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

---

## 📈 Future Roadmap
- [ ] **Multi-Document Knowledge Graphs:** Add support for cross-referencing multiple 10-K filings simultaneously.
- [ ] **Advanced Table Parsing:** Integrate `Unstructured` or `pdfplumber` for complex tabular data extraction.
- [ ] **Conversational Memory:** Implement `RunnableWithMessageHistory` to support multi-turn financial Q&A sessions.
- [ ] **Semantic Caching:** Integrate Redis/GPTCache to instantly serve answers for repeated financial queries.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to open an issue or submit a Pull Request.

## 📜 License
This project is open-source and available under the MIT License.
```
