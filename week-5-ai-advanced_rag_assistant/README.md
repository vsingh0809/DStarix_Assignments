# Advanced RAG Knowledge Base Assistant

A production-ready, modular Retrieval-Augmented Generation (RAG) system built with **LangChain**, **Google Gemini**, and **ChromaDB**. Designed with software engineering best practices including separation of concerns, structured JSON outputs, hybrid search, and multi-stage reranking.

## 🏛️ Architecture Highlights

1. **Modular Architecture:** Divided into configuration, loader, retriever, chain, schema, and logging modules.
2. **Hybrid Search Strategy:** Merges Sparse Keyword Search (`BM25`) and Dense Vector Search (`Chroma`) through an `EnsembleRetriever`.
3. **Query Transformation:** Employs `MultiQueryRetriever` to generate query variations and expand document recall.
4. **Contextual Compression (Reranking):** Uses an LLM chain extractor to filter out context noise before prompt construction.
5. **Type Safety & Structured Output:** Returns strictly typed Pydantic objects containing the response, confidence rating, and source attribution boolean.

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- A Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/)

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
git clone <your-repo-url>
cd advanced_rag_assistant

# Create virtual environment
python -m venv venv

# Activate Virtual Environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt