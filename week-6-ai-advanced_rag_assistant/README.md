# DStarix Production AI Assistant

A comprehensive, production-ready AI Assistant integrating Retrieval-Augmented Generation (RAG), dynamic Tool Calling, and Conversation Memory, packaged into a highly interactive Streamlit Web Interface.

## Features

- **Document Processing:** Upload `.pdf` or `.txt` files on the fly.
- **Dynamic RAG Tuning:** Adjust `chunk_size`, `chunk_overlap`, and `retrieval_count` directly from the UI to observe changes in answer quality.
- **Tool-Calling Agent:** The LLM autonomously decides whether to query the uploaded document (via vector search) or search the live web (via DuckDuckGo) based on the user's prompt.
- **Source Verification:** Expandable UI elements display the exact document chunks or web data used to generate the assistant's response.
- **Automated Testing:** Includes an independent test suite (`test_rag.py`) to validate pipeline integrity and retrieval accuracy.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone <your-repo-link>
   cd dstarix_production_assistant