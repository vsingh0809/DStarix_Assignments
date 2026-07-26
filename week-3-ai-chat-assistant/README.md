# DStarix Internship Guide Chatbot (RAG Implementation)

## Project Description
This repository contains a Retrieval-Augmented Generation (RAG) powered AI Chatbot designed to answer questions about the DStarix Internship Program strictly using the official Internship Guide context. Built using **LangChain**, **ChromaDB**, and **Google Gemini API**, the application indexes the guide into local vector embeddings and retrieves relevant context to generate accurate, hallucination-free answers.

## Features
- **Retrieval-Augmented Generation (RAG):** Answers are grounded strictly in the provided internship guide text.
- **Vector Storage with ChromaDB:** Efficient local similarity search using Google's `text-embedding-004` embedding model.
- **Document Chunking Strategy:** Implements `RecursiveCharacterTextSplitter` for semantic context retention.
- **Environment Isolation:** Credential security via `.env` file configuration.

## Technologies Used
- **Python 3.10+**
- **LangChain Framework**
- **ChromaDB** (Vector Database)
- **Google GenAI / Gemini 2.5 Flash** (LLM)
- **Google Text Embedding 004** (Embedding Model)
- **Python-Dotenv**

## Installation Instructions

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/vsingh0809/DStarix_Assignments.git](https://github.com/vsingh0809/DStarix_Assignments.git)
   cd DStarix_Assignments