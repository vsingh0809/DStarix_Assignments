import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import settings
from src.schemas import RAGConfig
from src.utils.logger import setup_logger

logger = setup_logger("DocumentLoader")

def process_and_vectorize(uploaded_file, config: RAGConfig):
    """Reads a file, applies chunking parameters, and returns an ephemeral vectorstore."""
    logger.info(f"Processing uploaded file: {uploaded_file.name}")
    
    file_path = settings.TEMP_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # 1. Load File
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(str(file_path))
    elif uploaded_file.name.endswith(".txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError("Unsupported format. Please upload PDF or TXT.")
        
    documents = loader.load()

    # 2. Split Text
    logger.info(f"Splitting text (Size: {config.chunk_size}, Overlap: {config.chunk_overlap})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Create Vector Store
    logger.info("Generating embeddings and building Vector Store...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL, 
        google_api_key=settings.GEMINI_API_KEY
    )
    
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        collection_name="active_session_docs"
    )
    
    # Cleanup
    os.remove(file_path)
    logger.info("Document successfully vectorized.")
    
    return vectorstore