from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    MultiQueryRetriever,
)
from langchain.retrievers.document_compressors import LLMChainExtractor

from config.settings import settings
from src.document_loader import load_and_split_documents
from src.utils.logger import setup_logger

logger = setup_logger("RetrieverBuilder")

def build_advanced_retriever(llm: ChatGoogleGenerativeAI):
    """Assembles a multi-stage hybrid retriever with compression and reranking."""
    chunks = load_and_split_documents()

    # 1. Sparse Search (BM25)
    logger.info("Initializing BM25 Sparse Retriever...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = settings.TOP_K_RESULTS

    # 2. Dense Search (Chroma Vector Database)
    logger.info("Initializing Chroma Vector Retriever...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        google_api_key=settings.GEMINI_API_KEY
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": settings.TOP_K_RESULTS}
    )

    # 3. Hybrid Search (Ensemble)
    logger.info("Merging retrievers into Hybrid Ensemble...")
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[settings.BM25_WEIGHT, settings.VECTOR_WEIGHT]
    )

    # 4. Query Transformation (Multi-Query Expansion)
    logger.info("Wrapping with MultiQueryRetriever for query variation expansion...")
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm
    )

    # 5. Reranking / Contextual Compression
    logger.info("Applying Contextual Compression (LLM Reranker)...")
    compressor = LLMChainExtractor.from_llm(llm)
    
    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever
    )

    return advanced_retriever