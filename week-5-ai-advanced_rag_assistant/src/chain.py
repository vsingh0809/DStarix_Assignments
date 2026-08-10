from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings
from src.retriever import build_advanced_retriever
from src.schemas import AssistantResponse
from src.utils.logger import setup_logger

logger = setup_logger("ChainBuilder")

def format_docs(docs) -> str:
    """Helper utility to format retrieved document chunks into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain():
    """Builds and returns the full RAG pipeline."""
    logger.info("Initializing Language Model...")
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=settings.TEMPERATURE
    )

    retriever = build_advanced_retriever(llm)

    system_prompt = (
        "You are an AI Knowledge Base Assistant. "
        "Answer the question using ONLY the provided context.\n"
        "If the information is not contained in the context, explicitly state that "
        "you do not know and set confidence to 'Low'.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {question}")
    ])

    structured_llm = llm.with_structured_output(AssistantResponse)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
    )

    logger.info("Advanced RAG Chain construction complete.")
    return rag_chain