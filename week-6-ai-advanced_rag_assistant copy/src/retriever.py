from langchain.tools.retriever import create_retriever_tool
from src.schemas import RAGConfig
from src.utils.logger import setup_logger

logger = setup_logger("Retriever")

def get_retriever_tool(vectorstore, config: RAGConfig):
    """Wraps the vectorstore into a LangChain tool with descriptions for the LLM."""
    if not vectorstore:
        return None
        
    logger.info(f"Building Knowledge Base Tool (Top K: {config.retrieval_count})")
    
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": config.retrieval_count}
    )
    
    rag_tool = create_retriever_tool(
        retriever,
        name="Knowledge_Base_Search",
        description="Searches the user's uploaded document. Use this tool FIRST to answer any questions regarding uploaded data, policies, or specific document content."
    )
    
    return rag_tool