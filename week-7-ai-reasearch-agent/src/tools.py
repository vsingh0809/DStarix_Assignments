from langchain_community.tools import DuckDuckGoSearchRun
from src.utils.logger import setup_logger

logger = setup_logger("Tools")

def get_research_tools() -> list:
    """Initializes and returns a list of tools available to the LangGraph agent."""
    logger.info("Initializing DuckDuckGo Web Search Tool...")
    
    # DuckDuckGo is used as a reliable, free web search tool
    web_search = DuckDuckGoSearchRun(
        name="web_search",
        description="Searches the web for up-to-date information, news, and facts. Use this to gather research data."
    )
    
    return [web_search]