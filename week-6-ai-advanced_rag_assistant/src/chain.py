from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config.settings import settings
from src.retriever import get_retriever_tool
from src.schemas import RAGConfig
from src.utils.logger import setup_logger

logger = setup_logger("ChainBuilder")

def build_agent_chain(vectorstore, config: RAGConfig):
    """Assembles the LLM, Web Search, and RAG into a cohesive LangChain Agent."""
    logger.info("Initializing Agent Chain...")
    
    # 1. Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=settings.TEMPERATURE
    )

    # 2. Define Tools
    web_search = DuckDuckGoSearchRun(name="Web_Search")
    tools = [web_search]

    rag_tool = get_retriever_tool(vectorstore, config)
    if rag_tool:
        tools.append(rag_tool)

    # 3. Define the Prompt (System Instructions + Memory Placeholder)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional AI Assistant. You have access to a web search tool and a document knowledge base. Answer the user accurately. Always prioritize the Knowledge Base if the user's question relates to uploaded data."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 4. Construct the Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Return executor with return_intermediate_steps=True to pull source citations later
    return AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=False, 
        return_intermediate_steps=True
    )