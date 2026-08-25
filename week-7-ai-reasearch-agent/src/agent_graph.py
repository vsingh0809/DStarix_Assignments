from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from config.settings import settings
from src.schemas import AgentState
from src.tools import get_research_tools
from src.utils.logger import setup_logger

logger = setup_logger("GraphBuilder")

def build_research_graph() -> StateGraph:
    """Builds and compiles the LangGraph state machine for the research assistant."""
    
    # 1. Initialize Tools and LLM
    tools = get_research_tools()
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=settings.TEMPERATURE
    )
    
    # Bind the tools to the LLM so it knows they exist
    llm_with_tools = llm.bind_tools(tools)

    # 2. Define the Agent Node
    def agent_node(state: AgentState):
        """Invokes the LLM to either respond or call a tool."""
        sys_msg = SystemMessage(
            content=(
                "You are an expert AI Research Assistant. "
                "Your goal is to gather information using your web search tool, "
                "process the findings, and generate a clear, well-structured summary. "
                "Always cite that you gathered the information from web sources. "
                "If the search yields no results, politely inform the user."
            )
        )
        # Combine system prompt with the current conversation history
        messages = [sys_msg] + state["messages"]
        
        try:
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"LLM Invocation Error: {e}")
            raise e

    # 3. Build the Graph
    logger.info("Assembling LangGraph Nodes and Edges...")
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools=tools))

    # Add Edges
    workflow.add_edge(START, "agent")
    
    # Conditional Routing: If the agent calls a tool, go to 'tools'. If finished, go to END.
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END}
    )
    
    # Once the tool finishes executing, ALWAYS return to the agent to synthesize the data
    workflow.add_edge("tools", "agent")

    # Compile into a runnable application
    logger.info("LangGraph compilation complete.")
    return workflow.compile()