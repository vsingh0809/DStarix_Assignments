import os
import sys
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables securely
load_dotenv()

def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is missing from your .env file.")
        print("Please check the setup instructions in README.md.")
        sys.exit(1)
    return api_key

# 1. Define the Agent's State
class State(TypedDict):
    """The state of the graph, storing the conversation history."""
    messages: Annotated[list, add_messages]

def initialize_research_agent(api_key):
    """Builds and compiles the LangGraph research agent."""
    
    # 2. Initialize the Search Tool
    # DuckDuckGo is used here as a completely free web search tool
    search_tool = DuckDuckGoSearchRun()
    tools = [search_tool]

    # 3. Initialize the LLM and bind the tools to it
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2  # Keep it low for factual research
    )
    llm_with_tools = llm.bind_tools(tools)

    # 4. Define the Chatbot Node Function
    def chatbot(state: State):
        """Processes the state and invokes the LLM."""
        sys_msg = SystemMessage(
            content="You are an advanced AI Research Agent. "
                    "When a user asks a question, use your search tool to gather information "
                    "from multiple web sources. Once you have enough context, synthesize the "
                    "information and provide a comprehensive, well-structured summary. "
                    "Always mention that you gathered this from web sources."
        )
        # Prepend the system message to the current conversation history
        messages = [sys_msg] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # 5. Build the LangGraph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("agent", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    # Add edges and conditional routing
    graph_builder.add_edge(START, "agent")
    
    # If the LLM decides to use a tool, go to "tools". Otherwise, end execution.
    graph_builder.add_conditional_edges("agent", tools_condition)
    
    # After the tool runs, always return to the agent to synthesize the findings
    graph_builder.add_edge("tools", "agent")

    # Compile the graph into a runnable application
    agent_graph = graph_builder.compile()
    return agent_graph

def run_cli():
    """Runs the interactive command-line interface."""
    api_key = get_api_key()
    
    print("Initializing Research Agent (Loading Tools and LangGraph)...")
    research_agent = initialize_research_agent(api_key)
    
    print("\n=======================================================")
    print(" **LangGraph AI Research Agent Initialized** ")
    print(" Give me a topic to research and summarize.")
    print(" Type 'exit' or 'quit' to end the session.")
    print("=======================================================\n")
    
    while True:
        try:
            user_query = input("You (Topic/Question): ").strip()
            
            if user_query.lower() in ['exit', 'quit']:
                print("\nShutting down Research Agent. Goodbye!")
                break
                
            if not user_query:
                continue
                
            print("\nAgent: Researching... (This may take a few seconds as I browse the web)\n")
            
            # Prepare the initial state
            initial_state = {"messages": [HumanMessage(content=user_query)]}
            
            # Stream the execution of the graph
            for event in research_agent.stream(initial_state, stream_mode="values"):
                # Grab the latest message in the state
                latest_message = event["messages"][-1]
                
                # Check if it's a tool call
                if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                    print(f"[*] Tool Triggered: Searching the web for '{latest_message.tool_calls[0]['args'].get('query', user_query)}'...")
            
            # Print the final synthesized response
            print("\nFinal Research Summary:")
            print(event["messages"][-1].content)
            print("\n" + "-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\nSession terminated by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred during execution: {e}")

if __name__ == "__main__":
    run_cli()