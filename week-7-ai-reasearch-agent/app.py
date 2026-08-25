import sys
from langchain_core.messages import HumanMessage

from config.settings import settings
from src.agent_graph import build_research_graph
from src.utils.logger import setup_logger

logger = setup_logger("CLI")

def run_application():
    try:
        settings.validate()
    except ValueError as e:
        logger.critical(f"Startup check failed: {e}")
        sys.exit(1)

    # Compile the graph once at startup
    graph = build_research_graph()
    
    print("\n" + "="*65)
    print(" 🔎 Modular LangGraph Research Assistant Initialized ")
    print(" Type a topic you want me to research and summarize.")
    print(" Type 'exit' or 'quit' to close the application.")
    print("="*65 + "\n")

    # Maintain a session-level state for continuous conversation
    current_state = {"messages": []}

    while True:
        try:
            user_input = input("\nUser (Topic/Query): ").strip()
            
            if user_input.lower() in ["exit", "quit"]:
                print("\nShutting down Research Assistant. Goodbye!")
                break
            if not user_input:
                print("Input cannot be empty. Please try again.")
                continue

            # Append user input to the state
            current_state["messages"].append(HumanMessage(content=user_input))
            
            print("\nAgent: Researching... (Tracking graph execution below)\n")
            
            # Stream the execution of the graph
            for event in graph.stream(current_state, stream_mode="values"):
                latest_msg = event["messages"][-1]
                
                # Check if the LLM decided to invoke a tool
                if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                    for tool_call in latest_msg.tool_calls:
                        query = tool_call['args'].get('query', 'Unknown')
                        print(f"   [*] Action: Executing web search for -> '{query}'")
            
            # Extract and print the final synthesized response
            final_response = event["messages"][-1].content
            print("\n" + "-"*40)
            print("📝 Final Research Summary:")
            print(final_response)
            print("-"*40)

        except KeyboardInterrupt:
            print("\n\nSession interrupted by user. Exiting gracefully...")
            break
        except Exception as e:
            logger.error(f"An unexpected runtime error occurred: {e}")
            # Reset state on failure to prevent infinite error loops
            current_state = {"messages": []}

if __name__ == "__main__":
    run_application()