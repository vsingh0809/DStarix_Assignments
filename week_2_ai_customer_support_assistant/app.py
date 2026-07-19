import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Load secure environment configurations
load_dotenv()

def initialize_support_bot():
    """Initializes the LangChain customer support runnable with prompt templates."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is missing from your .env file.")
        print("Please review the setup instructions in README.md.")
        sys.exit(1)
        
    # Initialize the Gemini model via LangChain integration
    # Using gemini-2.5-flash for rapid, cost-effective support interactions
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3  # Kept low for consistent, factual customer support answers
    )
    
    # Implementation of Chat Prompt Templates mapping out the persona constraints
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert, empathetic, and professional Customer Support Assistant. "
            "Your objective is to answer frequently asked customer questions accurately and concisely. "
            "Maintain a helpful tone and focus entirely on providing direct solutions. "
            "If you do not know the answer to a specific tracking or account question, gracefully "
            "advise the customer that a human representative will follow up."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Create the functional operational chain
    chain = prompt | llm
    
    # In-memory dictionary to track session-based chat histories
    ephemeral_store = {}
    
    def get_session_history(session_id: str) -> ChatMessageHistory:
        if session_id not in ephemeral_store:
            ephemeral_store[session_id] = ChatMessageHistory()
        return ephemeral_store[session_id]
        
    # Wrap the chain to inherently manage conversation context history
    configurable_chain = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )
    
    return configurable_chain

def run_cli_portal():
    """Launches the interactive CLI support portal interface."""
    support_bot = initialize_support_bot()
    session_config = {"configurable": {"session_id": "customer_support_session_1"}}
    
    print("\n=======================================================")
    print(" **LangChain Customer Support Portal Active** ")
    print(" Welcome! How can we assist you with your order or service today?")
    print(" (Type 'exit' or 'quit' to terminate the session safely.)")
    print("=======================================================\n")
    
    while True:
        try:
            user_query = input("Customer: ").strip()
            
            if user_query.lower() in ['exit', 'quit']:
                print("\nThank you for contacting customer support. Have a wonderful day!")
                break
                
            if not user_query:
                continue
                
            print("\nAgent: ", end="", flush=True)
            
            # Invoke the LangChain runnable chain
            response = support_bot.invoke({"input": user_query}, config=session_config)
            
            print(response.content)
            print("\n" + "-" * 55 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nSession terminated by user action. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn interface execution error occurred: {e}")
            break

if __name__ == "__main__":
    run_cli_portal()