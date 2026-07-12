import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

load_dotenv()

def get_api_key():
    """Retrieves and validates the Gemini API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is missing from your environment variables or .env file.")
        print("Please check the setup instructions in README.md.")
        sys.exit(1)
    return api_key

def start_chat():
    """Initializes and runs the CLI chat loop."""
    api_key = get_api_key()
    
    try:
       
        client = genai.Client(api_key=api_key)
        
        chat = client.chats.create(model="gemini-2.5-flash")
        
        print("\n==================================================")
        print(" **Gemini AI Chat Assistant Initialized!** ")
        print(" Type your message and press Enter.")
        print(" Type 'exit' or 'quit' to end the conversation.")
        print("==================================================\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit']:
                    print("\nGoodbye!")
                    break
                    
                if not user_input:
                    continue
                
                print("\nAI Assitant: ", end="", flush=True)
                response = chat.send_message(user_input)
                print(response.text)
                print("\n" + "-"*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except APIError as e:
                print(f"\nAPI Error occurred: {e}")
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                
    except Exception as e:
        print(f"Failed to initialize the Gemini client: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_chat()