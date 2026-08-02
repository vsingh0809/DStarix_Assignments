# AI Chat Assistant (CLI Implementation)

## Project Description
This project is a lightweight, terminal-based AI Chat Assistant built with Python using the official Google Gemini API via the `google-genai` SDK. The application acts as a stateful conversational agent, allowing users to interact with advanced language models directly inside their command-line interface. It is designed to be highly secure, maintaining credentials strictly out of the codebase via environment isolation, while remaining simple enough to be deployed locally in under two minutes.

## Features
- **Secure Environment Management:** Fully integrates `.env` workflows to ensure private API keys are never hardcoded or pushed to version control.
- **Robust System Validation:** Proactively checks for missing keys or bad API handshakes, providing clear, human-readable troubleshooting advice instead of raw tracebacks.
- **Clean Command Line Interface:** Formatted text boundaries allow users to easily distinguish between their inputs and the AI's generation.

## Technologies Used
- **Python 3.10+**: Core programming language.
- **Google GenAI SDK (`google-genai`)**: The official, high-performance developer library for Google's models.
- **Python-Dotenv (`python-dotenv`)**: Library used to parse key-value pairs from isolated configuration environments.
- **Gemini 2.5 Flash**: The target LLM used for quick, conversational, and highly accurate text generations.

## Installation Instructions
Before getting started, ensure you have Python version 3.10 or higher installed on your system. 

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/vsingh0809/DStarix_Assignments.git
   cd DStarix_Assignments
   ```

2. **Establish an Isolated Environment (Highly Recommended):**
   - *On macOS/Linux:*
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - *On Windows:*
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install Core Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Setup Instructions
The application relies on process environment variables to run securely. 

1. Create a new file named `.env` in the root directory of the project (the same folder where `app.py` sits).
2. Generate an API Key via [Google AI Studio](https://aistudio.google.com/).
3. Add the key inside your `.env` file using the exact format below:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```
4. Save the file. Ensure that `.env` is listed in your `.gitignore` so your private credentials are never exposed publicly.

## Usage Guide
To launch the application, run the main script from your terminal:

```bash
python app.py
```

- **Chatting:** Type out any question, task, or programming problem after the `You:` prompt and press `Enter`. 
- **Exiting Cleanly:** When you are done, type `exit` or `quit` and press `Enter`. Alternatively, hitting `Ctrl+C` will close the application immediately without throwing unhandled shell errors.

## Project Structure
```text
DStarix_Assignments/week-1-ai-chat-assistant
├── app.py               # Application logic, input validation, and main runtime loop
├── requirements.txt     # Pinpointed library versions needed for clean execution
├── pyproject.toml       # Modern Python packaging and dependency configuration
├── .env                 # Local configuration file holding sensitive credentials (User-Created)
└── README.md            # Comprehensive documentation and project guide
```

## Example Outputs
Below is a conceptual trace of how a typical terminal session behaves:

```text
==================================================
 **Gemini AI Chat Assistant Initialized!** 
 Type your message and press Enter.
 Type 'exit' or 'quit' to end the conversation.
==================================================

You: Tell me a quick joke about programmers.

AI: Why do programmers wear glasses? Because they can't C#!

--------------------------------------------------

You: Explain why that's funny.

AI: It's a pun! "C#" (pronounced C-sharp) is a popular object-oriented programming language developed by Microsoft, but it sounds exactly like "see sharp," which means having excellent vision.

--------------------------------------------------

You: quit

Goodbye!