# LangGraph Modular Research Assistant

A highly scalable, production-ready AI Research Assistant built using **LangGraph** and the **Google Gemini API**. This application utilizes a cyclical state graph to autonomously browse the internet (via DuckDuckGo) and summarize findings based on user prompts.

## 🏗️ Architecture

This project strictly adheres to modular Python coding standards:
- **`config/`**: Centralized `.env` parsing and application settings.
- **`src/schemas.py`**: Pydantic/TypedDict state definitions for type safety.
- **`src/tools.py`**: Isolated external tool integrations.
- **`src/agent_graph.py`**: The core LangGraph state machine (Nodes, Edges, and Routing).
- **`app.py`**: The execution loop, handling edge cases and user interruptions gracefully.

## 🚀 Setup Instructions

1. **Clone the repository & create an isolated environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate