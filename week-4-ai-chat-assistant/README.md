# LangGraph AI Research Agent

## Project Description
This repository contains an autonomous AI Research Agent built using **LangGraph** and the **Google Gemini API**. The agent leverages dynamic tool-calling via LangChain to browse the internet (using DuckDuckGo Search) when presented with a query. By mapping out a stateful graph of nodes and conditional edges, the AI intelligently decides when to search for external information, gathers data from multiple sources, and synthesizes a well-formatted summary.

## Features
- **Stateful Graph Execution:** Built with LangGraph's `StateGraph` to manage cyclical tool-calling and response synthesis.
- **Autonomous Web Searching:** Integrates `DuckDuckGoSearchRun` to scrape the web for up-to-date information without requiring additional paid API keys.
- **Dynamic Tool Routing:** Uses `tools_condition` to route the LLM to search functions only when external data is necessary to answer the user's prompt.
- **Real-Time Execution Tracking:** The CLI interface streams agent actions, showing the user exactly when a tool is triggered and what query is being searched.

## Technologies Used
- **Python 3.10+**
- **LangGraph & LangChain**
- **Google GenAI / Gemini 2.5 Flash** (Reasoning LLM)
- **DuckDuckGo Search** (Web Browsing Tool)
- **Python-Dotenv** (Credential Management)

## Installation Instructions

1. **Clone the Repository:**
   ```bash
   git clone <your-github-repo-url>
   cd <repository-folder>