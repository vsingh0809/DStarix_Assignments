import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from config.settings import settings
from src.schemas import RAGConfig
from src.document_loader import process_and_vectorize
from src.chain import build_agent_chain
from src.utils.logger import setup_logger

logger = setup_logger("StreamlitUI")
st.set_page_config(page_title="DStarix Production Assistant", layout="wide")

# ==========================================
# 1. Initialization
# ==========================================
def initialize_session():
    try:
        settings.validate()
    except Exception as e:
        st.error(f"Configuration Error: {e}")
        st.stop()
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

initialize_session()

# ==========================================
# 2. Sidebar: Configuration & File Upload
# ==========================================
with st.sidebar:
    st.header("⚙️ RAG Engine Settings")
    uploaded_file = st.file_uploader("Upload Document (PDF/TXT)", type=["pdf", "txt"])
    
    st.subheader("Tuning Parameters")
    c_size = st.slider("Chunk Size", 100, 2000, settings.DEFAULT_CHUNK_SIZE, 100)
    c_overlap = st.slider("Chunk Overlap", 0, 500, settings.DEFAULT_CHUNK_OVERLAP, 50)
    k_count = st.slider("Retrieval Count (K)", 1, 10, settings.DEFAULT_RETRIEVAL_COUNT)

    # Pack tuning parameters into a strict Pydantic object
    rag_config = RAGConfig(
        chunk_size=c_size, 
        chunk_overlap=c_overlap, 
        retrieval_count=k_count
    )

    if st.button("Process Document"):
        if uploaded_file:
            with st.spinner("Processing Knowledge Base..."):
                st.session_state.vectorstore = process_and_vectorize(uploaded_file, rag_config)
            st.success("Document active in Knowledge Base!")
        else:
            st.warning("Upload a file first.")
            
    if st.button("Clear Memory"):
        st.session_state.chat_history = []
        st.rerun()

# ==========================================
# 3. Main Chat Interface
# ==========================================
st.title("🤖 DStarix Production Assistant")
st.markdown("Features: **RAG Document Q&A**, **Web Search**, and **Conversation Memory**.")

# Render history
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# Handle user input
if user_input := st.chat_input("Ask a question about the document or search the web..."):
    # Render user prompt
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Build the agent on the fly with current state and RAG config
    agent_executor = build_agent_chain(
        vectorstore=st.session_state.vectorstore, 
        config=rag_config
    )

    # Generate AI Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing & Retrieving..."):
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })
            
            answer = response["output"]
            st.write(answer)
            
            # Extract and display Tool Source Data
            for action, observation in response.get("intermediate_steps", []):
                if action.tool == "Knowledge_Base_Search":
                    with st.expander("📄 View Retrieved Document Sources"):
                        st.markdown(f"```text\n{observation}\n```")
                elif action.tool == "Web_Search":
                    with st.expander("🌐 View Web Search Sources"):
                        st.write(observation)

    # Save to memory
    st.session_state.chat_history.append(AIMessage(content=answer))