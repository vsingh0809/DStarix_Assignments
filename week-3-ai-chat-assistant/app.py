import os
import sys
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

DOCUMENT_PATH = "Internship Rule Book.pdf"  # Change to "internship_guide.pdf" if using PDF
VECTOR_STORE_DIR = "./chroma_db"

def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY is missing from your .env file.")
        print("Please check the setup instructions in README.md.")
        sys.exit(1)
    return api_key

def load_and_split_document(file_path):
    """Loads the internship guide and splits it into manageable chunks."""
    if not os.path.exists(file_path):
        print(f"Error: Knowledge base file '{file_path}' not found.")
        print("Please place your 'internship_guide.txt' or '.pdf' in the root directory.")
        sys.exit(1)

    print(f"Loading document from: {file_path}...")
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()

    # Chunking Strategy: Recursive split by character to keep semantic blocks intact
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Document successfully split into {len(chunks)} chunks.")
    return chunks

def build_or_load_vectorstore(api_key):
    """Creates embeddings and persists vector store using ChromaDB."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    if os.path.exists(VECTOR_STORE_DIR) and os.listdir(VECTOR_STORE_DIR):
        print("Loading existing Chroma vector store...")
        vectorstore = Chroma(
            persist_directory=VECTOR_STORE_DIR,
            embedding_function=embeddings
        )
    else:
        print("Creating new vector embeddings and building Chroma vector store...")
        chunks = load_and_split_document(DOCUMENT_PATH)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_DIR
        )
        print("Vector store built and persisted successfully.")

    return vectorstore

def create_rag_chain(vectorstore, api_key):
    """Constructs the RAG pipeline using LangChain."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0  # Low temperature for factual, document-bound accuracy
    )

    # Retriever configuration using similarity search
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Context-focused Prompt Template
    system_prompt = (
        "You are an assistant for answering questions about the DStarix Internship Program.\n"
        "Use the following retrieved context to answer the user's question accurately.\n"
        "If you do not know the answer based on the context, say that you do not know based on the provided guide.\n"
        "Keep the answer clear, professional, and concise.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def run_cli_chat():
    api_key = get_api_key()
    vectorstore = build_or_load_vectorstore(api_key)
    rag_chain = create_rag_chain(vectorstore, api_key)

    print("\n=======================================================")
    print(" **DStarix Internship Guide RAG Chatbot Initialized** ")
    print(" Ask any question regarding the internship rules, policies, or tasks.")
    print(" Type 'exit' or 'quit' to end the session.")
    print("=======================================================\n")

    while True:
        try:
            user_query = input("You: ").strip()

            if user_query.lower() in ['exit', 'quit']:
                print("\nGoodbye!")
                break

            if not user_query:
                continue

            print("\nAI: ", end="", flush=True)
            response = rag_chain.invoke({"input": user_query})
            print(response["answer"])
            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n\nSession terminated. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    run_cli_chat()