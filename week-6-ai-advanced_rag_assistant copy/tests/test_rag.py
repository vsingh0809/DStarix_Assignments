import sys
from pathlib import Path

# Add the project root directory to the Python path to allow structured src imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.settings import settings
from src.schemas import RAGConfig
from src.document_loader import process_and_vectorize
from src.chain import build_agent_chain
from src.utils.logger import setup_logger

logger = setup_logger("RAG_TestSuite")

class MockUploadedFile:
    """Simulates a Streamlit uploaded file object for pure backend testing."""
    def __init__(self, name: str, content: str):
        self.name = name
        self.content = content.encode("utf-8")
        
    def getvalue(self) -> bytes:
        return self.content

def run_automated_rag_tests():
    """
    Fulfills Week 5 RAG Task #4. 
    Injects a sample document, builds an ephemeral vector store, 
    and tests retrieval accuracy against a deterministic QA matrix.
    """
    try:
        settings.validate()
    except Exception as err:
        logger.critical(f"Setup verification failed: {err}")
        return

    # 1. Generate an isolated, deterministic string to serve as the document text
    test_document_content = (
        "DStarix Cloud Infrastructure Guidelines:\n"
        "1. Staging environment deployments are executed automatically at 02:00 AM UTC daily.\n"
        "2. The allocation limit for serverless memory configurations is strictly 512MB per function.\n"
        "3. System architecture problems must be escalated directly to the DevOps Team Lead, Alex Mercer."
    )
    
    mock_file = MockUploadedFile("cloud_manifesto.txt", test_document_content)
    
    # 2. Configure deterministic test RAG parameters
    test_config = RAGConfig(
        chunk_size=150,
        chunk_overlap=15,
        retrieval_count=2
    )

    logger.info("Initializing testing sandbox environment...")
    
    # 3. Process mock document via production loader logic
    vectorstore = process_and_vectorize(mock_file, test_config)
    
    # 4. Generate the complete agent tooling chain
    agent_executor = build_agent_chain(vectorstore, test_config)

    # 5. Define evaluation suite queries
    eval_matrix = [
        {
            "query": "When are staging deployments executed?",
            "expected_keyword": "02:00 AM UTC"
        },
        {
            "query": "What is the memory limit for serverless configurations?",
            "expected_keyword": "512MB"
        },
        {
            "query": "Who is the DevOps Team Lead?",
            "expected_keyword": "Alex Mercer"
        }
    ]

    print("\n" + "="*60)
    print(" 🧪 RUNNING ISOLATED END-TO-END RAG EVALUATION SUITE")
    print("="*60)

    all_passed = True

    for idx, test_case in enumerate(eval_matrix, 1):
        print(f"\n[Test Case {idx}] Query: '{test_case['query']}'")
        
        # Invoke chain with an empty list for chat history to isolate tests
        response = agent_executor.invoke({
            "input": test_case["query"],
            "chat_history": []
        })
        
        actual_output = response["output"]
        print(f"🤖 Output: {actual_output}")
        
        # Extract intermediate structural steps to verify tool triggered correctly
        intermediate_steps = response.get("intermediate_steps", [])
        rag_tool_triggered = any(
            action.tool == "Knowledge_Base_Search" for action, _ in intermediate_steps
        )
        
        # Validation checks
        keyword_match = test_case["expected_keyword"].lower() in actual_output.lower()
        
        print(f"📋 Verification Metrics:")
        print(f"   - RAG Retrieval Triggered: {'✅ PASS' if rag_tool_triggered else '❌ FAIL'}")
        print(f"   - Content Accuracy Match:  {'✅ PASS' if keyword_match else '❌ FAIL'}")
        
        if not (rag_tool_triggered and keyword_match):
            all_passed = False
            
        print("-" * 60)

    print("\n" + "="*60)
    if all_passed:
        print(" 🎉 SUMMARY: ALL RAG PIPELINE TESTS PASSED SUCCESSFULLY!")
    else:
        print(" ⚠️ SUMMARY: SOME RAG TESTS FAILED. PLEASE CHECK TUNING PARAMS.")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_automated_rag_tests()