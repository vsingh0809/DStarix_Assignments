import sys
from config.settings import settings
from src.chain import build_rag_chain
from src.schemas import AssistantResponse
from src.utils.logger import setup_logger

logger = setup_logger("CLI")

def main():
    try:
        # Validate configuration settings
        settings.validate()
    except Exception as err:
        logger.critical(f"Configuration Error: {err}")
        sys.exit(1)

    logger.info("Starting Advanced Knowledge Base Assistant...")
    rag_chain = build_rag_chain()

    print("\n" + "=" * 65)
    print(" 🧠 Advanced RAG Knowledge Base Assistant Active")
    print(" Pipeline: Hybrid Search (BM25 + Chroma) | MultiQuery | Reranking")
    print(" Type 'exit' or 'quit' to end the program.")
    print("=" * 65 + "\n")

    while True:
        try:
            query = input("\nUser Question: ").strip()
            
            if query.lower() in ["exit", "quit"]:
                print("\nShutting down session. Goodbye!")
                break
                
            if not query:
                continue

            logger.info(f"Executing Query: '{query}'")
            
            # Invoke the LCEL Chain
            response: AssistantResponse = rag_chain.invoke(query)

            print("\n" + "-" * 45)
            print(f"🤖 Answer:       {response.answer}")
            print(f"📊 Confidence:   {response.confidence}")
            print(f"📚 Sources Used: {response.sources_used}")
            print("-" * 45)

        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()