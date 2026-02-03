from src.retrieve import search_documents
import sys
import os

# Add the current directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("Initializing GTO Poker Retrieval System (Local / No-LLM)...")

    # Check if DB exists
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db")
    if not os.path.exists(db_path):
        print("Warning: Database not found. Please run 'python src/ingest.py' first.")

    print("\n--- GTO Poker Search Engine (Type 'exit' to quit) ---")

    while True:
        try:
            query = input("\nEnter your search query: ")
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break

            if not query.strip():
                continue

            print("Retrieving relevant passages...")
            results = search_documents(query, k=3)

            print(f"\nFound {len(results)} relevant passages:\n")
            print("="*60)

            for i, (doc, score) in enumerate(results):
                source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page = doc.metadata.get('page', 'Unknown')
                try:
                    page = int(page) + 1
                except:
                    pass

                content = doc.page_content.strip()

                print(f"Result #{i+1} (Relevance Score: {score:.4f})")
                print(f"Source: {source} (Page {page})")
                print("-" * 30)
                print(f"{content}...") # Truncate if extremely long, or show full
                print("="*60)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
