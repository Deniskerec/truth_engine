"""
Community Notes Search Engine (Direct Mode).

Since we don't have the original tweets ("Lies"), 
we search the Notes ("Truths") directly.
"""

import psycopg2
from sentence_transformers import SentenceTransformer

# Database Config
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "truth_db",
    "user": "truth_user",
    "password": "truth_password",
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Notes are longer and more detailed, so we can be slightly looser with the threshold
SIMILARITY_THRESHOLD = 0.5  
TOP_K = 3

# ANSI Colors
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[96m"

def search_notes(cursor, embedding: list[float], top_k: int = TOP_K) -> list[tuple]:
    """
    Search against 'note_vector' (The Truth) directly.
    """
    cursor.execute(
        """
        SELECT note_text, tweet_url, note_vector <=> %s::vector AS distance
        FROM fact_checks
        ORDER BY distance ASC
        LIMIT %s
        """,
        (embedding, top_k)
    )
    return cursor.fetchall()

def display_results(results: list[tuple]):
    if not results:
        print("No matches found.")
        return

    best_distance = results[0][2]

    if best_distance < SIMILARITY_THRESHOLD:
        print(f"\n{GREEN}{BOLD}✅ RELEVANT FACT CHECKS FOUND{RESET}")
        print("-" * 60)
        
        for i, (note_text, tweet_url, distance) in enumerate(results, 1):
            if distance > SIMILARITY_THRESHOLD:
                continue
                
            similarity = (1 - distance) * 100
            print(f"{BOLD}Result #{i} ({similarity:.1f}% Match):{RESET}")
            print(f"{note_text}")
            print(f"{CYAN}Source: {tweet_url}{RESET}")
            print("-" * 60)
    else:
        print(f"\nNo exact matches. Here is the closest topic in the database:")
        print("-" * 60)
        note_text, tweet_url, distance = results[0]
        print(f"{note_text}")
        print(f"Source: {tweet_url}")

def main():
    print(f"Loading AI model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print(f"{BOLD}COMMUNITY NOTES SEARCH{RESET}")
    print("Database contains 172,000+ verified notes.")
    print("=" * 60)

    while True:
        try:
            print()
            user_input = input(f"{BOLD}Search a topic (or 'q' to quit):{RESET} ").strip()
            
            if user_input.lower() in ('q', 'quit', 'exit'):
                break
            
            if not user_input:
                continue

            # Generate embedding
            query_vector = model.encode(user_input).tolist()

            # Search DB
            results = search_notes(cursor, query_vector)

            # Show results
            display_results(results)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    cursor.close()
    conn.close()
    print("\nGoodbye.")

if __name__ == "__main__":
    main()