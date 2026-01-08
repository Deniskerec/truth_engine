"""
Initialize keyword filters table and populate with AI-related keywords.

This creates semantic filters that can be joined with fact_checks
to find notes about specific topics (e.g., AI generated content).
"""

import psycopg2
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "truth_db",
    "user": "truth_user",
    "password": "truth_password",
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Keywords to create filters for
KEYWORDS = [
    "AI generated",
    "AI video",
]


def main():
    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Connect to database
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cursor = conn.cursor()

    # Create keyword_filters table
    print("Creating keyword_filters table...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS keyword_filters (
            id SERIAL PRIMARY KEY,
            keyword TEXT UNIQUE NOT NULL,
            keyword_vector vector(384),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Create HNSW index for fast similarity search
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS keyword_filters_vector_idx
        ON keyword_filters
        USING hnsw (keyword_vector vector_cosine_ops);
    """)
    print("Table and index created.")

    # Insert keywords with embeddings
    print(f"\nInserting {len(KEYWORDS)} keywords...")
    for keyword in KEYWORDS:
        # Generate embedding
        embedding = model.encode(keyword).tolist()

        # Upsert
        cursor.execute(
            """
            INSERT INTO keyword_filters (keyword, keyword_vector)
            VALUES (%s, %s::vector)
            ON CONFLICT (keyword) DO UPDATE SET
                keyword_vector = EXCLUDED.keyword_vector
            """,
            (keyword, embedding)
        )
        print(f"  Inserted: '{keyword}'")

    cursor.close()
    conn.close()

    print("\n" + "=" * 60)
    print("Done! Keyword filters created.")
    print("=" * 60)

    # Print usage instructions
    print("""
USAGE: Find fact_checks that semantically match your keywords

-- Option 1: Find notes similar to ANY keyword (threshold: 0.5)
SELECT
    fc.id,
    fc.tweet_url,
    fc.note_text,
    kf.keyword,
    fc.note_vector <=> kf.keyword_vector AS distance
FROM fact_checks fc
CROSS JOIN keyword_filters kf
WHERE fc.note_vector <=> kf.keyword_vector < 0.5
ORDER BY distance ASC
LIMIT 100;

-- Option 2: Find notes matching a SPECIFIC keyword
SELECT
    fc.id,
    fc.tweet_url,
    fc.note_text,
    fc.note_vector <=> kf.keyword_vector AS distance
FROM fact_checks fc
JOIN keyword_filters kf ON kf.keyword = 'AI generated'
WHERE fc.note_vector <=> kf.keyword_vector < 0.5
ORDER BY distance ASC
LIMIT 50;

-- Option 3: Create a VIEW for easy access
CREATE VIEW ai_related_notes AS
SELECT DISTINCT ON (fc.id)
    fc.id,
    fc.tweet_id,
    fc.tweet_url,
    fc.tweet_text,
    fc.note_text,
    kf.keyword AS matched_keyword,
    fc.note_vector <=> kf.keyword_vector AS similarity_distance
FROM fact_checks fc
CROSS JOIN keyword_filters kf
WHERE fc.note_vector <=> kf.keyword_vector < 0.5
ORDER BY fc.id, similarity_distance ASC;

-- Then just query the view:
SELECT * FROM ai_related_notes LIMIT 100;
""")


if __name__ == "__main__":
    main()
