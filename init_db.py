import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "truth_db",
    "user": "truth_user",
    "password": "truth_password",
}


def init_database():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()

    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("pgvector extension enabled")

    # Create fact_checks table (Lie vs Truth pairs)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fact_checks (
            id SERIAL PRIMARY KEY,
            tweet_id BIGINT UNIQUE NOT NULL,
            tweet_url TEXT,
            tweet_text TEXT,
            tweet_vector vector(384),
            note_text TEXT NOT NULL,
            note_vector vector(384),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("fact_checks table created")

    # Create HNSW index on tweet_vector (The Lie)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS fact_checks_tweet_vector_idx
        ON fact_checks
        USING hnsw (tweet_vector vector_cosine_ops);
    """)
    print("HNSW index created on tweet_vector (The Lie)")

    # Create HNSW index on note_vector (The Truth)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS fact_checks_note_vector_idx
        ON fact_checks
        USING hnsw (note_vector vector_cosine_ops);
    """)
    print("HNSW index created on note_vector (The Truth)")

    cur.close()
    conn.close()
    print("Database initialization complete")


if __name__ == "__main__":
    init_database()
