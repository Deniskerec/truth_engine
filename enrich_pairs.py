"""
Enrich fact_checks using the Twitter Syndication API.
This API is more reliable than oEmbed and provides cleaner data.
"""

import time
import psycopg2
import requests
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "truth_db",
    "user": "truth_user",
    "password": "truth_password",
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 100
REQUEST_DELAY = 1.0  # Syndication API is faster, 1s is safe

# Header to look like a browser requesting an embedded tweet
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Origin": "https://platform.twitter.com",
    "Referer": "https://platform.twitter.com/"
}

def fetch_tweet_details(tweet_url: str) -> tuple[str, str] | None:
    """
    Fetch tweet details using the CDN Syndication API.
    Returns: (text, format) or None.
    """
    # Extract Tweet ID from URL
    try:
        # url looks like https://twitter.com/i/web/status/123456789
        tweet_id = tweet_url.split("/")[-1].split("?")[0]
        if not tweet_id.isdigit():
            return None
    except:
        return None

    # This is the magic endpoint
    api_url = f"https://cdn.syndication.twimg.com/tweet-result?id={tweet_id}&token=x"

    try:
        response = requests.get(api_url, headers=HEADERS, timeout=10)
        
        if response.status_code == 404:
            return "MISSING", "Unknown"
        
        response.raise_for_status()
        data = response.json()

        # 1. Get Text
        text = data.get("text", "")

        # 2. Detect Format from JSON metadata
        tweet_format = "Text"
        
        # Check for Video
        if "video" in data:
            tweet_format = "Video"
        # Check for Photos
        elif "photos" in data and len(data["photos"]) > 0:
            tweet_format = "Photo"
        
        return text, tweet_format

    except Exception as e:
        print(f"  Error fetching {tweet_id}: {e}")
        return None

def get_null_rows(cursor, limit: int = BATCH_SIZE) -> list[tuple]:
    """Get rows where tweet_text is NULL."""
    # We explicitly exclude rows we've already marked as 'Missing' to prevent loops
    cursor.execute(
        """
        SELECT id, tweet_url
        FROM fact_checks
        WHERE tweet_text IS NULL 
        AND (tweet_format IS NULL OR tweet_format != 'Missing')
        LIMIT %s
        """,
        (limit,)
    )
    return cursor.fetchall()

def update_row(cursor, row_id: int, tweet_text: str, tweet_vector: list[float] | None, tweet_format: str):
    """Update database row."""
    if tweet_vector is None:
        # Mark missing so we don't try again
        cursor.execute(
            """
            UPDATE fact_checks
            SET tweet_text = %s, tweet_format = 'Missing'
            WHERE id = %s
            """,
            ("MISSING_OR_DELETED", row_id)
        )
    else:
        cursor.execute(
            """
            UPDATE fact_checks
            SET tweet_text = %s, 
                tweet_vector = %s::vector,
                tweet_format = %s
            WHERE id = %s
            """,
            (tweet_text, tweet_vector, tweet_format, row_id)
        )

def main():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    total_processed = 0
    total_success = 0
    total_missing = 0

    print("Starting enrichment loop (Syndication API)...")
    print("=" * 50)

    while True:
        rows = get_null_rows(cursor)
        if not rows:
            print("\nNo more rows to process. Done!")
            break

        print(f"\nProcessing batch of {len(rows)} rows...")

        for row_id, tweet_url in rows:
            total_processed += 1
            print(f"[{total_processed}] Checking: {tweet_url}")

            result = fetch_tweet_details(tweet_url)

            if result:
                tweet_text, tweet_format = result
                
                if tweet_text == "MISSING":
                    update_row(cursor, row_id, "MISSING", None, "Missing")
                    conn.commit()
                    total_missing += 1
                    print(f"  -> Tweet deleted/suspended (marked as missing)")
                else:
                    # Generate embedding
                    tweet_vector = model.encode(tweet_text).tolist()
                    update_row(cursor, row_id, tweet_text, tweet_vector, tweet_format)
                    conn.commit()
                    total_success += 1
                    preview = tweet_text[:50].replace('\n', ' ')
                    print(f"  -> Success [{tweet_format}]: {preview}...")
            else:
                print(f"  -> API Error (Skipping)")

            time.sleep(REQUEST_DELAY)

    cursor.close()
    conn.close()
    print("\n" + "=" * 50)
    print(f"Enrichment complete.")
    print(f"Found: {total_success}")
    print(f"Missing/Deleted: {total_missing}")

if __name__ == "__main__":
    main()