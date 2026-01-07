"""
Ingest X Community Notes data into PostgreSQL with embeddings.

Auto-downloads the latest TSV files from X's public data archive,
filters for CURRENTLY_RATED_HELPFUL notes, deduplicates by tweet,
generates embeddings, and batch inserts into the fact_checks table.
"""

import os
import zipfile
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
import requests
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "truth_db",
    "user": "truth_user",
    "password": "truth_password",
}

BATCH_SIZE = 1000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATA_DIR = "data"

BASE_URL = "https://ton.twimg.com/birdwatch-public-data"

# Maps base filename to its subdirectory
DATASETS = {
    "notes-00000": "notes",
    "noteStatusHistory-00000": "noteStatusHistory",
}


def download_file(url: str, local_path: str, desc: str) -> bool:
    """Download a file with progress bar. Returns True on success."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(local_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True

    except requests.exceptions.RequestException:
        if os.path.exists(local_path):
            os.remove(local_path)
        return False


def download_latest_data() -> tuple[str, str] | None:
    """
    Download the latest Community Notes data files.
    Tries today, yesterday, and day before yesterday.
    Handles new URL structure with subdirectories and ZIP files.
    Returns tuple of (notes_path, status_path) or None if failed.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Try today, yesterday, and day before
    for days_ago in range(3):
        date = datetime.now() - timedelta(days=days_ago)
        year = date.strftime("%Y")
        month = date.strftime("%m")
        day = date.strftime("%d")

        date_str = f"{year}/{month}/{day}"
        print(f"Trying to download data from {date_str}...")

        success = True
        local_paths = []

        for base_name, subdir in DATASETS.items():
            tsv_filename = f"{base_name}.tsv"
            # This is where we WANT the final file to be (with date prefix)
            local_tsv_path = os.path.join(DATA_DIR, f"{year}-{month}-{day}_{tsv_filename}")
            local_paths.append(local_tsv_path)

            # Skip if TSV file already exists
            if os.path.exists(local_tsv_path):
                print(f"  {tsv_filename} already exists, skipping download")
                continue

            # ---------------------------------------------------------
            # STRATEGY 1: Try ZIP (New Format)
            # ---------------------------------------------------------
            zip_url = f"{BASE_URL}/{year}/{month}/{day}/{subdir}/{base_name}.zip"
            local_zip_path = os.path.join(DATA_DIR, f"{year}-{month}-{day}_{base_name}.zip")

            print(f"  Trying ZIP: {base_name}.zip...")
            if download_file(zip_url, local_zip_path, f"  {base_name}.zip"):
                # Extract TSV from ZIP
                try:
                    with zipfile.ZipFile(local_zip_path, 'r') as zf:
                        # Find the actual name of the file inside the zip (usually just "notes-00000.tsv")
                        internal_files = [n for n in zf.namelist() if n.endswith('.tsv')]
                        
                        if internal_files:
                            internal_name = internal_files[0]
                            # Extract to temp location
                            zf.extract(internal_name, DATA_DIR)
                            
                            # CRITICAL FIX: Rename the extracted file to our date-stamped name
                            extracted_raw_path = os.path.join(DATA_DIR, internal_name)
                            
                            # Remove existing file if present to avoid errors
                            if os.path.exists(local_tsv_path):
                                os.remove(local_tsv_path)
                                
                            os.rename(extracted_raw_path, local_tsv_path)
                            print(f"  Extracted & Renamed to: {local_tsv_path}")
                        else:
                            print(f"  No TSV found in ZIP")
                            success = False
                            break
                except Exception as e:
                    print(f"  Extraction failed: {e}")
                    success = False
                    break
                finally:
                    # Clean up ZIP file
                    if os.path.exists(local_zip_path):
                        os.remove(local_zip_path)

            # ---------------------------------------------------------
            # STRATEGY 2: Fallback to direct TSV (New Format)
            # ---------------------------------------------------------
            else:
                tsv_url = f"{BASE_URL}/{year}/{month}/{day}/{subdir}/{tsv_filename}"
                print(f"  ZIP failed, trying TSV: {tsv_filename}...")

                if not download_file(tsv_url, local_tsv_path, f"  {tsv_filename}"):
                    
                    # ---------------------------------------------------------
                    # STRATEGY 3: Legacy URL (Old Format)
                    # ---------------------------------------------------------
                    old_tsv_url = f"{BASE_URL}/{year}/{month}/{day}/{tsv_filename}"
                    print(f"  Trying legacy URL...")

                    if not download_file(old_tsv_url, local_tsv_path, f"  {tsv_filename}"):
                        print(f"  Failed to download {base_name}")
                        success = False
                        break

            print(f"  Ready: {local_tsv_path}")

        if success:
            print(f"Successfully downloaded data from {date_str}")
            return tuple(local_paths)

    print("Failed to download data from the last 3 days")
    return None


def load_and_filter_notes(notes_path: str, status_path: str) -> pd.DataFrame:
    """Load TSV files and filter for currently helpful notes."""
    print("Loading notes TSV...")
    # Using low_memory=False to suppress mixed type warnings
    notes_df = pd.read_csv(notes_path, sep="\t", low_memory=False)

    print("Loading status history TSV...")
    status_df = pd.read_csv(status_path, sep="\t", low_memory=False)

    # Filter for CURRENTLY_RATED_HELPFUL status
    helpful_status = status_df[status_df["currentStatus"] == "CURRENTLY_RATED_HELPFUL"]

    # Join on noteId to get only helpful notes
    helpful_notes = notes_df.merge(
        helpful_status[["noteId"]],
        on="noteId",
        how="inner"
    )

    print(f"Found {len(helpful_notes)} helpful notes out of {len(notes_df)} total")
    return helpful_notes


def generate_embeddings_batch(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def insert_batch(cursor, batch_data: list[tuple]):
    """Insert a batch of records into fact_checks."""
    execute_values(
        cursor,
        """
        INSERT INTO fact_checks (tweet_id, tweet_url, note_text, note_vector)
        VALUES %s
        ON CONFLICT (tweet_id) DO UPDATE SET
            tweet_url = EXCLUDED.tweet_url,
            note_text = EXCLUDED.note_text,
            note_vector = EXCLUDED.note_vector
        """,
        batch_data,
        template="(%s, %s, %s, %s::vector)"
    )


def main():
    # Download latest data
    paths = download_latest_data()
    if paths is None:
        print("Could not download data. Exiting.")
        return

    notes_path, status_path = paths

    # Load and filter data
    helpful_notes = load_and_filter_notes(notes_path, status_path)

    if helpful_notes.empty:
        print("No helpful notes found. Exiting.")
        return

    # --- CRITICAL FIX: Deduplicate by tweetId ---
    # We only need one note per tweet. This prevents batch collisions.
    print(f"Deduplicating notes... (Initial count: {len(helpful_notes)})")
    helpful_notes = helpful_notes.drop_duplicates(subset=['tweetId'], keep='first')
    print(f"Unique tweets to process: {len(helpful_notes)}")
    # --------------------------------------------

    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Connect to database
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Process in batches
    total_notes = len(helpful_notes)
    total_batches = (total_notes + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Processing {total_notes} notes in {total_batches} batches of {BATCH_SIZE}...")

    inserted_count = 0

    for batch_start in tqdm(range(0, total_notes, BATCH_SIZE), desc="Inserting batches"):
        batch_end = min(batch_start + BATCH_SIZE, total_notes)
        batch_df = helpful_notes.iloc[batch_start:batch_end]

        # Extract tweet IDs and summaries
        tweet_ids = batch_df["tweetId"].tolist()
        summaries = batch_df["summary"].fillna("").tolist()

        # Build tweet URLs
        tweet_urls = [f"https://twitter.com/i/web/status/{tid}" for tid in tweet_ids]

        # Generate embeddings for note text (The Truth)
        note_vectors = generate_embeddings_batch(model, summaries)

        # Prepare batch data for insertion
        batch_data = [
            (tweet_id, tweet_url, note_text, note_vector)
            for tweet_id, tweet_url, note_text, note_vector
            in zip(tweet_ids, tweet_urls, summaries, note_vectors)
        ]

        # Insert batch
        insert_batch(cursor, batch_data)
        conn.commit()

        inserted_count += len(batch_data)

    cursor.close()
    conn.close()

    print(f"Successfully inserted {inserted_count} fact checks with embeddings")


if __name__ == "__main__":
    main()