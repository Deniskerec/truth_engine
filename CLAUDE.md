# Truth Engine

A semantic search engine for X (Twitter) Community Notes. It lets you fact-check headlines by finding similar community notes using vector embeddings and PostgreSQL with pgvector.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  X Community    │     │   PostgreSQL    │     │  check_truth.py │
│  Notes TSV      │────▶│   + pgvector    │◀────│  (Search CLI)   │
│  Data Files     │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       ▲
        │                       │
        ▼                       │
┌─────────────────┐             │
│ ingest_notes.py │─────────────┘
│ (ETL Pipeline)  │
└─────────────────┘
```

## Components

### 1. Database Setup (`init_db.py`)

Initializes PostgreSQL with pgvector extension and creates the schema.

**Table: `community_notes`**
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| note_id | BIGINT | Unique X note identifier |
| summary_text | TEXT | The community note content |
| embedding | vector(384) | Semantic embedding from all-MiniLM-L6-v2 |
| created_at | TIMESTAMP | Insertion timestamp |

**Index**: HNSW index on embeddings for fast approximate nearest neighbor search.

### 2. Data Ingestion (`ingest_notes.py`)

ETL pipeline that:
1. Loads X Community Notes TSV files
2. Joins `notes-00000.tsv` with `noteStatusHistory-00000.tsv`
3. Filters for `CURRENTLY_RATED_HELPFUL` status only
4. Generates 384-dimensional embeddings using `all-MiniLM-L6-v2`
5. Batch inserts into PostgreSQL (1000 rows per batch)
6. Shows progress with tqdm

### 3. Search Engine (`check_truth.py`)

Interactive CLI that:
1. Takes a headline/claim as input
2. Converts it to an embedding
3. Finds top 3 nearest neighbors using cosine distance (`<=>`)
4. Color-codes results:
   - **Red**: Distance < 0.4 = potential misinformation match
   - **Green**: Distance >= 0.4 = no concerning matches

## Data Source

### Where does the data come from?

X (Twitter) publishes Community Notes data publicly at:
**https://x.com/i/communitynotes/download-data**

The data is released as TSV (tab-separated values) files:
- `notes-00000.tsv` - All community notes with their content
- `noteStatusHistory-00000.tsv` - Status history including current rating

### What data do we use?

We only ingest notes marked as `CURRENTLY_RATED_HELPFUL`. These are notes that:
- Have been reviewed by the community
- Achieved consensus as being helpful/accurate
- Are actively displayed on X

### When is new data available?

X updates the Community Notes data dump **daily**. The files are refreshed every 24 hours.

### How to update the database

1. Download fresh TSV files from X
2. Place them in the project directory
3. Run the ingestion script:
   ```bash
   python ingest_notes.py
   ```

The script uses `ON CONFLICT` upserts, so existing notes are updated and new notes are added.

## Database Credentials

```
Host:     localhost
Port:     5432
Database: truth_db
User:     truth_user
Password: truth_password
```

## Embedding Model

**Model**: `all-MiniLM-L6-v2` from Sentence Transformers

- Dimensions: 384
- Optimized for semantic similarity
- Fast inference, small footprint (~80MB)
- Good balance of speed and accuracy for English text

## Usage

### First-time setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize database (requires PostgreSQL with pgvector)
python init_db.py

# Download data from X and place TSV files in project root
# Then ingest the data
python ingest_notes.py
```

### Running the fact-checker

```bash
python check_truth.py
```

Example session:
```
Loading embedding model: all-MiniLM-L6-v2
Connecting to database...

==================================================
Community Notes Fact-Checker
==================================================

Enter a headline to fact-check (or 'q' to quit): vaccines cause autism

⚠️  MATCH FOUND
--------------------------------------------------
#1 [Distance: 0.2341] (76.6% similar)
This claim has been debunked. Multiple large-scale studies...

Enter a headline to fact-check (or 'q' to quit): q
Goodbye!
```

## Similarity Threshold

The threshold of **0.4** cosine distance means:
- < 0.4: High similarity (60%+ match) - likely related misinformation
- >= 0.4: Low similarity - probably not in the database

Cosine distance of 0 = identical, 2 = opposite.

## File Structure

```
truth-engine/
├── CLAUDE.md           # This file
├── requirements.txt    # Python dependencies
├── init_db.py          # Database initialization
├── ingest_notes.py     # Data ingestion pipeline
├── check_truth.py      # Interactive search CLI
├── notes-00000.tsv     # X data (download separately)
└── noteStatusHistory-00000.tsv  # X data (download separately)
```

## Dependencies

- `psycopg2-binary` - PostgreSQL adapter
- `pandas` - TSV loading and data manipulation
- `sentence-transformers` - Embedding generation
- `tqdm` - Progress bars

## Requirements

- Python 3.10+
- PostgreSQL 14+ with pgvector extension
- ~2GB RAM for embedding model + data processing
