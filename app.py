"""
Truth Engine - Web Frontend

A simple FastAPI app to search fact-checks and display results.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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
SIMILARITY_THRESHOLD = 0.4

app = FastAPI(title="Truth Engine")

# Load model at startup
model = None
conn = None


@app.on_event("startup")
def startup():
    global model, conn
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)


@app.on_event("shutdown")
def shutdown():
    global conn
    if conn:
        conn.close()


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Truth Engine</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 2rem;
        }
        .container { max-width: 800px; margin: 0 auto; }
        h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #ff6b6b, #feca57);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #888; margin-bottom: 2rem; }
        .search-box {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
        }
        input[type="text"] {
            flex: 1;
            padding: 1rem;
            font-size: 1rem;
            border: 2px solid #333;
            border-radius: 8px;
            background: #1a1a1a;
            color: #fff;
            outline: none;
            transition: border-color 0.2s;
        }
        input[type="text"]:focus { border-color: #ff6b6b; }
        button {
            padding: 1rem 2rem;
            font-size: 1rem;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            background: linear-gradient(90deg, #ff6b6b, #ee5a5a);
            color: #fff;
            cursor: pointer;
            transition: transform 0.1s;
        }
        button:hover { transform: scale(1.02); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .results { display: flex; flex-direction: column; gap: 1rem; }
        .result-card {
            background: #1a1a1a;
            border-radius: 12px;
            padding: 1.5rem;
            border-left: 4px solid #333;
        }
        .result-card.match { border-left-color: #ff6b6b; }
        .result-card.safe { border-left-color: #26de81; }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
        }
        .badge {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .badge.match { background: #ff6b6b22; color: #ff6b6b; }
        .badge.safe { background: #26de8122; color: #26de81; }
        .score { color: #888; font-size: 0.9rem; }
        .section { margin-bottom: 1rem; }
        .section-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            color: #666;
            margin-bottom: 0.5rem;
        }
        .tweet-text { color: #ff8888; }
        .note-text { color: #88ff88; }
        .tweet-url {
            font-size: 0.8rem;
            color: #4a9eff;
            text-decoration: none;
        }
        .tweet-url:hover { text-decoration: underline; }
        .loading { text-align: center; color: #888; padding: 2rem; }
        .empty { text-align: center; color: #666; padding: 3rem; }
        .stats {
            background: #1a1a1a;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            display: flex;
            gap: 2rem;
        }
        .stat { text-align: center; }
        .stat-value { font-size: 1.5rem; font-weight: bold; color: #feca57; }
        .stat-label { font-size: 0.8rem; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Truth Engine</h1>
        <p class="subtitle">Semantic fact-checker powered by X Community Notes</p>

        <div id="stats" class="stats">
            <div class="stat">
                <div class="stat-value" id="total-count">-</div>
                <div class="stat-label">Total Fact Checks</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="enriched-count">-</div>
                <div class="stat-label">With Tweet Text</div>
            </div>
        </div>

        <div class="search-box">
            <input type="text" id="query" placeholder="Enter a claim or headline to fact-check..." />
            <button id="search-btn" onclick="search()">Search</button>
        </div>

        <div id="results" class="results"></div>
    </div>

    <script>
        // Load stats on page load
        fetch('/api/stats')
            .then(r => r.json())
            .then(data => {
                document.getElementById('total-count').textContent = data.total.toLocaleString();
                document.getElementById('enriched-count').textContent = data.enriched.toLocaleString();
            });

        // Search on Enter key
        document.getElementById('query').addEventListener('keypress', e => {
            if (e.key === 'Enter') search();
        });

        async function search() {
            const query = document.getElementById('query').value.trim();
            if (!query) return;

            const btn = document.getElementById('search-btn');
            const results = document.getElementById('results');

            btn.disabled = true;
            btn.textContent = 'Searching...';
            results.innerHTML = '<div class="loading">Searching...</div>';

            try {
                const res = await fetch('/api/search?q=' + encodeURIComponent(query));
                const data = await res.json();

                if (data.results.length === 0) {
                    results.innerHTML = '<div class="empty">No matching fact-checks found.</div>';
                } else {
                    results.innerHTML = data.results.map(r => `
                        <div class="result-card ${r.is_match ? 'match' : 'safe'}">
                            <div class="result-header">
                                <span class="badge ${r.is_match ? 'match' : 'safe'}">
                                    ${r.is_match ? '⚠️ MATCH FOUND' : '✅ Low Similarity'}
                                </span>
                                <span class="score">${(r.similarity * 100).toFixed(1)}% similar (distance: ${r.distance.toFixed(4)})</span>
                            </div>
                            ${r.tweet_text ? `
                                <div class="section">
                                    <div class="section-label">Original Tweet (The Claim)</div>
                                    <div class="tweet-text">${escapeHtml(r.tweet_text)}</div>
                                </div>
                            ` : ''}
                            <div class="section">
                                <div class="section-label">Community Note (The Fact-Check)</div>
                                <div class="note-text">${escapeHtml(r.note_text)}</div>
                            </div>
                            <a class="tweet-url" href="${r.tweet_url}" target="_blank">View original tweet →</a>
                        </div>
                    `).join('');
                }
            } catch (err) {
                results.innerHTML = '<div class="empty">Error searching. Please try again.</div>';
            }

            btn.disabled = false;
            btn.textContent = 'Search';
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""


@app.get("/api/stats")
def get_stats():
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fact_checks")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM fact_checks WHERE tweet_text IS NOT NULL AND tweet_text != 'MISSING_OR_DELETED'")
    enriched = cursor.fetchone()[0]
    cursor.close()
    return {"total": total, "enriched": enriched}


@app.get("/api/search")
def search(q: str):
    # Generate embedding for query
    query_embedding = model.encode(q).tolist()

    cursor = conn.cursor()

    # Search against tweet_vector (The Lie) - finds claims similar to query
    cursor.execute(
        """
        SELECT
            tweet_id,
            tweet_url,
            tweet_text,
            note_text,
            tweet_vector <=> %s::vector AS distance
        FROM fact_checks
        WHERE tweet_vector IS NOT NULL
        ORDER BY distance
        LIMIT 5
        """,
        (query_embedding,)
    )

    results = []
    for row in cursor.fetchall():
        tweet_id, tweet_url, tweet_text, note_text, distance = row
        results.append({
            "tweet_id": tweet_id,
            "tweet_url": tweet_url,
            "tweet_text": tweet_text if tweet_text and tweet_text != "MISSING_OR_DELETED" else None,
            "note_text": note_text,
            "distance": float(distance),
            "similarity": 1 - float(distance),
            "is_match": float(distance) < SIMILARITY_THRESHOLD
        })

    cursor.close()
    return {"query": q, "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
