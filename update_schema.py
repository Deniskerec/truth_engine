import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "truth_db",
    "user": "truth_user",
    "password": "truth_password",
}

def add_format_column():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    try:
        print("Adding 'tweet_format' column to fact_checks table...")
        cursor.execute("ALTER TABLE fact_checks ADD COLUMN IF NOT EXISTS tweet_format TEXT;")
        conn.commit()
        print("Success! Schema updated.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    add_format_column()