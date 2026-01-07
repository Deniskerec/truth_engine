# Truth Engine

PostgreSQL database with pgvector extension for the Truth Engine project.

## Quick Start

```bash
# Start the database
docker-compose up -d

# Check it's running
docker-compose ps

# Connect to the database
psql -h localhost -U truth_user -d truth_db

# Stop the database
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

## Connection Details

- **Host:** localhost
- **Port:** 5432
- **Database:** truth_db
- **User:** truth_user
- **Password:** truth_password

## Enable pgvector Extension

Once connected, enable the vector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```
# twitter_to_vector
# truth_engine
# truth_engine
