# coda-search-template

> A production-ready semantic search service — deployable on Railway in under an hour.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com/)
[![pgvector](https://img.shields.io/badge/pgvector-HNSW-orange.svg)](https://github.com/pgvector/pgvector)
[![Railway](https://img.shields.io/badge/deploy-Railway-blueviolet.svg)](https://railway.app/)

---

## What this is

This template gives you a fully working REST API that converts text queries into vector embeddings and retrieves the most semantically similar items from a PostgreSQL database — without any keyword matching or boolean logic.

A user can search for `"something warm to drink on a cold morning"` and receive results for `French Press`, `Electric Kettle`, and `Cold Brew Coffee Maker` — ranked by meaning, not exact words.

**Stack**

| Layer | Technology | Why |
|---|---|---|
| API framework | FastAPI | Async, self-documenting, production-proven |
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, CPU-friendly, ~80 MB, MIT licence |
| Vector database | pgvector on Supabase | Managed Postgres with HNSW indexing |
| Deployment | Railway | Zero-config Docker deploys, env var management |

---

## Live demo — e-commerce product search

The sample dataset included in `data/sample_products.json` contains **50 generic product descriptions** across categories: Kitchen, Electronics, Fitness, Office, Outdoor, Home, Garden, and Health.

Once the service is running, try these queries:

```bash
# "I need something for my morning coffee ritual"
curl -X POST https://your-service.railway.app/search \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "morning coffee ritual", "top_k": 3}'

# "gear for a multi-day camping trip"
curl -X POST https://your-service.railway.app/search \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "gear for a multi-day camping trip", "top_k": 5}'

# Filter by category
curl -X POST https://your-service.railway.app/search \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "ergonomic workspace", "top_k": 5, "filters": {"category": "Office"}}'
```

---

## Architecture

```
Client
  │
  ▼
FastAPI  ──────────────────────────────────────────────────────────
  │                                                                │
  ├─ POST /search                                                  │
  │    └─ embed query (all-MiniLM-L6-v2)                          │
  │    └─ SELECT … ORDER BY embedding <=> :vec LIMIT k            │
  │                                                                │
  ├─ POST /ingest/item  |  POST /ingest/batch                     │
  │    └─ embed description(s)                                     │
  │    └─ INSERT INTO items (…, embedding)                        │
  │                                                                │
  └─ GET /health                                                   │
                                                                   │
Supabase PostgreSQL                                                │
  └─ items table                                                   │
       ├─ id, name, description, metadata (JSONB), created_at     │
       └─ embedding VECTOR(384)                                    │
            └─ HNSW index (vector_cosine_ops, m=16, ef=64) ───────┘
```

The HNSW index means approximate nearest-neighbour search scales to millions of rows while maintaining sub-100 ms database query times.

---

## Local setup

### Prerequisites

- Python 3.11+
- A Supabase project (free tier is sufficient for development)
- `git`

### 1. Clone and install

```bash
git clone https://github.com/coda-builds/coda-search-template.git
cd coda-search-template

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```dotenv
DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[REF].supabase.co:5432/postgres
API_KEY=your-strong-random-key
```

Generate a secure key with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Set up the database

Run once to create the `items` table, HNSW index, and helper triggers:

```bash
python scripts/setup_db.py
```

You should see:

```
INFO  Enabling pgvector extension …
INFO  Creating items table …
INFO  Creating HNSW index (cosine distance) …
INFO  Creating GIN index on metadata …
✅  Database setup complete.
```

### 4. Load the sample dataset

```bash
uvicorn app.main:app --reload &   # start the server in the background

python scripts/ingest_sample_data.py
```

### 5. Test it

```bash
curl -X POST http://localhost:8000/search \
  -H "x-api-key: $(grep API_KEY .env | cut -d= -f2)" \
  -H "Content-Type: application/json" \
  -d '{"query": "lightweight running shoes", "top_k": 3}'
```

Interactive API docs are available at **http://localhost:8000/docs**.

---

## Deploying to Railway

### First deploy

1. Push this repository to GitHub.
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Select your repository. Railway detects the `Dockerfile` and `railway.toml` automatically.
4. Open **Variables** and add:

| Variable | Value |
|---|---|
| `DATABASE_URL` | Your Supabase connection string |
| `API_KEY` | A strong random key |
| `ENVIRONMENT` | `production` |

5. Click **Deploy**. Railway builds the Docker image (approximately 3–5 minutes on first build because it downloads and caches the embedding model).
6. Once the health check at `/health` returns `200 OK`, the service is live.

### Subsequent deploys

Push to `main`. Railway rebuilds and redeploys automatically. The Docker layer cache means subsequent builds take under 60 seconds.

### Environment variables reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | ✅ | — | Supabase PostgreSQL URI |
| `API_KEY` | ✅ | — | Shared secret for all endpoints |
| `ENVIRONMENT` | — | `development` | Set to `production` to disable docs |
| `EMBEDDING_MODEL` | — | `all-MiniLM-L6-v2` | Any sentence-transformers model |
| `EMBEDDING_DIMENSIONS` | — | `384` | Must match the chosen model |
| `DEFAULT_TOP_K` | — | `10` | Default results per search |
| `SIMILARITY_THRESHOLD` | — | `0.3` | Minimum cosine similarity to return |
| `PORT` | — | `8000` | Railway injects this automatically |

---

## API reference

### `POST /search`

Perform a semantic search.

**Headers:** `x-api-key: YOUR_KEY`

**Body:**

```json
{
  "query": "lightweight running shoes",
  "top_k": 10,
  "similarity_threshold": 0.3,
  "filters": { "category": "Footwear" }
}
```

**Response:**

```json
{
  "query": "lightweight running shoes",
  "total": 2,
  "latency_ms": 84.3,
  "results": [
    {
      "similarity": 0.8921,
      "item": {
        "id": "uuid",
        "name": "UltraLight Trail Running Shoe",
        "description": "Breathable mesh upper…",
        "metadata": { "category": "Footwear", "price": 129.99 },
        "created_at": "2024-01-01T00:00:00Z"
      }
    }
  ]
}
```

### `POST /ingest/item`

Ingest a single item.

```json
{
  "name": "My Product",
  "description": "A clear description the model will embed.",
  "metadata": { "category": "Electronics", "price": 99.99 }
}
```

### `POST /ingest/batch`

Ingest up to 500 items in one request. Embeddings are computed in a single batched forward pass.

```json
{
  "items": [
    { "name": "Item A", "description": "...", "metadata": {} },
    { "name": "Item B", "description": "...", "metadata": {} }
  ]
}
```

### `DELETE /ingest/{item_id}`

Remove an item by UUID.

### `GET /health`

Returns service status, database connectivity, and model name. No authentication required. Used by Railway's health check.

---

## Swapping in your own dataset

The service is entirely data-agnostic. To replace the sample product catalogue:

**1. Prepare your data**

Create a JSON array matching this structure:

```json
[
  {
    "name": "Short, human-readable label",
    "description": "The text that will be embedded. Write this as you would a document summary — the richer the language, the better the search quality.",
    "metadata": {
      "any_key": "any_value"
    }
  }
]
```

> **Tip:** The `description` field is what gets embedded. Include all searchable attributes in natural language — category, key features, use cases. A description like `"Lightweight waterproof hiking boot with ankle support for rocky terrain"` will retrieve far better results than `"Boot, size 10, SKU-4421"`.

**2. Clear the existing data (optional)**

```sql
-- Run in the Supabase SQL editor
TRUNCATE TABLE items;
```

**3. Ingest your data**

```bash
# Replace the sample file path
BASE_URL=https://your-service.railway.app \
API_KEY=your-key \
python -c "
import json, httpx, os
with open('your_data.json') as f:
    items = json.load(f)
r = httpx.post(
    os.environ['BASE_URL'] + '/ingest/batch',
    headers={'x-api-key': os.environ['API_KEY']},
    json={'items': items},
    timeout=120,
)
print(r.json())
"
```

**4. Changing the embedding model**

To use a different model (e.g. `all-mpnet-base-v2` for higher accuracy, or a multilingual model):

1. Update `EMBEDDING_MODEL` in your `.env` / Railway variables.
2. Update `EMBEDDING_DIMENSIONS` to match (e.g. `all-mpnet-base-v2` → `768`).
3. Re-run `scripts/setup_db.py` — it will recreate the table and index with the new dimensions.
4. Re-ingest all your data (embeddings from different models are not compatible).

---

## Benchmarking

Run the included benchmark script against any live instance:

```bash
# Against local server
python scripts/benchmark.py --runs 100 --concurrency 5

# Against Railway deployment
python scripts/benchmark.py \
  --url https://your-service.railway.app \
  --api-key YOUR_KEY \
  --runs 200 \
  --concurrency 10
```

**Typical results on Railway Starter plan (1 vCPU, 512 MB RAM, 50 items):**

```
──────────────────────────────────────────────────
  Results (100/100 successful)
──────────────────────────────────────────────────
  p50 latency :   178 ms
  p95 latency :   412 ms
  p99 latency :   589 ms
  Max latency :   743 ms
  SLA (<2000ms): ✅ PASS  (0/100 breached)
──────────────────────────────────────────────────
```

The HNSW index keeps database query latency under 10 ms for most workloads. Total end-to-end latency is dominated by embedding generation (~100–200 ms on CPU). Upgrade to a Railway Pro plan with more RAM if you need sub-100 ms p99 latency.

---

## Running tests

```bash
pytest tests/ -v
```

---

## Client-ready delivery timeline

This template is designed to be delivered as a production service within **4–5 working days**:

| Day | Work |
|---|---|
| **Day 1** | Repository setup, Supabase project creation, database initialisation (`setup_db.py`), local environment verified end-to-end |
| **Day 2** | Dataset preparation — cleaning and formatting client data into the ingest schema; batch ingestion and search quality review |
| **Day 3** | Railway deployment, environment variable configuration, health check confirmed, benchmark run against production |
| **Day 4** | API key handover, integration guide written for the client's frontend or backend team, example curl commands tested |
| **Day 5** | Buffer for client feedback, search threshold tuning, optional custom metadata filters, final documentation review |

**What the client receives:**

- A live HTTPS endpoint on Railway (or any Docker host)
- A private API key for authenticated access
- This README adapted for their dataset and domain
- A Postman collection or OpenAPI export for their team
- A benchmark report confirming sub-2-second latency SLA

---

## Project structure

```
coda-search-template/
├── app/
│   ├── main.py            # FastAPI app, lifespan, CORS
│   ├── config.py          # Pydantic-settings configuration
│   ├── database.py        # Async SQLAlchemy engine + session
│   ├── embeddings.py      # sentence-transformers wrapper (singleton)
│   ├── models.py          # ORM model + Pydantic schemas
│   └── routes/
│       ├── search.py      # POST /search
│       └── ingest.py      # POST /ingest/item, /batch; DELETE /ingest/{id}
├── scripts/
│   ├── setup_db.py        # One-time DB setup (extension, table, HNSW index)
│   ├── ingest_sample_data.py
│   └── benchmark.py       # Latency benchmark with p50/p95/p99 output
├── data/
│   └── sample_products.json   # 50 generic product descriptions
├── tests/
│   └── test_search.py
├── Dockerfile
├── railway.toml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Licence

MIT. Use freely in client projects.
# coda-search-template
