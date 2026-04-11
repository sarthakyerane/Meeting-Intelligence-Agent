# Meeting Intelligence Agent

> Agentic meeting analysis with cross-meeting memory, semantic search, and Claude Desktop integration via MCP.

## Architecture

```
Claude Desktop (MCP Client)
        │
        ▼
MCP Server (port 8001)          ← 5 tools: upload, search, action items, contradictions, history
        │ HTTP
        ▼
FastAPI Backend (port 8000)
        │
        ├── faster-whisper      ← audio → text
        ├── LLM fallback chain  ← Gemini → Groq → Ollama
        ├── Redis               ← semantic cache (cosine sim in-memory)
        ├── ChromaDB            ← embeddings (semantic search + contradictions)
        └── MySQL               ← structured storage
```

## Quick Start

### 1. Prerequisites
- Docker Desktop
- A Gemini API key from [aistudio.google.com](https://aistudio.google.com)
- (Optional) Groq API key from [console.groq.com](https://console.groq.com) — free
- (Optional) Ollama running locally for offline fallback

### 2. Setup
```bash
cd c:\Meeting
copy .env.example .env
# Edit .env and add your API keys
notepad .env
```

### 3. Run
```bash
docker compose up --build -d
```

All services start automatically. The backend will be ready at `http://localhost:8000`.

### 4. Verify
```bash
# Health check
curl http://localhost:8000/health

# API docs (interactive Swagger UI)
# Open in browser: http://localhost:8000/docs
```

---

## API Endpoints

### Upload a Meeting
```bash
# Raw text
curl -X POST http://localhost:8000/meetings/upload \
  -F "title=Sprint Planning Q2" \
  -F "project=ProjectAlpha" \
  -F "transcript=John said we should use PostgreSQL. Sarah said MySQL is better. Tom will set up the CI pipeline by Friday."

# Audio file
curl -X POST http://localhost:8000/meetings/upload \
  -F "title=Design Review" \
  -F "project=ProjectAlpha" \
  -F "audio=@meeting.mp3"
```

Response includes `X-LLM-Provider` header showing which AI provider was used.

### Search Decisions (with Semantic Cache)
```bash
# First call: CACHE MISS (~2000ms)
curl "http://localhost:8000/decisions/search?q=database+choice"

# Second call: CACHE HIT (~40ms)
curl "http://localhost:8000/decisions/search?q=which+database+did+we+pick"
```

### Get Action Items
```bash
curl "http://localhost:8000/action-items?owner=Tom&status=pending"
```

### Cross-Meeting Contradictions
```bash
curl "http://localhost:8000/contradictions?project=ProjectAlpha"
```

### Project History
```bash
curl "http://localhost:8000/project/ProjectAlpha/history"
```

### Run Eval (Quality Metrics)
```bash
curl -X POST http://localhost:8000/eval/run | python -m json.tool
```

---

## Claude Desktop Integration (MCP)

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "meeting-intelligence": {
      "command": "python",
      "args": ["C:/Meeting/mcp_server/server.py"],
      "env": {
        "BACKEND_URL": "http://localhost:8000"
      }
    }
  }
}
```

> **Note:** Install MCP server deps first:
> ```bash
> cd c:\Meeting\mcp_server
> pip install -r requirements.txt
> ```

Then in Claude Desktop you can ask:
- *"Search past decisions about the database"*
- *"What are Tom's open action items?"*
- *"Are there any contradictions in ProjectAlpha?"*
- *"Summarize the history of ProjectAlpha"*

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |
| `GROQ_API_KEY` | Optional | Groq API key (free, fast fallback) |
| `OLLAMA_BASE_URL` | Optional | Ollama instance URL (offline fallback) |
| `MYSQL_*` | ✅ | MySQL connection settings |
| `REDIS_URL` | ✅ | Redis connection URL |
| `CHROMA_HOST/PORT` | ✅ | ChromaDB connection |

---

## Technical Notes

### LLM Fallback Chain
The system tries each provider in order: **Gemini 2.0 Flash → Groq (llama-3.3-70b) → Ollama (llama3)**. 
Every response includes an `X-LLM-Provider` header. Only if all three fail does the request error.

### Semantic Cache
Uses Redis with in-memory cosine similarity (numpy). Threshold: 0.92. TTL: 24h.

> **Known limitation:** Similarity is computed in Python over all cached vectors (O(n)). 
> Fine for demos up to ~200 cached queries. Production upgrade: Redis Stack with HNSW vector search.

### Speaker Diarization
Planned for Week 2 (pyannote.audio). Week 1 produces plain text transcripts.

---

## Project Structure
```
c:\Meeting\
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       ├── config.py
│       ├── database.py
│       ├── models.py
│       ├── schemas.py
│       ├── routers/
│       │   ├── meetings.py       # upload, retrieve, analyze
│       │   ├── decisions.py      # list, semantic search + cache
│       │   ├── action_items.py   # list, update status
│       │   ├── intelligence.py   # contradictions, project history
│       │   └── eval.py           # ground-truth evaluation
│       ├── services/
│       │   ├── agent.py          # 4-step agentic pipeline
│       │   ├── llm_service.py    # Gemini → Groq → Ollama chain
│       │   ├── whisper_service.py# faster-whisper audio → text
│       │   ├── chroma_service.py # embeddings + semantic search
│       │   └── cache_service.py  # Redis semantic cache
│       └── tests/
│           └── eval_fixtures.json
└── mcp_server/
    ├── Dockerfile
    ├── requirements.txt
    └── server.py                 # 5 MCP tools
```
