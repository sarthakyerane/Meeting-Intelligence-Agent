from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.database import init_db
from app.routers import meetings, decisions, action_items, intelligence, eval as eval_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initializing database tables...")
    init_db()
    logger.info("Database ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Meeting Intelligence Agent",
    description=(
        "Agentic meeting analysis: extract decisions, action items, conflicts, "
        "and unresolved questions. Cross-meeting contradiction detection. "
        "Semantic search with Redis cache. LLM fallback chain: Gemini → Groq → Ollama."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meetings.router)
app.include_router(decisions.router)
app.include_router(action_items.router)
app.include_router(intelligence.router)
app.include_router(eval_router.router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "service": "meeting-intelligence-backend"}
