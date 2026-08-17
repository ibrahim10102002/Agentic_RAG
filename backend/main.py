import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from retriever import load_retriever_components
from agent import run

components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading components...")
    components.update(load_retriever_components())
    print("✓ Server ready")
    yield
    components.clear()

app = FastAPI(
    title="Agentic RAG API",
    description="SEC 10-K research assistant with dynamic routing and self-correction",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query:        str
    answer:       str
    cited_chunks: list
    sources:      list
    confidence:   dict
    trace:        list
    elapsed_sec:  float
    attempts:     int


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "Agentic RAG API is running"}


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": len(components) > 0,
        "chunks_indexed": len(components.get("chunks", [])),
    }


@app.get("/companies")
def companies():
    """Returns the list of indexed companies — useful for the frontend UI."""
    return {
        "companies": [
            {"name": "Apple",      "ticker": "AAPL"},
            {"name": "Microsoft",  "ticker": "MSFT"},
            {"name": "Alphabet",   "ticker": "GOOGL"},
            {"name": "Meta",       "ticker": "META"},
            {"name": "Nvidia",     "ticker": "NVDA"},
            {"name": "Amazon",     "ticker": "AMZN"},
            {"name": "Tesla",      "ticker": "TSLA"},
            {"name": "Netflix",    "ticker": "NFLX"},
            {"name": "Salesforce", "ticker": "CRM"},
            {"name": "AMD",        "ticker": "AMD"},
        ]
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Full agentic RAG pipeline:
      1. Route — classify query type and section filter
      2. Reformulate — rewrite weak queries
      3. Retrieve — hybrid search with metadata filtering
      4. Confidence check — retry if results are thin
      5. Generate — cited answer over confident results

    Returns the answer plus the full agent trace so the
    frontend can show every decision the agent made.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if len(request.query) > 500:
        raise HTTPException(status_code=400, detail="Query too long (max 500 chars)")

    try:
        result = run(request.query, components)
        return QueryResponse(**result)
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))