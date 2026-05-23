"""
FEAST Lab RAG Backend
Built by Rohitha Sresta Ganji
"""

import os
import faiss
import fitz
import numpy as np

from openai import OpenAI
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMBED_MODEL    = "text-embedding-3-small"
CHAT_MODEL     = "gpt-4o-mini"
CHUNK_SIZE     = 400
CHUNK_OVERLAP  = 50
TOP_K          = 4
EMBED_DIM      = 1536
DOCS_FOLDER    = Path("docs")

client = OpenAI(api_key=OPENAI_API_KEY)

# ── FAISS Vector Store ────────────────────────────────────────────────────
@dataclass
class VectorStore:
    index: Any = field(default_factory=lambda: faiss.IndexFlatIP(EMBED_DIM))
    metadata: List[Dict] = field(default_factory=list)

    def add(self, text: str, embedding: List[float], source: str, chunk_idx: int):
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.metadata.append({"text": text, "source": source, "chunk_index": chunk_idx})

    def search(self, query_embedding: List[float], top_k: int = TOP_K) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, min(top_k, self.index.ntotal))
        return [
            {**self.metadata[idx], "score": float(score)}
            for score, idx in zip(scores[0], indices[0]) if idx >= 0
        ]

    @property
    def total_chunks(self):
        return self.index.ntotal

    def sources(self):
        return list(set(m["source"] for m in self.metadata))

vs = VectorStore()

# ── Text helpers ──────────────────────────────────────────────────────────
def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(str(path))
        text = "".join(page.get_text() + "\n" for page in doc)
        doc.close()
        return text
    return path.read_text(encoding="utf-8", errors="ignore")

def chunk_text(text: str) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + CHUNK_SIZE])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding

def load_all_docs():
    if not DOCS_FOLDER.exists():
        print(f"⚠️  docs/ folder not found")
        return
    files = list(DOCS_FOLDER.glob("*.pdf")) + list(DOCS_FOLDER.glob("*.txt"))
    if not files:
        print("⚠️  No files found in docs/")
        return
    print(f"📂 Loading {len(files)} document(s)...")
    for path in files:
        print(f"   Processing: {path.name}")
        try:
            text = extract_text(path)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)
                vs.add(chunk, embedding, path.name, i)
            print(f"   ✓ {path.name} — {len(chunks)} chunks ({vs.total_chunks} total)")
        except Exception as e:
            print(f"   ✗ Error on {path.name}: {e}")
    print(f"\n✅ Ready — {vs.total_chunks} vectors across {len(vs.sources())} doc(s)")

# ── FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(title="FEAST Lab RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks_used: List[Dict]

@app.on_event("startup")
async def startup():
    load_all_docs()

# ── Homepage route ────────────────────────────────────────────────────────
@app.get("/")
async def homepage():
    return FileResponse("static/index.html")

# ── API routes ────────────────────────────────────────────────────────────
@app.get("/status")
async def status():
    return {
        "total_chunks": vs.total_chunks,
        "documents": vs.sources(),
        "ready": vs.total_chunks > 0
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if vs.total_chunks == 0:
        raise HTTPException(status_code=503, detail="Vector store empty.")

    query_embedding = get_embedding(req.message)
    results = vs.search(query_embedding, top_k=TOP_K)

    context = "\n\n---\n\n".join(
        f"[Source: {r['source']} | Chunk {r['chunk_index']+1} | Similarity: {r['score']:.3f}]\n{r['text']}"
        for r in results
    )

    system_prompt = f"""You are the FEAST Lab assistant at the University of Missouri–Columbia.
FEAST Lab is a nutrition and food science research lab directed by Dr. Kiruba Krishnaswami.

Answer questions using ONLY the retrieved context below. Be concise, friendly, and accurate.
If the answer isn't in the context, say you don't have that specific information and suggest
contacting the lab at feast-lab@missouri.edu or calling (573) 882-4400.

RETRIEVED CONTEXT (top {TOP_K} chunks via FAISS cosine similarity):
{context}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages += req.history[-6:]
    messages.append({"role": "user", "content": req.message})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.3
    )

    return ChatResponse(
        answer=response.choices[0].message.content,
        sources=list(set(r["source"] for r in results)),
        chunks_used=results
    )
