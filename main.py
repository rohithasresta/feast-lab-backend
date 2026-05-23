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

from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

# ─────────────────────────────────────────────

# CONFIG

# ─────────────────────────────────────────────

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

EMBED_MODEL = "text-embedding-3-small"

CHAT_MODEL  = "gpt-4o-mini"

CHUNK_SIZE    = 400

CHUNK_OVERLAP = 50

TOP_K     = 4

EMBED_DIM = 1536

DOCS_FOLDER = Path("docs")

client = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────────

# VECTOR STORE

# ─────────────────────────────────────────────

@dataclass

class VectorStore:

    index: Any = field(

        default_factory=lambda: faiss.IndexFlatIP(EMBED_DIM)

    )

    metadata: List[Dict] = field(default_factory=list)

    def add(

        self,

        text: str,

        embedding: List[float],

        source: str,

        chunk_idx: int

    ):

        vec = np.array([embedding], dtype=np.float32)

        faiss.normalize_L2(vec)

        self.index.add(vec)

        self.metadata.append({

            "text": text,

            "source": source,

            "chunk_index": chunk_idx

        })

    def search(

        self,

        query_embedding: List[float],

        top_k: int = TOP_K

    ):

        if self.index.ntotal == 0:

            return []

        vec = np.array([query_embedding], dtype=np.float32)

        faiss.normalize_L2(vec)

        scores, indices = self.index.search(

            vec,

            min(top_k, self.index.ntotal)

        )

        return [

            {

                **self.metadata[idx],

                "score": float(score)

            }

            for score, idx in zip(scores[0], indices[0])

            if idx >= 0

        ]

    @property

    def total_chunks(self):

        return self.index.ntotal

    def sources(self):

        return list(

            set(m["source"] for m in self.metadata)

        )

vs = VectorStore()

# ─────────────────────────────────────────────

# HELPERS

# ─────────────────────────────────────────────

def extract_text(path: Path):

    if path.suffix.lower() == ".pdf":

        doc = fitz.open(str(path))

        text = "".join(

            page.get_text() + "\n"

            for page in doc

        )

        doc.close()

        return text

    return path.read_text(

        encoding="utf-8",

        errors="ignore"

    )

def chunk_text(text: str):

    words = text.split()

    chunks = []

    i = 0

    while i < len(words):

        chunk = " ".join(

            words[i:i + CHUNK_SIZE]

        )

        if len(chunk.strip()) > 30:

            chunks.append(chunk)

        i += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

def get_embedding(text: str):

    response = client.embeddings.create(

        model=EMBED_MODEL,

        input=text

    )

    return response.data[0].embedding

def load_all_docs():

    if not DOCS_FOLDER.exists():

        print("docs folder not found")

        return

    files = (

        list(DOCS_FOLDER.glob("*.pdf")) +

        list(DOCS_FOLDER.glob("*.txt"))

    )

    if not files:

        print("No docs found")

        return

    print(f"Loading {len(files)} documents")

    for path in files:

        try:

            text = extract_text(path)

            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):

                embedding = get_embedding(chunk)

                vs.add(

                    chunk,

                    embedding,

                    path.name,

                    i

                )

            print(f"Loaded {path.name}")

        except Exception as e:

            print(e)

# ─────────────────────────────────────────────

# FASTAPI

# ─────────────────────────────────────────────

app = FastAPI(

    title="FEAST Lab API"

)

# STATIC FILES

app.mount(

    "/static",

    StaticFiles(directory="static"),

    name="static"

)

# CORS

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_methods=["*"],

    allow_headers=["*"],

)

# ─────────────────────────────────────────────

# MODELS

# ─────────────────────────────────────────────

class ChatRequest(BaseModel):

    message: str

    history: List[Dict] = []

class ChatResponse(BaseModel):

    answer: str

    sources: List[str]

    chunks_used: List[Dict]

# ─────────────────────────────────────────────

# STARTUP

# ─────────────────────────────────────────────

@app.on_event("startup")

async def startup():

    load_all_docs()

# ─────────────────────────────────────────────

# WEBSITE

# ─────────────────────────────────────────────

@app.get("/")

async def homepage():

    return FileResponse("static/index.html")

# ─────────────────────────────────────────────

# STATUS

# ─────────────────────────────────────────────

@app.get("/status")

async def status():

    return {

        "ready": vs.total_chunks > 0,

        "documents": vs.sources(),

        "total_chunks": vs.total_chunks

    }

# ─────────────────────────────────────────────

# CHAT

# ─────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)

async def chat(req: ChatRequest):

    if vs.total_chunks == 0:

        raise HTTPException(

            status_code=503,

            detail="Vector store empty"

        )

    query_embedding = get_embedding(req.message)

    results = vs.search(

        query_embedding,

        top_k=TOP_K

    )

    context = "\n\n---\n\n".join(

        f"[Source: {r['source']}]\n{r['text']}"

        for r in results

    )

    system_prompt = f"""

You are the FEAST Lab assistant.

Answer only using the provided context.

Context:

{context}

"""

    messages = [

        {

            "role": "system",

            "content": system_prompt

        }

    ]

    messages += req.history[-6:]

    messages.append({

        "role": "user",

        "content": req.message

    })

    response = client.chat.completions.create(

        model=CHAT_MODEL,

        messages=messages,

        max_tokens=500,

        temperature=0.3

    )

    return ChatResponse(

        answer=response.choices[0].message.content,

        sources=list(

            set(r["source"] for r in results)

        ),

        chunks_used=results

    )