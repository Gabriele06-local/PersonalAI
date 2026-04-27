import csv
import io
import json
import os
import re
import sqlite3
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from threading import Lock

import fitz
import requests
from chromadb import PersistentClient
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("PERSONAL_AI_DATA_DIR", str(BASE_DIR / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "graph.db"
WEB_DIR = BASE_DIR / "web"

for directory in [DATA_DIR, UPLOAD_DIR, CHROMA_DIR, WEB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


app = FastAPI(title="Personal AI MVP v1.5", version="1.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

embed_model: SentenceTransformer | None = None
collection = None
_model_lock = Lock()


def get_collection():
    global collection
    if collection is None:
        chroma_client = PersistentClient(path=str(CHROMA_DIR))
        collection = chroma_client.get_or_create_collection(name="personal_ai_docs")
    return collection


def get_embed_model() -> SentenceTransformer:
    global embed_model
    if embed_model is not None:
        return embed_model
    with _model_lock:
        if embed_model is None:
            embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return embed_model


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                strength REAL NOT NULL,
                doc_source TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


init_db()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_pdf_text(file_path: Path) -> str:
    text_parts: list[str] = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()


def read_txt_text(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore").strip()


def chunk_text(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(cleaned):
        chunk = cleaned[start : start + size]
        chunks.append(chunk)
        start += step
    return chunks


def extract_graph_llm(text: str) -> dict[str, Any]:
    prompt = textwrap.dedent(
        f"""
        Estrai un knowledge graph dal testo.
        Rispondi SOLO con JSON valido, senza markdown.
        Schema:
        {{
          "entities": [{{"name":"string","type":"string"}}],
          "relations": [{{"source":"string","target":"string","strength":0.0}}]
        }}
        Regole:
        - strength tra 0.1 e 1.0
        - evita duplicati
        - max 12 entities, max 18 relations
        Testo:
        {text[:5000]}
        """
    ).strip()

    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    raw = response.json().get("response", "").strip()
    # Tenta parse diretto, poi fallback estrazione blocco JSON.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise
        return json.loads(match.group(0))


def extract_graph_fallback(text: str) -> dict[str, Any]:
    # Fallback semplice quando LLM non e disponibile.
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9_\-]{2,}\b", text)
    unique = []
    for item in candidates:
        if item not in unique:
            unique.append(item)
    unique = unique[:10]
    entities = [{"name": token, "type": "Concept"} for token in unique]
    relations = []
    for i in range(len(unique) - 1):
        relations.append({"source": unique[i], "target": unique[i + 1], "strength": 0.5})
    return {"entities": entities, "relations": relations}


def normalize_graph_payload(payload: dict[str, Any]) -> dict[str, Any]:
    entities = payload.get("entities", [])
    relations = payload.get("relations", [])

    clean_entities = []
    seen_entities = set()
    for entity in entities:
        name = str(entity.get("name", "")).strip()
        ent_type = str(entity.get("type", "Concept")).strip() or "Concept"
        key = (name.lower(), ent_type.lower())
        if not name or key in seen_entities:
            continue
        seen_entities.add(key)
        clean_entities.append({"name": name, "type": ent_type})

    clean_relations = []
    seen_rel = set()
    for relation in relations:
        source = str(relation.get("source", "")).strip()
        target = str(relation.get("target", "")).strip()
        try:
            strength = float(relation.get("strength", 0.5))
        except (TypeError, ValueError):
            strength = 0.5
        strength = max(0.1, min(1.0, strength))
        key = (source.lower(), target.lower())
        if not source or not target or key in seen_rel:
            continue
        seen_rel.add(key)
        clean_relations.append({"source": source, "target": target, "strength": strength})

    return {"entities": clean_entities, "relations": clean_relations}


def save_graph(doc_source: str, graph_data: dict[str, Any]) -> None:
    timestamp = utc_now_iso()
    conn = sqlite3.connect(DB_PATH)
    try:
        for entity in graph_data["entities"]:
            conn.execute(
                "INSERT INTO entities (name, type, source, timestamp) VALUES (?, ?, ?, ?)",
                (entity["name"], entity["type"], doc_source, timestamp),
            )
        for relation in graph_data["relations"]:
            conn.execute(
                "INSERT INTO relations (source, target, strength, doc_source, timestamp) VALUES (?, ?, ?, ?, ?)",
                (
                    relation["source"],
                    relation["target"],
                    relation["strength"],
                    doc_source,
                    timestamp,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def build_graph_context(question: str, limit: int = 10) -> list[dict[str, Any]]:
    tokens = [token.lower() for token in re.findall(r"\w+", question) if len(token) > 2]
    if not tokens:
        return []

    placeholders = " OR ".join(["lower(source) LIKE ? OR lower(target) LIKE ?"] * len(tokens))
    params: list[Any] = []
    for token in tokens:
        like_token = f"%{token}%"
        params.extend([like_token, like_token])

    sql = (
        f"SELECT source, target, strength, doc_source, timestamp FROM relations "
        f"WHERE {placeholders} ORDER BY strength DESC, id DESC LIMIT ?"
    )
    params.append(limit)

    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    return [
        {
            "source": row[0],
            "target": row[1],
            "strength": row[2],
            "doc_source": row[3],
            "timestamp": row[4],
        }
        for row in rows
    ]


def llm_answer(question: str, rag_contexts: list[str], graph_context: list[dict[str, Any]]) -> str:
    prompt = textwrap.dedent(
        f"""
        Sei un assistente AI locale. Rispondi in italiano in modo pratico.
        Usa solo le informazioni disponibili nei contesti.

        Domanda:
        {question}

        Contesto RAG:
        {chr(10).join([f"- {c}" for c in rag_contexts])}

        Contesto Graph:
        {json.dumps(graph_context, ensure_ascii=True)}
        """
    ).strip()
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    return response.json().get("response", "").strip()


@app.get("/")
def root() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model": OLLAMA_MODEL}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(status_code=400, detail="Supportati solo PDF e TXT")

    file_id = str(uuid.uuid4())
    out_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    content = await file.read()
    out_path.write_bytes(content)

    if suffix == ".pdf":
        full_text = read_pdf_text(out_path)
    else:
        full_text = read_txt_text(out_path)

    if not full_text:
        raise HTTPException(status_code=400, detail="File vuoto o non leggibile")

    chunks = chunk_text(full_text, size=700, overlap=120)
    if not chunks:
        raise HTTPException(status_code=400, detail="Nessun chunk generato")

    local_embed_model = get_embed_model()
    local_collection = get_collection()
    embeddings = local_embed_model.encode(chunks).tolist()
    ids = [f"{file_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file.filename, "chunk_index": i, "timestamp": utc_now_iso()} for i in range(len(chunks))]
    local_collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

    try:
        graph_raw = extract_graph_llm(full_text)
    except Exception:
        graph_raw = extract_graph_fallback(full_text)

    graph = normalize_graph_payload(graph_raw)
    save_graph(file.filename or out_path.name, graph)

    return JSONResponse(
        {
            "ok": True,
            "file": file.filename,
            "chunks": len(chunks),
            "entities": len(graph["entities"]),
            "relations": len(graph["relations"]),
        }
    )


@app.post("/query")
def query(req: QueryRequest) -> dict[str, Any]:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Domanda vuota")

    local_collection = get_collection()
    rag = local_collection.query(query_texts=[question], n_results=max(1, min(10, req.top_k)))
    rag_docs = rag.get("documents", [[]])[0]
    rag_meta = rag.get("metadatas", [[]])[0]
    rag_contexts = [doc for doc in rag_docs if isinstance(doc, str)]

    graph_context = build_graph_context(question, limit=12)

    try:
        answer = llm_answer(question, rag_contexts, graph_context)
    except Exception:
        answer = (
            "LLM non raggiungibile: ecco il contesto disponibile.\n"
            f"RAG: {len(rag_contexts)} chunk trovati.\n"
            f"Graph: {len(graph_context)} relazioni trovate."
        )

    return {
        "question": question,
        "answer": answer,
        "rag_context": [{"text": rag_docs[i], "meta": rag_meta[i]} for i in range(min(len(rag_docs), len(rag_meta)))],
        "graph_context": graph_context,
    }


@app.get("/graph")
def graph() -> dict[str, Any]:
    conn = sqlite3.connect(DB_PATH)
    try:
        entities = conn.execute(
            "SELECT id, name, type, source, timestamp FROM entities ORDER BY id DESC LIMIT 300"
        ).fetchall()
        relations = conn.execute(
            "SELECT source, target, strength, doc_source, timestamp FROM relations ORDER BY id DESC LIMIT 500"
        ).fetchall()
    finally:
        conn.close()

    return {
        "entities": [
            {"id": row[0], "name": row[1], "type": row[2], "source": row[3], "timestamp": row[4]} for row in entities
        ],
        "relations": [
            {
                "source": row[0],
                "target": row[1],
                "strength": row[2],
                "doc_source": row[3],
                "timestamp": row[4],
            }
            for row in relations
        ],
    }


@app.get("/export/{kind}")
def export(kind: str) -> Any:
    if kind not in {"json", "csv"}:
        raise HTTPException(status_code=400, detail="Formato supportato: json|csv")

    graph_data = graph()
    if kind == "json":
        return graph_data

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["source", "target", "strength", "doc_source", "timestamp"])
    writer.writeheader()
    for row in graph_data["relations"]:
        writer.writerow(row)
    return JSONResponse({"csv": output.getvalue()})

