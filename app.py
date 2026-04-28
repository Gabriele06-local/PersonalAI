import csv
import io
import json
import os
import re
import sqlite3
import textwrap
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from threading import Lock

import fitz
import requests
from chromadb import PersistentClient
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
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
FREE_UPLOAD_LIMIT = int(os.getenv("FREE_UPLOAD_LIMIT", "10"))
GUMROAD_UPGRADE_URL = os.getenv("GUMROAD_UPGRADE_URL", "https://gumroad.com/personalai")
UPGRADE_SECRET = os.getenv("UPGRADE_SECRET", "local-dev-upgrade-secret")
PRO_STATUS_PATH = DATA_DIR / "pro_status.json"


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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_paths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                start_node TEXT NOT NULL,
                end_node TEXT NOT NULL,
                path_json TEXT NOT NULL,
                user_id INTEGER,
                tag TEXT,
                pinned INTEGER NOT NULL DEFAULT 0,
                created TEXT NOT NULL
            )
            """
        )
        # Migrazione leggera per installazioni gia esistenti.
        try:
            conn.execute("ALTER TABLE saved_paths ADD COLUMN tag TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE saved_paths ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        conn.commit()
    finally:
        conn.close()


init_db()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


class UpgradeRequest(BaseModel):
    enabled: bool = True


class SavePathRequest(BaseModel):
    start: str
    end: str
    path: list[str]
    user_id: int | None = None
    name: str | None = None
    tag: str | None = None
    pinned: bool = False


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_pro_status() -> bool:
    if not PRO_STATUS_PATH.exists():
        return False
    try:
        payload = json.loads(PRO_STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("enabled", False))


def set_pro_status(enabled: bool) -> None:
    PRO_STATUS_PATH.write_text(
        json.dumps({"enabled": enabled, "updated_at": utc_now_iso()}, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def get_upload_count() -> int:
    files = [p for p in UPLOAD_DIR.glob("*") if p.is_file()]
    return len(files)


def get_usage_summary() -> dict[str, Any]:
    current_uploads = get_upload_count()
    pro_enabled = get_pro_status()
    can_upload = pro_enabled or current_uploads < FREE_UPLOAD_LIMIT
    return {
        "plan": "pro" if pro_enabled else "free",
        "pro_enabled": pro_enabled,
        "uploads_count": current_uploads,
        "uploads_limit": None if pro_enabled else FREE_UPLOAD_LIMIT,
        "remaining_uploads": None if pro_enabled else max(0, FREE_UPLOAD_LIMIT - current_uploads),
        "usage_ratio": 0.0 if pro_enabled else (current_uploads / max(1, FREE_UPLOAD_LIMIT)),
        "can_upload": can_upload,
        "upgrade_url": GUMROAD_UPGRADE_URL,
    }


def make_graph_node_id(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return base or "node"


def normalize_entity_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def build_node_map(entity_rows: list[tuple[str, str]], relation_rows: list[tuple[str, str]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    node_map: dict[str, str] = {}
    type_map: dict[str, str] = {name: ent_type for (name, ent_type) in entity_rows}
    nodes: list[dict[str, Any]] = []

    all_names: list[str] = []
    for (name, _) in entity_rows:
        if name not in all_names:
            all_names.append(name)
    for (source, target) in relation_rows:
        if source not in all_names:
            all_names.append(source)
        if target not in all_names:
            all_names.append(target)

    used_ids: set[str] = set()
    for name in all_names:
        candidate = make_graph_node_id(name)
        unique_id = candidate
        suffix = 2
        while unique_id in used_ids:
            unique_id = f"{candidate}-{suffix}"
            suffix += 1
        used_ids.add(unique_id)
        node_map[name] = unique_id
        nodes.append(
            {
                "id": unique_id,
                "data": {"label": name, "type": type_map.get(name, "Concept")},
            }
        )
    return nodes, node_map


@app.middleware("http")
async def freemium_limit_middleware(request: Request, call_next):
    if request.url.path == "/upload" and request.method.upper() == "POST":
        usage = get_usage_summary()
        if not usage["can_upload"]:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Hai raggiunto il limite Free ({FREE_UPLOAD_LIMIT} upload). Upgrade Pro per sbloccare upload illimitati.",
                    "upgrade_url": GUMROAD_UPGRADE_URL,
                },
            )
    return await call_next(request)


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
    return {"ok": True, "model": OLLAMA_MODEL, "usage": get_usage_summary()}


@app.get("/usage")
def usage() -> dict[str, Any]:
    return get_usage_summary()


@app.post("/upgrade")
def upgrade(payload: UpgradeRequest, request: Request) -> dict[str, Any]:
    if request.headers.get("x-upgrade-secret") != UPGRADE_SECRET:
        raise HTTPException(status_code=401, detail="Secret upgrade non valido")
    set_pro_status(payload.enabled)
    return {"ok": True, "usage": get_usage_summary()}


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


@app.get("/graph/nodes")
def graph_nodes(limit: int = 120) -> list[dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    try:
        entities = conn.execute(
            "SELECT DISTINCT name, type FROM entities ORDER BY id DESC LIMIT ?",
            (max(10, min(limit, 500)),),
        ).fetchall()
        relations = conn.execute(
            "SELECT source, target FROM relations ORDER BY id DESC LIMIT ?",
            (max(20, min(limit * 2, 1000)),),
        ).fetchall()
    finally:
        conn.close()
    nodes, _ = build_node_map(entities, relations)
    return nodes


@app.get("/graph/edges")
def graph_edges(limit: int = 200) -> list[dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    try:
        entities = conn.execute(
            "SELECT DISTINCT name, type FROM entities ORDER BY id DESC LIMIT ?",
            (max(10, min(limit, 500)),),
        ).fetchall()
        relations = conn.execute(
            "SELECT source, target, strength FROM relations ORDER BY id DESC LIMIT ?",
            (max(20, min(limit, 1000)),),
        ).fetchall()
    finally:
        conn.close()

    simple_rel = [(row[0], row[1]) for row in relations]
    _, node_map = build_node_map(entities, simple_rel)
    edges = []
    for idx, (source_name, target_name, strength) in enumerate(relations):
        source_id = node_map.get(source_name)
        target_id = node_map.get(target_name)
        if not source_id or not target_id:
            continue
        edges.append(
            {
                "id": f"e-{idx}-{source_id}-{target_id}",
                "source": source_id,
                "target": target_id,
                "label": f"{float(strength):.2f}",
                "strength": float(strength),
            }
        )
    return edges


@app.get("/graph/path")
def graph_path(start: str, end: str, undirected: bool = False) -> dict[str, Any]:
    start_norm = normalize_entity_name(start)
    end_norm = normalize_entity_name(end)
    if not start_norm or not end_norm:
        raise HTTPException(status_code=400, detail="Parametri start/end obbligatori")

    conn = sqlite3.connect(DB_PATH)
    try:
        relations = conn.execute(
            "SELECT source, target FROM relations ORDER BY id DESC LIMIT 5000"
        ).fetchall()
        entities = conn.execute(
            "SELECT DISTINCT name, type FROM entities ORDER BY id DESC LIMIT 1500"
        ).fetchall()
    finally:
        conn.close()

    # Mappa normalizzata -> display originale.
    display_name: dict[str, str] = {}
    for name, _ in entities:
        key = normalize_entity_name(name)
        if key and key not in display_name:
            display_name[key] = name
    for source, target in relations:
        source_key = normalize_entity_name(source)
        target_key = normalize_entity_name(target)
        if source_key and source_key not in display_name:
            display_name[source_key] = source
        if target_key and target_key not in display_name:
            display_name[target_key] = target

    if start_norm not in display_name or end_norm not in display_name:
        return {
            "path": [],
            "node_ids": [],
            "edge_pairs": [],
            "message": "Entita non trovata nel graph",
            "start": start,
            "end": end,
        }

    adjacency: dict[str, set[str]] = {}
    for source, target in relations:
        src = normalize_entity_name(source)
        tgt = normalize_entity_name(target)
        if not src or not tgt:
            continue
        adjacency.setdefault(src, set()).add(tgt)
        if undirected:
            adjacency.setdefault(tgt, set()).add(src)

    queue = deque([(start_norm, [start_norm])])
    visited = {start_norm}
    found_path: list[str] | None = None

    while queue:
        current, path = queue.popleft()
        if current == end_norm:
            found_path = path
            break
        for nxt in adjacency.get(current, set()):
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, path + [nxt]))

    if not found_path:
        return {
            "path": [],
            "node_ids": [],
            "edge_pairs": [],
            "message": "Nessun percorso trovato",
            "start": display_name[start_norm],
            "end": display_name[end_norm],
        }

    # Costruzione IDs coerenti con /graph/nodes.
    relation_pairs = [(row[0], row[1]) for row in relations]
    _, node_map = build_node_map(entities, relation_pairs)
    node_ids = [node_map.get(display_name[n], "") for n in found_path]
    edge_pairs: list[dict[str, str]] = []
    for i in range(len(found_path) - 1):
        src_name = display_name[found_path[i]]
        tgt_name = display_name[found_path[i + 1]]
        edge_pairs.append(
            {
                "source": node_map.get(src_name, ""),
                "target": node_map.get(tgt_name, ""),
            }
        )

    return {
        "path": [display_name[n] for n in found_path],
        "node_ids": [nid for nid in node_ids if nid],
        "edge_pairs": [pair for pair in edge_pairs if pair["source"] and pair["target"]],
        "hops": max(0, len(found_path) - 1),
        "start": display_name[start_norm],
        "end": display_name[end_norm],
        "undirected": undirected,
    }


@app.post("/path/save")
def save_path_frequent(payload: SavePathRequest) -> dict[str, Any]:
    start = payload.start.strip()
    end = payload.end.strip()
    if not start or not end:
        raise HTTPException(status_code=400, detail="start/end obbligatori")
    if not payload.path:
        raise HTTPException(status_code=400, detail="path non puo essere vuoto")

    name = payload.name.strip() if payload.name else f"{start} -> {end}"
    tag = payload.tag.strip().lower() if payload.tag else None
    pinned = 1 if payload.pinned else 0
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(
            """
            INSERT INTO saved_paths (name, start_node, end_node, path_json, user_id, tag, pinned, created)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                start,
                end,
                json.dumps(payload.path, ensure_ascii=True),
                payload.user_id,
                tag,
                pinned,
                utc_now_iso(),
            ),
        )
        conn.commit()
        saved_id = cursor.lastrowid
    finally:
        conn.close()
    return {"ok": True, "id": saved_id}


@app.get("/paths/presets")
def get_preset_paths(user_id: int | None = None, limit: int = 10) -> dict[str, Any]:
    defaults = [
        {"name": "Startup Journey", "start": "Startup", "end": "Funding", "kind": "default", "tag": "startup", "pinned": True},
        {
            "name": "Compliance IVA -> Sanzione",
            "start": "IVA",
            "end": "Sanzione",
            "kind": "default",
            "tag": "compliance",
            "pinned": True,
        },
    ]

    conn = sqlite3.connect(DB_PATH)
    try:
        bounded_limit = max(1, min(limit, 50))
        if user_id is None:
            rows = conn.execute(
                """
                SELECT id, name, start_node, end_node, tag, pinned, created
                FROM saved_paths
                ORDER BY pinned DESC, id DESC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, name, start_node, end_node, tag, pinned, created
                FROM saved_paths
                WHERE user_id = ?
                ORDER BY pinned DESC, id DESC
                LIMIT ?
                """,
                (user_id, bounded_limit),
            ).fetchall()
    finally:
        conn.close()

    saved = [
        {
            "id": row[0],
            "name": row[1] or f"{row[2]} -> {row[3]}",
            "start": row[2],
            "end": row[3],
            "tag": row[4] or "custom",
            "pinned": bool(row[5]),
            "created": row[6],
            "kind": "saved",
        }
        for row in rows
    ]
    allowed_saved = max(0, max(1, min(limit, 50)) - len(defaults))
    return {"presets": defaults + saved[:allowed_saved]}


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

