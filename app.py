# app.py
from dotenv import load_dotenv
load_dotenv()
import os
import re
import asyncio
from typing import Optional, List, Dict
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect
import httpx
import math
import base64, zlib
import shutil
import subprocess
import json
import logging
from datetime import datetime


from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

from app_security import require_api_key
from cache import get_cached_answer, cache_answer, semantic_cache_search, semantic_cache_upsert
from database import create_table, add_module, get_modules, delete_module  # Import the database functions


# --- Config ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "kb_chunks")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
TOP_K = int(os.getenv("TOP_K", "6"))

EXTRACTIVE_THRESHOLD = float(os.getenv("EXTRACTIVE_THRESHOLD", "0.85"))
SMALL_MODEL_THRESHOLD = float(os.getenv("SMALL_MODEL_THRESHOLD", "0.65"))
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
ADMIN_INGEST_SCRIPT = os.getenv("ADMIN_INGEST_SCRIPT", "ingest.py")  # path to your ingest script


# CORS / origins
_origins_env = os.getenv("ALLOW_ORIGINS", "")
if _origins_env:
    origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
else:
    origins = []
    
    
# configure logger (simple)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_admin")


# Collections to try deleting (module-specific collections + cache)
COLLECTIONS_TO_TRY = [
    QDRANT_COLLECTION,
    os.getenv("QDRANT_CACHE_COLLECTION", "kb_cache"),
]

# FastAPI init
app = FastAPI(title="RAG Chat (EN/JA)")

@app.on_event("startup")
def startup():
    create_table()  # Ensure the modules table exists
    
@app.get("/admin/modules", response_model=List[str], include_in_schema=False)
async def get_modules_list():
    """
    Get the list of available modules from the database.
    """
    modules = get_modules()  # Fetch modules from the database
    return modules

# API endpoint to add a new module
@app.post("/admin/modules", include_in_schema=False)
async def add_new_module(module_name: str = Form(...)):
    """
    Add a new module to the database.
    """
    # Check if the module already exists
    existing_modules = get_modules()
    if module_name in existing_modules:
        raise HTTPException(status_code=400, detail="Module already exists")
    
    # Add the new module
    add_module(module_name)    
    _append_admin_log(f"Module added: {module_name}")
    _write_admin_status({"pid": None, "status": "module_added", "module": module_name})
    return JSONResponse({"ok": True, "module": module_name})


# API endpoint to delete a module
@app.delete("/admin/modules/{module_name}", include_in_schema=False)
async def delete_existing_module(module_name: str):
    """
    Delete a module from the database.
    """
    modules = get_modules()
    if module_name not in modules:
        raise HTTPException(status_code=404, detail="Module not found")
    
    delete_module(module_name)
    _append_admin_log(f"Module deleted: {module_name}")
    _write_admin_status({"pid": None, "status": "module_deleted", "module": module_name})
    return JSONResponse({"ok": True, "deleted_module": module_name})





app.mount("/static", StaticFiles(directory="public"), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse("public/index.html")

from fastapi.responses import RedirectResponse, HTMLResponse

@app.get("/admin", dependencies=[Depends(require_api_key)], include_in_schema=False)
def admin_panel():
    """
    Protected admin interface — requires X-API-Key header.
    Only accessible to authorized users.
    """
    p = Path("public") / "admin.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="admin.html not found.")
    return FileResponse(str(p))



app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Qdrant + embed model init
qc = QdrantClient(url=QDRANT_URL)
st = SentenceTransformer(EMBED_MODEL)
try:
    EMB_DIM = st.get_sentence_embedding_dimension()
except Exception:
    EMB_DIM = len(st.encode(["test"], normalize_embeddings=True)[0])

E5 = EMBED_MODEL.lower().startswith("intfloat/multilingual-e5")

# Async HTTP client for Ollama
HTTP_CLIENT = httpx.AsyncClient(timeout=300.0)


@app.on_event("shutdown")
async def shutdown_event():
    try:
        await HTTP_CLIENT.aclose()
    except Exception:
        pass


# --- Data models ---
class ChatRequest(BaseModel):
    text: str
    lang: Optional[str] = None
    top_k: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[dict]


# --- Helpers ---
async def embed_query(q: str):
    """
    Run embedding in a thread so we don't block the event loop.
    """
    def _encode(text):
        q_ = ("passage: " + text) if E5 else text
        return st.encode([q_], normalize_embeddings=True)[0].tolist()
    return await asyncio.to_thread(_encode, q)


def detect_lang(s: str) -> str:
    try:
        return detect(s[:4000])
    except Exception:
        if re.search(r'[\u3040-\u30ff\u3400-\u9fff]', s):
            return "ja"
        return "en"


def _decompress_payload_text(b64: str) -> str:
    if not b64:
        return ""
    try:
        compressed = base64.b64decode(b64.encode("ascii"))
        return zlib.decompress(compressed).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def looks_like_ui_fragment(s: str) -> bool:
    if not s or not s.strip():
        return True
    txt = s.strip()

    s_low = txt.lower()
    bad_indicators = ["<div", "<p", "<br", "<!--", "</", "&nbsp;", "&quot;", "&lt;", "&gt;",
                      "http://", "https://", "class=\"", "style=\"", "margin-left", "onclick="]
    for pat in bad_indicators:
        if pat in s_low:
            return True

    ui_phrases = [
        "メッセージ本文（任意）", "ご利用のメールソフト", "推奨画面解像度",
        "管理者サイトで作業", "管理者ガイド", "お礼メールについて"
    ]
    for pat in ui_phrases:
        if pat in txt:
            return True

    # Minimum substantive length: use character count (works for CJK languages)
    plain_chars = re.sub(r"\s+", "", txt)
    if len(plain_chars) < 30:         # <--- tune this (30 chars is reasonable)
        return True

    return False


def load_chunk_text(chunk_path: Optional[str], max_chars: int = 3000) -> str:
    """
    Backwards-compatible loader for chunk_path stored on disk.
    If chunk_path points to an existing file, read it. Otherwise return empty string.
    NOTE: prefer payload['full_text_z'] (decompressed) — build_prompt will try full_text_z first.
    """
    if not chunk_path:
        return ""
    try:
        p = chunk_path
        if not os.path.isabs(p):
            p = os.path.abspath(p)
        with open(p, "r", encoding="utf-8") as fh:
            txt = fh.read()
        return txt[:max_chars]
    except Exception:
        return ""


def build_prompt(question: str, lang: str, hits: List[dict]):
    """
    Build a prompt using up to 10 hits. Prefer payload['full_text_z'] (decompressed),
    fallback to chunk_path on disk, then text_excerpt or legacy 'text'.
    """
    intro = f"You are a helpful assistant. Answer in {('Japanese' if lang.startswith('ja') else 'English')}."
    rules = "Use ONLY the context. If unsure, say you don't know. Cite sources as [1], [2], ..."
    ctx_lines = []
    cited = []

    for i, h in enumerate(hits, start=1):
        payload = h.get("payload", {}) or {}
        # prefer full chunk text stored in payload
        full = ""
        if payload.get("full_text_z"):
            full = _decompress_payload_text(payload.get("full_text_z"))
        if not full and payload.get("chunk_path"):
            full = load_chunk_text(payload.get("chunk_path"))
        if not full:
            full = payload.get("text_excerpt") or payload.get("text") or ""
        snippet = (full[:700] + "...") if len(full) > 700 else full

        title = payload.get("doc_title") or payload.get("title") or payload.get("source_path") or payload.get("url") or ""
        ctx_lines.append(f"[{i}] {snippet}")
        cited.append({
            "n": i,
            "title": title,
            "meta": {
                "url": payload.get("url"),
                "source_path": payload.get("source_path"),
                "chunk_path": payload.get("chunk_path"),
                "page": payload.get("page"),
                "lang": payload.get("lang"),
                "content_hash": payload.get("content_hash") or payload.get("doc_id") or ""
            }
        })

    ctx = "\n".join(ctx_lines[:10])
    prompt = f"""{intro}
{rules}

Question:
{question}

Context:
{ctx}

Answer:"""
    return prompt, cited


OLLAMA_PROMPT_MAX_CHARS = int(os.getenv("OLLAMA_PROMPT_MAX_CHARS", "10000"))


async def call_ollama_async(prompt: str, model: str = None):
    """
    Async call to Ollama with prompt-length guard.
    If prompt is too long, it will be truncated (preserves start + hint).
    """
    model = model or LLM_MODEL
    # Guard / trim
    if OLLAMA_PROMPT_MAX_CHARS and len(prompt) > OLLAMA_PROMPT_MAX_CHARS:
        # Keep beginning and trailing hint so LLM still sees question
        head = prompt[: int(OLLAMA_PROMPT_MAX_CHARS * 0.7)]
        tail = "\n\n[...context truncated...]\n\n" + prompt[-int(OLLAMA_PROMPT_MAX_CHARS * 0.3):]
        prompt = head + tail
        print(f"WARN: prompt truncated to {OLLAMA_PROMPT_MAX_CHARS} chars before sending to Ollama")

    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    body = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    try:
        r = await HTTP_CLIENT.post(url, json=body)
    except httpx.RequestError as e:
        print("ERROR: Ollama request error:", e)
        return f"[Ollama request failed: {e}]"
    if r.status_code >= 400:
        print("ERROR: Ollama returned status", r.status_code, r.text[:1000])
        return f"[Ollama error {r.status_code}: {r.text}]"
    try:
        data = r.json()
    except Exception as e:
        print("ERROR: Ollama returned non-JSON response:", e, r.text[:1000])
        return f"[Ollama returned non-JSON response: {e}]"
    if isinstance(data, dict):
        # normalize common shapes
        return data.get("response", "") or data.get("text", "") or ""
    return str(data)


def cosine_similarity(a: list, b: list) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        ai = float(a[i]); bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


# --- Health endpoint (keeps simple) ---
@app.get("/health")
def health():
    try:
        qc.get_collections()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# --- Chat endpoint ---
@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat(req: ChatRequest):
    # language handling: detect language from text and reconcile with any explicit user choice
    detected = detect_lang(req.text)
    user_lang = (req.lang or detected or "en").strip()[:2].lower()
    if req.lang and req.lang[:2].lower() != detected[:2].lower():
        # prefer detected language (safer) but log the mismatch
        print(f"WARN: user selected lang={req.lang} but detected={detected}. Using detected language.")
        user_lang = detected[:2].lower()

    k = req.top_k or TOP_K

    # 1) exact-match (Redis) cache
    cached = await get_cached_answer(req.text, user_lang)
    if cached:
        print("BRANCH: exact-match cache hit")
        return cached

    # 2) embed (async)
    vec = await embed_query(req.text)

    # 3) semantic cache quick check
    sem_hit = semantic_cache_search(vec, top_k=1, score_threshold=0.78)
    if sem_hit:
        payload = sem_hit.get("payload", {}) or {}
        payload_vec = payload.get("vec")
        cached_lang = (payload.get("lang") or "").lower()
        if payload_vec and cached_lang and cached_lang.startswith(user_lang[:2]):
            cos = cosine_similarity(vec, payload_vec)
            if cos >= 0.82:
                # KB overlap validation
                try:
                    kb_top = qc.search(collection_name=QDRANT_COLLECTION, query_vector=vec, limit=3, with_payload=True)
                    kb_ids = set()
                    for h in kb_top:
                        pid = h.payload.get("content_hash") or h.payload.get("doc_id") or h.payload.get("source_path") or getattr(h, "id", None)
                        if pid:
                            kb_ids.add(str(pid))
                    cached_cited = set()
                    for c in payload.get("citations") or []:
                        meta = c.get("meta") or {}
                        cid = meta.get("content_hash") or meta.get("doc_id") or meta.get("source_path") or c.get("title") or ""
                        if cid:
                            cached_cited.add(str(cid))
                    overlap = 0.0
                    if kb_ids:
                        overlap = len(kb_ids & cached_cited) / float(len(kb_ids))
                    if overlap >= 0.5:
                        print(f"BRANCH: semantic-cache validated (cos={cos:.3f}, overlap={overlap:.2f})")
                        return {"answer": payload.get("answer"), "citations": payload.get("citations", [])}
                except Exception as e:
                    print("WARN: KB validation failed, continuing to retrieval:", e)

    # 4) Standard Qdrant search (same-language first)
    try:
        flt = Filter(must=[FieldCondition(key="lang", match=MatchValue(value="ja" if user_lang.startswith("ja") else "en"))])
        res = qc.search(collection_name=QDRANT_COLLECTION, query_vector=vec, limit=k, with_payload=True, score_threshold=None, query_filter=flt)
        hits = res
        if len(hits) < max(2, k // 2):
            res2 = qc.search(collection_name=QDRANT_COLLECTION, query_vector=vec, limit=k, with_payload=True)
            seen = set(); merged = []
            for h in hits + res2:
                pid = h.payload.get("content_hash") or getattr(h, "id", None)
                if pid not in seen:
                    seen.add(pid); merged.append(h)
            hits = merged[:k]
    except Exception:
        hits = qc.search(collection_name=QDRANT_COLLECTION, query_vector=vec, limit=k, with_payload=True)

    # normalize hits
    hits_d = []
    for h in hits:
        payload = getattr(h, "payload", None) or (h.payload if hasattr(h, "payload") else (h.get("payload") if isinstance(h, dict) else {}))
        score = getattr(h, "score", None) or (h.score if hasattr(h, "score") else (h.get("score") if isinstance(h, dict) else None))
        hits_d.append({"payload": payload or {}, "score": score})

    # Build prompt using full chunk text when available
    prompt, citations = build_prompt(req.text, user_lang, hits_d)
    # after retrieving hits_d and building prompt/citations
    if len(hits_d) > 4:
        # build snippets list from the full chunk text when available
        snippets = []
        for h in hits_d[:6]:
            payload = h.get("payload", {}) or {}
            full = ""
            if payload.get("full_text_z"):
                full = _decompress_payload_text(payload.get("full_text_z"))
            if not full and payload.get("chunk_path"):
                full = load_chunk_text(payload.get("chunk_path"))
            if not full:
                full = payload.get("text_excerpt") or payload.get("text") or ""
            snippets.append(full)
        summary = await summarize_snippets(snippets)
        if summary:
            # build a rapid 'summary' prompt for the big model instead of full context
            prompt = f"You are a helpful assistant. Answer using ONLY the summary below.\n\nSummary:\n{summary}\n\nQuestion:\n{req.text}\n\nAnswer:"
            print("INFO: using summarized context (small-model) to reduce prompt size")

    # Decide routing: if retrieval extremely confident -> return extractive snippet
    top_score = hits_d[0]["score"] if hits_d and hits_d[0].get("score") is not None else 0.0
    print(f"DEBUG: top retrieval score = {top_score:.4f}")
    use_extractive = top_score >= EXTRACTIVE_THRESHOLD and not user_lang.startswith("ja")

    if use_extractive:
        # return extractive snippet (first hit) but validate it first
        top_payload = hits_d[0]["payload"]
        snippet = ""
        if top_payload.get("full_text_z"):
            snippet = _decompress_payload_text(top_payload.get("full_text_z"))
        if not snippet and top_payload.get("chunk_path"):
            snippet = load_chunk_text(top_payload.get("chunk_path"))
        if not snippet:
            snippet = top_payload.get("text_excerpt") or top_payload.get("text") or ""
        snippet = (snippet[:700] + "...") if len(snippet) > 700 else snippet
        citation = citations[:1]

        # debug: show the source of the extractive snippet
        try:
            print("EXTRACTIVE SOURCE:", top_payload.get("doc_title"), top_payload.get("source_path"), top_payload.get("content_hash"))
        except Exception:
            pass

        # Validate snippet: avoid returning UI/html/too-short fragments
        if looks_like_ui_fragment(snippet):
            print("WARN: extractive snippet looks like UI/HTML fragment — skipping extractive return and calling LLM")
            # fall through to model generation branch below
        else:
            # safe to return extractive snippet
            await cache_answer(req.text, user_lang, {"answer": snippet, "citations": citation})
            # upsert semantic cache for this question
            from hashlib import sha1
            cache_id = int(sha1((req.text + user_lang).encode("utf-8")).hexdigest()[:12], 16)
            payload = {"question": req.text, "lang": user_lang, "answer": snippet, "citations": citation}
            try:
                await semantic_cache_upsert(cache_id, vec, payload)
            except Exception as e:
                print("WARN: semantic_cache_upsert failed:", e)
            print("BRANCH: extractive (returned snippet)")
            return {"answer": snippet, "citations": citation}

    # otherwise call model (we use only LLM_MODEL here as you requested; you can add LLM_SMALL later)
    model_to_use = LLM_MODEL
    print(f"BRANCH: calling LLM {model_to_use} for generation. Prompt length: {len(prompt)}")
    answer = await call_ollama_async(prompt, model=model_to_use)

    # cache exact-match and semantic cache upsert
    await cache_answer(req.text, user_lang, {"answer": answer, "citations": citations})
    from hashlib import sha1
    cache_id = int(sha1((req.text + user_lang).encode("utf-8")).hexdigest()[:12], 16)
    sem_payload = {"question": req.text, "lang": user_lang, "answer": answer, "citations": citations}
    try:
        await semantic_cache_upsert(cache_id, vec, sem_payload)
    except Exception as e:
        print("WARN: semantic_cache_upsert failed:", e)

    return {"answer": answer, "citations": citations}


async def summarize_snippets(snippets: List[str], max_chars: int = 1200):
    """
    Simple summarization wrapper: concatenates snippets and asks small model for 3-bullet summary.
    Returns summary string.
    """
    text = "\n\n".join([s[:800] for s in snippets])  # avoid huge payload to small model
    prompt = f"Summarize the following snippets into 3 short bullets (keep factual):\n\n{text}\n\nSummary:"
    # call small model first (faster / cheaper)
    summary = await call_ollama_async(prompt, model=os.getenv("LLM_MODEL_SMALL", LLM_MODEL))
    return summary or ""


from fastapi.responses import HTMLResponse

# --- Admin login + protected content endpoints -----------------------

@app.get("/admin-login", include_in_schema=False)
def admin_login_page():
    """
    Simple login page where user pastes API key. The page validates the key
    via /admin/auth and then fetches /admin/content (with header) and writes it into the DOM.
    """
    html = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Admin Login</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    body{font-family:system-ui,Arial,sans-serif;background:#f6f7fb;padding:30px}
    .card{max-width:640px;margin:0 auto;background:white;padding:20px;border-radius:10px;box-shadow:0 8px 24px rgba(0,0,0,0.06)}
    input[type=text]{width:100%;padding:10px;border:1px solid #ddd;border-radius:8px}
    button{padding:10px 14px;border-radius:8px;border:0;background:#111827;color:white;cursor:pointer}
    .small{font-size:12px;color:#666;margin-top:8px}
  </style>
</head>
<body>
  <div class="card">
    <h2>Admin Login</h2>
    <p>Paste your admin API key below to open the admin interface.</p>
    <input id="key" type="text" placeholder="X-API-Key" />
    <div style="margin-top:12px;">
      <button id="btn">Open Admin</button>
    </div>
    <p class="small">The key is stored in <code>sessionStorage</code> for this tab only.</p>
    <pre id="log" style="margin-top:12px;background:#f3f4f6;padding:8px;border-radius:6px;max-height:200px;overflow:auto"></pre>
  </div>

<script>
const logEl = document.getElementById('log');
function log(txt){ logEl.textContent += txt + "\\n"; logEl.scrollTop = logEl.scrollHeight; }

document.getElementById('btn').onclick = async () => {
  const key = document.getElementById('key').value.trim();
  if(!key){ alert('Enter API key'); return; }
  log('Validating key...');
  try {
    // validate the key via server-side endpoint
    const r = await fetch('/admin/auth', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ key })
    });
    const data = await r.json();
    if(!r.ok || !data.ok){
      log('Invalid key: ' + (data.error || r.status));
      alert('Invalid key');
      return;
    }
    // store key locally for admin requests
    sessionStorage.setItem('dev_api_key', key);
    log('Key validated. Fetching admin interface...');
    // fetch protected admin content using header
    const rc = await fetch('/admin/content', { headers: { 'X-API-Key': key } });
    if(!rc.ok){
      log('Failed to fetch admin content: ' + rc.status);
      alert('Failed to fetch admin interface (check server logs)');
      return;
    }
    const html = await rc.text();
    // replace current page with the admin UI content,
    // the admin UI will use the stored sessionStorage dev_api_key
    document.open();
    document.write(html);
    document.close();
        

  } catch (e) {
    log('Error: ' + e.toString());
    alert('Error: ' + e.toString());
  }
};
</script>
</body>
</html>
"""
    return HTMLResponse(content=html, status_code=200)


@app.post("/admin/auth", include_in_schema=False)
async def admin_auth(body: dict):
    """
    Validate a provided key in JSON body: { "key": "<the_key>" }.
    Returns { ok: true } when valid.
    Uses same validation helpers as app_security (plain keys or Redis-hashed set).
    """
    from app_security import _is_valid_plain_key, _is_valid_redis_key  # uses functions defined in app_security.py

    if not body or "key" not in body:
        return JSONResponse({"ok": False, "error": "missing key"}, status_code=400)
    key = (body.get("key") or "").strip()
    if not key:
        return JSONResponse({"ok": False, "error": "empty key"}, status_code=400)

    try:
        if _is_valid_plain_key(key) or _is_valid_redis_key(key):
            return JSONResponse({"ok": True})
    except Exception as e:
        # defensive: if app_security internals fail, return false
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    return JSONResponse({"ok": False, "error": "invalid key"}, status_code=401)


@app.get("/admin/content", include_in_schema=False)
def admin_content_page():
    """
    Serves the admin UI HTML file.
    The file itself checks for sessionStorage.dev_api_key in the browser
    and redirects to /admin-login if not found.
    """
    p = Path("public") / "admin.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="admin.html not found.")
    return FileResponse(str(p))

 


# -------------------
# Admin endpoints
# -------------------

@app.post("/admin/upload", dependencies=[Depends(require_api_key)])
async def admin_upload(file: UploadFile = File(...), module: str = Form(...)):
    """
    Upload a single file to DOCS_DIR/<module> (preserves original filename).
    Protected by require_api_key dependency (expects X-API-Key header).
    """
    # ensure module exists in DB
    modules = get_modules()
    if module not in modules:
        raise HTTPException(status_code=404, detail=f"Module not found: {module}")

    # ensure module docs dir exists
    target_dir = Path(DOCS_DIR) / module
    target_dir.mkdir(parents=True, exist_ok=True)

    # sanitize filename
    filename = os.path.basename(file.filename) or ""
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    target = target_dir / filename

    try:
        with open(target, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Write failed: {e}")
    finally:
        await file.close()

    _append_admin_log(f"Uploaded file {filename} -> {target}")
    _write_admin_status({"pid": None, "status": "uploaded", "path": str(target), "module": module})
    return JSONResponse({"ok": True, "path": str(target), "module": module})



@app.post("/admin/ingest", dependencies=[Depends(require_api_key)])
async def admin_ingest(body: Dict = None):
    """
    Trigger the server-side ingest script for a specific module.
    Expects JSON body like { "module": "C01" }.
    Returns immediately with PID and command used.
    """
    module = None
    if body:
        module = body.get("module") or body.get("module_name")

    if not module:
        raise HTTPException(status_code=400, detail="Missing module in request body")

    # verify module exists
    modules = get_modules()
    if module not in modules:
        raise HTTPException(status_code=404, detail=f"Module not found: {module}")

    script = ADMIN_INGEST_SCRIPT
    if not os.path.exists(script):
        return JSONResponse({"ok": False, "error": f"ingest script not found at {script}"}, status_code=500)

    cmd = [shutil.which("python") or "python", script]
    # copy current env and add module indicator
    env = os.environ.copy()
    env["INGEST_MODULE"] = module

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        _append_admin_log(f"Started ingest subprocess pid={proc.pid} module={module} cmd={' '.join(cmd)}")
        _write_admin_status({"pid": proc.pid, "status": "started", "module": module, "cmd": cmd})
        return JSONResponse({"ok": True, "pid": proc.pid, "cmd": cmd, "module": module})
    except Exception as e:
        logger.exception("Failed to start ingest subprocess: %s", e)
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/admin/delete_collection", dependencies=[Depends(require_api_key)])
async def admin_delete_collection(body: Dict):
    """
    Delete Qdrant collection(s) or delete points for a module.
    If body contains { "module": "<module>", "confirm": true } -> delete only points with payload.module == module
    If body contains { "confirm_all": true, "confirm": true } -> delete whole configured collections (danger).
    """
    if not body or not body.get("confirm"):
        raise HTTPException(status_code=400, detail="Missing confirm in body")

    module = body.get("module")
    confirm_all = bool(body.get("confirm_all"))

    try:
        # list collections
        cols_resp = qc.get_collections()
        cols = [c.name for c in getattr(cols_resp, "collections", [])]

        result = {"ok": True, "module": module, "collections_before": cols, "deleted_collections": [], "deleted_points_summary": {}}

        if module:
            # delete points in the configured main collection that belong to the module
            if QDRANT_COLLECTION not in cols:
                result["ok"] = False
                result["message"] = f"Collection {QDRANT_COLLECTION} not found"
                _append_admin_log(result["message"])
                return JSONResponse(result)

            # prepare a filter to delete by payload.module == module
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            filt = Filter(must=[FieldCondition(key="module", match=MatchValue(value=module))])

            # Try to construct a PointsSelector (preferred for many qdrant-client versions).
            # PointsSelector may live in different modules depending on qdrant-client version.
            PointsSelector = None
            try:
                # preferred location
                from qdrant_client.http.models import PointsSelector as _PS
                PointsSelector = _PS
            except Exception:
                try:
                    from qdrant_client.models import PointsSelector as _PS
                    PointsSelector = _PS
                except Exception:
                    PointsSelector = None

            # perform delete using points_selector when available, else fall back to filter= (defensive)
            try:
                if PointsSelector is not None:
                    ps = PointsSelector(filter=filt)
                    res = qc.delete(collection_name=QDRANT_COLLECTION, points_selector=ps)
                else:
                    # fallback for client versions that accept 'filter' keyword (some accept it directly)
                    res = qc.delete(collection_name=QDRANT_COLLECTION, filter=filt)

                _append_admin_log(f"Deleted points for module={module} from collection {QDRANT_COLLECTION} -> {res}")
                # store response for visibility (stringify safely)
                try:
                    result["deleted_points_summary"][QDRANT_COLLECTION] = res if isinstance(res, (dict, list, str)) else str(res)
                except Exception:
                    result["deleted_points_summary"][QDRANT_COLLECTION] = str(res)
            except Exception as e:
                _append_admin_log(f"Failed deleting points for module {module}: {e}")
                result["ok"] = False
                result["error"] = str(e)
            return JSONResponse(result)

        if confirm_all:
            # delete the configured collections entirely (dangerous)
            deleted = []
            errors = {}
            for col in COLLECTIONS_TO_TRY:
                if col in cols:
                    try:
                        qc.delete_collection(collection_name=col)
                        deleted.append(col)
                        _append_admin_log(f"Deleted collection {col}")
                    except Exception as e:
                        errors[col] = str(e)
                        _append_admin_log(f"Failed deleting collection {col}: {e}")
            result["deleted_collections"] = deleted
            result["errors"] = errors
            return JSONResponse(result)

        # fallback: nothing done
        raise HTTPException(status_code=400, detail="Specify 'module' to delete module points OR 'confirm_all': true to delete whole collections")
    except Exception as e:
        logger.exception("Unexpected error while deleting collections: %s", e)
        _append_admin_log(f"Unexpected error while deleting collections: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)



    
    
STATUS_DIR = os.getenv("INGEST_STATUS_DIR", "data/ingest")
STATUS_FILE = os.path.join(STATUS_DIR, "ingest_status.json")
INGEST_LOG = os.path.join(STATUS_DIR, "ingest.log")

def _is_pid_alive(pid: int) -> bool:
    try:
        # Prefer psutil if installed (more cross-platform)
        try:
            import psutil
            return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
        except Exception:
            # Fallback: os.kill(pid, 0) on POSIX; on Windows fallback to psutil failure -> assume alive if can't check
            if os.name == "posix":
                os.kill(pid, 0)
                return True
            # On Windows, if psutil not present, try opening via subprocess tasklist (best-effort)
            return True
    except OSError:
        return False
    except Exception:
        return True

@app.get("/admin/ingest_status", dependencies=[Depends(require_api_key)])
def admin_ingest_status():
    """
    Returns JSON with the last ingest status and whether the PID is still alive.
    """
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r", encoding="utf-8") as fh:
                st = json.load(fh)
        else:
            st = {"status": "idle", "message": "no ingest has run yet"}
    except Exception as e:
        st = {"status": "unknown", "error": str(e)}

    alive = False
    pid = st.get("pid")
    if pid:
        try:
            alive = _is_pid_alive(int(pid))
        except Exception:
            alive = False

    # include last N lines of log if available
    tail = ""
    try:
        if os.path.exists(INGEST_LOG):
            with open(INGEST_LOG, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
                tail = "\n".join(lines[-200:])  # last 200 lines
    except Exception:
        tail = ""

    st["pid_alive"] = bool(alive)
    st["log_tail"] = tail
    return JSONResponse(st)

# Helpers to write the same files ingest.py uses so the UI can poll them
def _append_admin_log(msg: str):
    try:
        os.makedirs(os.path.dirname(INGEST_LOG), exist_ok=True)
        with open(INGEST_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.utcnow().isoformat()}Z] {msg}\n")
    except Exception as e:
        logger.exception("failed _append_admin_log: %s", e)

def _write_admin_status(data: dict):
    try:
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
        data = dict(data)
        data["updated_at"] = datetime.utcnow().isoformat() + "Z"
        with open(STATUS_FILE, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("failed _write_admin_status: %s", e)

