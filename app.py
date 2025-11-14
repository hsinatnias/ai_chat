# app.py
from dotenv import load_dotenv
load_dotenv()
import os
import re
import asyncio
from typing import Optional, List, Dict
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException, Form, Response, status
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect
import httpx
import math
import base64
import zlib
import shutil
import subprocess
import json
import logging
from datetime import datetime
from pathlib import Path
from auth.sessions import create_session, current_session, get_session, destroy_session, set_session_cookie
from db import get_async_db
import sqlalchemy
from admin import router as admin_router
from admin import public_admin_router as public_admin_router


# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# sentence-transformers import guarded (may be heavy / not available in some environments)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

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

OLLAMA_PROMPT_MAX_CHARS = int(os.getenv("OLLAMA_PROMPT_MAX_CHARS", "10000"))
OLLAMA_STARTUP_TIMEOUT = int(os.getenv("OLLAMA_STARTUP_TIMEOUT", "120"))  # seconds to wait for model availability
QDRANT_STARTUP_TIMEOUT = int(os.getenv("QDRANT_STARTUP_TIMEOUT", "30"))  # seconds to wait for qdrant

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

# make startup asynchronous so we can probe Ollama/Qdrant readiness
@app.on_event("startup")
async def startup():
    # ensure DB table exists (run in thread if blocking)
    try:
        await asyncio.to_thread(create_table)
    except Exception as e:
        logger.exception("create_table failed on startup: %s", e)

    # Wait for Qdrant to be reachable (best-effort)
    qc_probe = QdrantClient(url=QDRANT_URL)
    q_start = datetime.utcnow().timestamp()
    while True:
        try:
            qc_probe.get_collections()
            logger.info("Qdrant reachable at %s", QDRANT_URL)
            break
        except Exception as e:
            if datetime.utcnow().timestamp() - q_start > QDRANT_STARTUP_TIMEOUT:
                logger.warning("Timed out waiting for Qdrant (%ss)", QDRANT_STARTUP_TIMEOUT)
                break
            await asyncio.sleep(1)

    # Wait for Ollama and the configured model to be available (best-effort)
    model_to_check = LLM_MODEL
    start = datetime.utcnow().timestamp()
    async_client = httpx.AsyncClient(timeout=5.0)
    while True:
        try:
            r = await async_client.get(f"{OLLAMA_URL.rstrip('/')}/v1/models")
            if r.status_code == 200:
                try:
                    body = r.json()
                    data = body.get("data") or []
                    ids = [d.get("id") for d in data if isinstance(d, dict)]
                    if model_to_check in ids:
                        logger.info("Ollama model '%s' present", model_to_check)
                        break
                    else:
                        logger.info("Ollama reachable but model '%s' not loaded yet. Available models: %s", model_to_check, ids)
                except Exception as je:
                    logger.debug("Ollama model list JSON parse error: %s", je)
            else:
                logger.debug("Ollama models endpoint returned HTTP %s", r.status_code)
        except Exception as e:
            # This is expected while Ollama or the network is warming up; log debug
            logger.debug("Waiting for Ollama: %s", e)

        if datetime.utcnow().timestamp() - start > OLLAMA_STARTUP_TIMEOUT:
            logger.warning("Timed out waiting for Ollama/model '%s' (%ss)", model_to_check, OLLAMA_STARTUP_TIMEOUT)
            break
        await asyncio.sleep(2)

    try:
        await async_client.aclose()
    except Exception:
        pass

# Static files & root
app.mount("/static", StaticFiles(directory="public"), name="static")

@app.get("/", include_in_schema=False)
def root(session = Depends(get_session)):
    if not session:
        return RedirectResponse(url="/login")
    return FileResponse("public/index.html")

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

# Try to initialize SentenceTransformer but don't let a failure crash the app
st = None
if SentenceTransformer is not None:
    try:
        st = SentenceTransformer(EMBED_MODEL)
        try:
            EMB_DIM = st.get_sentence_embedding_dimension()
        except Exception:
            EMB_DIM = len(st.encode(["test"], normalize_embeddings=True)[0])
    except Exception as e:
        logger.exception("Failed to initialize SentenceTransformer: %s", e)
        st = None
        EMB_DIM = 1536  # fallback guess
else:
    logger.warning("sentence_transformers not installed or import failed; embeddings may be disabled")
    st = None
    EMB_DIM = 1536

E5 = EMBED_MODEL.lower().startswith("intfloat/multilingual-e5")

# Async HTTP client for Ollama
HTTP_CLIENT = httpx.AsyncClient(timeout=300.0)

# after qc and HTTP_CLIENT are initialized in app.py
from core import utils as core_utils   # optional if you want to reuse helpers centrally
from api.routes_chat import router as chat_router
from auth import sessions as auth_sessions

from api.routes_analytics import router as analytics_router
app.include_router(analytics_router)

# expose objects on app.state for routers to consume
app.state.qc = qc
app.state.http_client = HTTP_CLIENT
app.state.st = st
app.state.embed_model = EMBED_MODEL

# expose run_qdrant_search (same logic, defensive)
def run_qdrant_search(vec, user_lang, top_k=None):
    k = top_k or TOP_K
    try:
        flt = Filter(must=[FieldCondition(key="lang", match=MatchValue(value="ja" if (user_lang or "").startswith("ja") else "en"))])
        # ensure vec is not None
        qvec = vec or []
        res = qc.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=k, with_payload=True, score_threshold=None, query_filter=flt)
        hits = res or []
        if len(hits) < max(2, k // 2):
            res2 = qc.search(collection_name=QDRANT_COLLECTION, query_vector=qvec, limit=k, with_payload=True)
            seen = set(); merged = []
            for h in list(hits) + list(res2):
                try:
                    pid = h.payload.get("content_hash") or getattr(h, "id", None)
                except Exception:
                    pid = getattr(h, "id", None)
                if pid not in seen:
                    seen.add(pid); merged.append(h)
            hits = merged[:k]
        hits_d = []
        for h in hits:
            payload = getattr(h, "payload", None) or (h.payload if hasattr(h, "payload") else (h.get("payload") if isinstance(h, dict) else {}))
            score = getattr(h, "score", None) or (h.score if hasattr(h, "score") else (h.get("score") if isinstance(h, dict) else None))
            hits_d.append({"payload": payload or {}, "score": score})
        return hits_d
    except Exception as e:
        logger.exception("run_qdrant_search failed: %s", e)
        return []
app.state.run_qdrant_search = run_qdrant_search

# include router after app.state prepared
app.include_router(chat_router)

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
    Returns [] on failure or if st is not available.
    """
    if app.state.st is None:
        # Prefer core.utils' lazy loader
        if hasattr(core_utils, "embed_query"):
            return await core_utils.embed_query(q)
        return []
    def _encode(text):
        q_ = ("passage: " + text) if E5 else text
        return app.state.st.encode([q_], normalize_embeddings=True)[0].tolist()
    try:
        return await asyncio.to_thread(_encode, q)
    except Exception:
        logger.exception("embed_query failed")
        return []

def detect_lang(s: str) -> str:
    """
    Prefer quick CJK heuristic for short Japanese/Chinese/Korean text,
    then fall back to langdetect for longer text.
    """
    if not s or not s.strip():
        return "en"
    # Quick check for CJK / Japanese characters
    if re.search(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]', s):
        return "ja"
    try:
        # langdetect can misclassify very short texts; we only pass a trimmed substring
        return detect(s[:4000])
    except Exception:
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
    intro = f"You are a helpful assistant. Answer in {('Japanese' if (lang or '').startswith('ja') else 'English')}."
    rules = "Use ONLY the context. If unsure, say you don't know. Cite sources as [1], [2], ..."
    ctx_lines = []
    cited = []

    for i, h in enumerate(hits or [], start=1):
        payload = h.get("payload", {}) or {}
        # prefer full chunk text stored in payload
        full = ""
        if payload.get("full_text_z"):
            try:
                compressed = payload.get("full_text_z")
                full = _decompress_payload_text(compressed)
            except Exception:
                full = ""
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

async def call_ollama_async(prompt: str, model: str = None):
    """
    Async call to Ollama with prompt-length guard.
    Returns dict: {"ok": True, "text": "..."} or {"ok": False, "error": "...", "status": <int?>}
    """
    model = model or LLM_MODEL
    # Guard / trim
    if OLLAMA_PROMPT_MAX_CHARS and len(prompt) > OLLAMA_PROMPT_MAX_CHARS:
        head = prompt[: int(OLLAMA_PROMPT_MAX_CHARS * 0.7)]
        tail = "\n\n[...context truncated...]\n\n" + prompt[-int(OLLAMA_PROMPT_MAX_CHARS * 0.3):]
        prompt = head + tail
        logger.warning("WARN: prompt truncated to %s chars before sending to Ollama", OLLAMA_PROMPT_MAX_CHARS)

    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    body = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    try:
        r = await HTTP_CLIENT.post(url, json=body)
    except httpx.RequestError as e:
        logger.error("Ollama request error: %s", e)
        return {"ok": False, "error": f"request_error:{e}"}
    if r.status_code >= 400:
        # include model-not-found bodies too
        text = r.text or ""
        logger.warning("Ollama returned status %s: %s", r.status_code, text[:1000])
        return {"ok": False, "status": r.status_code, "error": text}
    try:
        data = r.json()
    except Exception as e:
        logger.error("Ollama returned non-JSON response: %s", e)
        return {"ok": False, "error": f"non-json:{e}", "raw": r.text[:2000]}

    # try to normalize response shapes
    if isinstance(data, dict):
        # Ollama newer versions use "response" or "text" or nested shapes
        text = data.get("response") or data.get("text") or ""
        # sometimes response may be list-like
        if not text:
            # check for keys like 'choices' or 'output'
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                # try to join text fields
                parts = []
                for ch in data["choices"]:
                    if isinstance(ch, dict):
                        t = ch.get("text") or ch.get("message") or ch.get("output")
                        if isinstance(t, str) and t:
                            parts.append(t)
                text = "\n".join(parts).strip()
            elif isinstance(data.get("output"), list):
                # some shapes: output is list of dicts with "content"
                parts = []
                for o in data.get("output", []):
                    if isinstance(o, dict):
                        parts.append(o.get("content", ""))
                text = "\n".join(p for p in parts if p).strip()
        return {"ok": True, "text": text}
    return {"ok": True, "text": str(data)}

def cosine_similarity(a: list, b: list) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    # protect against different lengths
    L = min(len(a), len(b))
    for i in range(L):
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

async def summarize_snippets(snippets: List[str], max_chars: int = 1200):
    """
    Simple summarization wrapper: concatenates snippets and asks small model for 3-bullet summary.
    Returns summary string.
    """
    text = "\n\n".join([s[:800] for s in snippets])  # avoid huge payload to small model
    prompt = f"Summarize the following snippets into 3 short bullets (keep factual):\n\n{text}\n\nSummary:"
    # call small model first (faster / cheaper)
    summary = await call_ollama_async(prompt, model=os.getenv("LLM_MODEL_SMALL", LLM_MODEL))
    if isinstance(summary, dict) and summary.get("ok"):
        return summary.get("text", "") or ""
    return ""

# --- Login + auth endpoints (keep in app.py) ---
@app.get("/login", include_in_schema=False)
def login_page():
    html = """
    <!doctype html><html><head><meta charset="utf-8"><title>Login</title></head><body>
    <div style="max-width:420px;margin:40px auto;font-family:system-ui,Arial">
        <h2>Login</h2>
        <form id="f">
          <label>Email</label><br/>
          <input id="email" name="email" type="email" required style="width:100%;padding:8px"/><br/><br/>
          <label>Password</label><br/>
          <input id="password" name="password" type="password" required style="width:100%;padding:8px"/><br/><br/>
          <button type="submit" style="padding:8px 12px">Sign in</button>
        </form>
        <div id="msg" style="color:red;margin-top:12px"></div>
        <script>
        document.getElementById('f').onsubmit = async (e) => {
            e.preventDefault();
            const email = document.getElementById('email').value.trim();
            const password = document.getElementById('password').value;
            if(!email) { document.getElementById('msg').textContent = 'Email required'; return; }
            if(!password) { document.getElementById('msg').textContent = 'Password required'; return; }

            try {
              const resp = await fetch('/api/auth/login', {
                method: 'POST',
                credentials: 'same-origin',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
              });

              console.log('login response status', resp.status);

              if (resp.ok) {
                window.location = '/';
                return;
              }

              const txt = await resp.text();
              // try parse JSON message first
              try {
                const j = JSON.parse(txt);
                document.getElementById('msg').textContent = j.error || 'Login failed';
              } catch (e) {
                document.getElementById('msg').textContent = txt || 'Login failed';
              }
            } catch (err) {
              console.error('login error', err);
              document.getElementById('msg').textContent = 'Login error: ' + err.toString();
            }
        };
        </script>
    </div>
    </body></html>
    """
    return HTMLResponse(content=html, status_code=200)


@app.post("/api/auth/login")
async def api_login(body: dict, db = Depends(get_async_db)):
    """
    Expects JSON { "email": "<email>", "password": "<password>" }.
    Returns generic "invalid credentials" on failure to avoid username enumeration.
    """
    email = (body.get("email") if isinstance(body, dict) else None) or None
    password = (body.get("password") if isinstance(body, dict) else None) or None

    if not email:
        return JSONResponse({"ok": False, "error": "email required"}, status_code=400)
    if not password:
        return JSONResponse({"ok": False, "error": "password required"}, status_code=400)

    try:
        q = await db.execute(
            sqlalchemy.text("SELECT id, password_hash, is_active FROM users WHERE email = :email"),
            {"email": email}
        )
        row = q.first()
    except Exception:
        # don't leak DB internals
        return JSONResponse({"ok": False, "error": "db error"}, status_code=500)

    if not row:
        return JSONResponse({"ok": False, "error": "invalid credentials"}, status_code=401)

    user_id, password_hash, is_active = row[0], row[1], row[2]

    # Optional: check active flag
    if is_active is False:
        return JSONResponse({"ok": False, "error": "invalid credentials"}, status_code=401)

    if not password_hash:
        # If you allow social-login users without password, handle separately.
        return JSONResponse({"ok": False, "error": "invalid credentials"}, status_code=401)

    try:
        if not verify_password(password, password_hash):
            return JSONResponse({"ok": False, "error": "invalid credentials"}, status_code=401)
    except Exception:
        # malformed hash or verify error
        return JSONResponse({"ok": False, "error": "invalid credentials"}, status_code=401)

    # success: create session and set cookie
    sid = await create_session(user_id)
    resp = JSONResponse({"ok": True, "session_id": sid})
    # ensure set_session_cookie sets Secure/HttpOnly/SameSite attributes (see notes)
    set_session_cookie(resp, sid)
    return resp

# GET /api/auth/whoami -> useful for client-side checks
@app.get("/api/auth/whoami")
async def api_whoami(session = Depends(current_session)):
    # current_session raises 401 when not authenticated (good for protected endpoint)
    return {"session_id": session.get("session_id"), "user_id": session.get("user_id")}


# POST /api/auth/logout
@app.post("/api/auth/logout")
async def api_logout(response: Response, session = Depends(current_session)):
    # current_session ensures user is authenticated
    sid = session.get("session_id")
    if sid:
        try:
            await destroy_session(sid)
        except Exception:
            logger.exception("destroy_session failed for sid=%s", sid)
    # remove cookie
    response.delete_cookie(os.getenv("SESSION_COOKIE_NAME", "session_id"), path="/")
    return {"ok": True}


# include admin routes (moved to admin.py)
app.include_router(public_admin_router)
app.include_router(admin_router)
