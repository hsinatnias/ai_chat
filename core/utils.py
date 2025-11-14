# core/utils.py
import os, re, asyncio, base64, zlib, logging
from typing import List, Optional
from pathlib import Path
from langdetect import detect
import httpx

logger = logging.getLogger("rag_admin")

# These env values mirror what's in your app.py
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_PROMPT_MAX_CHARS = int(os.getenv("OLLAMA_PROMPT_MAX_CHARS", "10000"))
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")

# --- LLM helpers ---------------------------------------------------------
# A thin wrapper that re-uses your HTTP client if you want. If you prefer, import HTTP_CLIENT from app.
async def call_ollama_async_http_client(http_client, prompt: str, model: Optional[str] = None):
    model = model or LLM_MODEL
    if OLLAMA_PROMPT_MAX_CHARS and len(prompt) > OLLAMA_PROMPT_MAX_CHARS:
        head = prompt[: int(OLLAMA_PROMPT_MAX_CHARS * 0.7)]
        tail = "\n\n[...context truncated...]\n\n" + prompt[-int(OLLAMA_PROMPT_MAX_CHARS * 0.3):]
        prompt = head + tail
        logger.warning("WARN: prompt truncated to %s chars before sending to Ollama", OLLAMA_PROMPT_MAX_CHARS)

    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    body = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    try:
        r = await http_client.post(url, json=body)
    except httpx.RequestError as e:
        logger.error("Ollama request error: %s", e)
        return {"ok": False, "error": f"request_error:{e}"}
    if r.status_code >= 400:
        text = r.text or ""
        logger.warning("Ollama returned status %s: %s", r.status_code, text[:1000])
        return {"ok": False, "status": r.status_code, "error": text}
    try:
        data = r.json()
    except Exception as e:
        logger.error("Ollama returned non-JSON response: %s", e)
        return {"ok": False, "error": f"non-json:{e}", "raw": r.text[:2000]}

    # Normalize shapes (same logic as your app.py)
    if isinstance(data, dict):
        text = data.get("response") or data.get("text") or ""
        if not text:
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                parts = []
                for ch in data["choices"]:
                    if isinstance(ch, dict):
                        t = ch.get("text") or ch.get("message") or ch.get("output")
                        if isinstance(t, str) and t:
                            parts.append(t)
                text = "\n".join(parts).strip()
            elif isinstance(data.get("output"), list):
                parts = []
                for o in data.get("output", []):
                    if isinstance(o, dict):
                        parts.append(o.get("content", ""))
                text = "\n".join(p for p in parts if p).strip()
        return {"ok": True, "text": text}
    return {"ok": True, "text": str(data)}

async def call_ollama_async(prompt: str, model: Optional[str] = None, timeout: int = 300):
    """
    Convenience wrapper: create a temporary httpx.AsyncClient and call the http_client wrapper.
    Returns same normalized shape as call_ollama_async_http_client.
    """
    model = model or LLM_MODEL
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await call_ollama_async_http_client(client, prompt, model=model)

# --- language / IO helpers ----------------------------------------------
def detect_lang(s: str) -> str:
    if not s or not s.strip():
        return "en"
    if re.search(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]', s):
        return "ja"
    try:
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

def load_chunk_text(chunk_path: Optional[str], max_chars: int = 3000) -> str:
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
    intro = f"You are a helpful assistant. Answer in {('Japanese' if lang.startswith('ja') else 'English')}."
    rules = "Use ONLY the context. If unsure, say you don't know. Cite sources as [1], [2], ..."
    ctx_lines = []
    cited = []
    for i, h in enumerate(hits, start=1):
        payload = h.get("payload", {}) or {}
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

def cosine_similarity(a: list, b: list) -> float:
    dot = 0.0; na = 0.0; nb = 0.0
    L = min(len(a), len(b))
    for i in range(L):
        ai = float(a[i]); bi = float(b[i])
        dot += ai * bi
        na += ai * ai
        nb += bi * bi
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))

# --- embedding helper (async, lazy load) -------------------------------
_EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
_USE_E5 = _EMBED_MODEL.lower().startswith("intfloat/multilingual-e5")
_st = None

def _ensure_st():
    global _st
    if _st is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st = SentenceTransformer(_EMBED_MODEL)
            logger.info("Loaded SentenceTransformer model: %s", _EMBED_MODEL)
        except Exception as e:
            logger.exception("Failed to load SentenceTransformer (%s): %s", _EMBED_MODEL, e)
            _st = None
    return _st

async def embed_query(text: str):
    """
    Async wrapper that returns a single embedding vector (list[float]) or [] on failure.
    Runs encoding in a thread to avoid blocking the event loop.
    """
    st = _ensure_st()
    if st is None:
        # If we couldn't load the model, return empty vector so callers degrade gracefully.
        return []
    def _encode(q):
        q_ = ("passage: " + q) if _USE_E5 else q
        return st.encode([q_], normalize_embeddings=True)[0].tolist()
    try:
        return await asyncio.to_thread(_encode, text)
    except Exception:
        logger.exception("embed_query failed")
        return []
