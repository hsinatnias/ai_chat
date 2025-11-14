# api/routes_chat.py
import os
import uuid
import sqlalchemy
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from core import utils as core_utils
from auth.sessions import current_session
from cache import get_cached_answer, cache_answer, semantic_cache_upsert
from db import get_async_db

router = APIRouter(prefix="/api", tags=["chat"])

class ChatIn(BaseModel):
    text: str
    lang: Optional[str] = None
    conversation_id: Optional[str] = None
    top_k: Optional[int] = None

class ChatOut(BaseModel):
    answer: str
    citations: List[dict]

@router.post("/chat", response_model=ChatOut)
async def chat(payload: ChatIn, request: Request, session = Depends(current_session), db=Depends(get_async_db)):
    """
    Chat endpoint that:
      - uses session (cookie) via current_session
      - uses app.state.qc and app.state.http_client at request time
      - persists conversation/messages via async DB dependency
      - calls Ollama via core_utils.call_ollama_async_http_client(http_client, prompt)
    """
    # get runtime clients/helpers from app.state (set in app.py)
    qc = getattr(request.app.state, "qc", None)
    http_client = getattr(request.app.state, "http_client", None)
    run_qdrant_search = getattr(request.app.state, "run_qdrant_search", None)

    if qc is None or http_client is None:
        raise HTTPException(status_code=500, detail="Server not fully initialized (qc/http_client missing)")

    # session values
    session_id = session.get("session_id")
    user_id = session.get("user_id") or None

    # language detection
    detected = core_utils.detect_lang(payload.text)
    user_lang = (payload.lang or detected or "en")[:2].lower()

    # 1) exact-match cache
    cached = await get_cached_answer(payload.text, user_lang)
    if cached:
        return {"answer": cached.get("answer"), "citations": cached.get("citations", [])}

    # 2) embeddings
    vec = []
    if hasattr(core_utils, "embed_query"):
        try:
            vec = await core_utils.embed_query(payload.text) or []
        except Exception:
            vec = []
    else:
        # fallback to app-level embed if present
        embed_fn = getattr(request.app, "embed_query", None) or getattr(request.app.state, "embed_query", None)
        if callable(embed_fn):
            try:
                vec = await embed_fn(payload.text) or []
            except Exception:
                vec = []

    # 3) retrieval: prefer run_qdrant_search provided on app.state
    hits_d = []
    if run_qdrant_search:
        try:
            hits_d = run_qdrant_search(vec, user_lang, payload.top_k or None) or []
        except Exception:
            hits_d = []
    else:
        # fallback inline search (best-effort)
        k = payload.top_k or int(os.getenv("TOP_K", "6"))
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            flt = Filter(must=[FieldCondition(key="lang", match=MatchValue(value="ja" if user_lang.startswith("ja") else "en"))])
        except Exception:
            flt = None
        try:
            res = qc.search(collection_name=os.getenv("QDRANT_COLLECTION", "kb_chunks"), query_vector=vec or [], limit=k, with_payload=True, query_filter=flt)
            hits = res or []
            # normalize into list of {'payload':..., 'score':...}
            for h in hits:
                payload_ = getattr(h, "payload", None) or {}
                score = getattr(h, "score", None) or None
                hits_d.append({"payload": payload_, "score": score})
        except Exception:
            hits_d = []

    # build prompt & citations
    prompt, citations = core_utils.build_prompt(payload.text, user_lang, hits_d)

    # choose conversation (create if missing)
    conv_id = payload.conversation_id or str(uuid.uuid4())
    # Persist conversation record (idempotent)
    await db.execute(
        sqlalchemy.text("INSERT INTO conversations (id, session_id, user_id) VALUES (:id,:sess,:user) ON CONFLICT (id) DO NOTHING"),
        {"id": conv_id, "sess": session_id, "user": user_id}
    )
    await db.commit()

    # persist user message
    await db.execute(
        sqlalchemy.text(
            "INSERT INTO messages (id, conversation_id, session_id, user_id, role, content) "
            "VALUES (:id,:conv,:sess,:user,'user',:content)"
        ),
        {"id": str(uuid.uuid4()), "conv": conv_id, "sess": session_id, "user": user_id, "content": payload.text}
    )
    await db.commit()

    # call LLM using app-state http_client via helper
    res = await core_utils.call_ollama_async_http_client(http_client, prompt)
    if not res or (isinstance(res, dict) and not res.get("ok") and not res.get("text")):
        raise HTTPException(status_code=503, detail="LLM generation failed")
    answer_text = res.get("text") if isinstance(res, dict) else str(res)

    # persist assistant message
    await db.execute(
        sqlalchemy.text(
            "INSERT INTO messages (id, conversation_id, session_id, user_id, role, content) "
            "VALUES (:id,:conv,:sess,:user,'assistant',:content)"
        ),
        {"id": str(uuid.uuid4()), "conv": conv_id, "sess": session_id, "user": user_id, "content": answer_text}
    )
    await db.commit()

    # cache & semantic upsert (best-effort)
    try:
        await cache_answer(payload.text, user_lang, {"answer": answer_text, "citations": citations})
    except Exception:
        pass

    try:
        import hashlib
        cache_id = int(hashlib.sha1((payload.text + user_lang).encode("utf-8")).hexdigest()[:12], 16)
        await semantic_cache_upsert(cache_id, vec or [], {"question": payload.text, "lang": user_lang, "answer": answer_text, "citations": citations})
    except Exception:
        pass

    return {"answer": answer_text, "citations": citations}
