# cache.py
import os
import hashlib
import msgpack
import asyncio
from typing import Any, Optional
from redis.asyncio import Redis
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Config (env-compatible with app.py / ingest.py)
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
QDRANT_CACHE_COLLECTION = os.getenv("QDRANT_CACHE_COLLECTION", "kb_cache")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))

# Optional override for embed dim (useful for startup-time creation)
# If set, must be integer (e.g. EMB_DIM=1536). If not set, we infer from vectors passed in.
EMB_DIM_FALLBACK = int(os.getenv("EMB_DIM", "0")) or None

# Redis client for exact-match cache
redis = Redis.from_url(REDIS_URL, decode_responses=False)

# === cache.py edits ===
# Replace existing cache_answer / get_cached_answer with these:

def make_key(prefix: str, *parts) -> str:
    h = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).hexdigest()
    return f"{prefix}:{h}"

async def cache_answer(question: str, lang: str, answer_obj: dict, ttl: int = CACHE_TTL, module: str | None = None):
    """
    exact-match cache (msgpack)
    - attaches `_module` to the stored object when provided so we can invalidate by module.
    """
    key = make_key("answer", question, lang)
    obj = dict(answer_obj) if isinstance(answer_obj, dict) else {"answer": answer_obj}
    if module:
        obj["_module"] = module
    packed = msgpack.packb(obj)
    await redis.set(key, packed, ex=ttl)

async def get_cached_answer(question: str, lang: str):
    key = make_key("answer", question, lang)
    data = await redis.get(key)
    if not data:
        return None
    try:
        return msgpack.unpackb(data, raw=False)
    except Exception:
        # gracefully handle corrupted data
        return None

# Optional helper to list answer keys (for debugging)
async def list_answer_keys(limit: int = 100):
    keys = []
    async for k in redis.scan_iter(match="answer:*"):
        keys.append(k)
        if len(keys) >= limit:
            break
    return keys


# Replace semantic_cache_upsert with module-aware version:

async def semantic_cache_upsert(id_: int, vector: list, payload: dict, module: str | None = None):
    """
    Upsert a single point into Qdrant cache collection.
    - Adds 'vec', 'kb_version' and optional 'module' to payload so validation/overlap checks work.
    - Runs blocking qc.upsert in a thread to avoid blocking event loop.
    """
    try:
        # ensure collection exists (infer dim from vector if possible)
        dim = len(vector) if vector and isinstance(vector, (list, tuple)) else 0
        try:
            _ensure_cache_collection(dim)
        except Exception:
            # continue; actual upsert will fail if collection doesn't exist or wrong dim
            pass

        payload = dict(payload)
        payload["vec"] = vector
        payload["kb_version"] = KB_VERSION
        if module:
            payload["module"] = module

        pt = PointStruct(id=id_, vector=vector, payload=payload)
        # use thread to avoid blocking asyncio loop
        await asyncio.to_thread(qc.upsert, collection_name=QDRANT_CACHE_COLLECTION, points=[pt])
        print(f"INFO: semantic_cache_upsert id={id_} vec_len={(len(vector) if vector else 0)} payload_keys={list(payload.keys())}")
    except Exception as e:
        print("ERROR: semantic_cache_upsert failed:", e)
        raise


# --- Semantic cache using Qdrant ---
qc = QdrantClient(url=QDRANT_URL)

def _get_existing_collections() -> list:
    """
    Safely return list of existing collection names. If listing fails, returns [].
    """
    try:
        resp = qc.get_collections()
        if not resp or getattr(resp, "collections", None) is None:
            return []
        return [c.name for c in resp.collections]
    except Exception as e:
        print("WARN: failed to list Qdrant collections:", e)
        return []

def _ensure_cache_collection(dim: int):
    """
    Create cache collection if missing. Safe to call repeatedly.
    Prefer EMB_DIM_FALLBACK if provided and dim == 0.
    """
    try:
        cols = _get_existing_collections()
        if QDRANT_CACHE_COLLECTION in cols:
            return
        # choose size: prefer explicit dim; fallback to EMB_DIM_FALLBACK; if neither, raise so caller knows
        size = dim or EMB_DIM_FALLBACK
        if not size or size <= 0:
            raise ValueError("Vector dimension unknown: pass vector to upsert/search or set EMB_DIM env var")
        from qdrant_client.models import Distance, VectorParams
        print(f"INFO: creating Qdrant collection '{QDRANT_CACHE_COLLECTION}' size={size}")
        qc.create_collection(collection_name=QDRANT_CACHE_COLLECTION,
                             vectors_config=VectorParams(size=size, distance=Distance.COSINE))
    except Exception as e:
        # tolerate races / "already exists" — just log and continue
        print(f"WARN: _ensure_cache_collection: could not create collection '{QDRANT_CACHE_COLLECTION}': {e}")

# top-level
KB_VERSION = int(os.getenv("KB_VERSION", "1"))



def semantic_cache_search(vector: list, top_k: int = 1, score_threshold: float = 0.78, kb_version: int | None = None):
    """
    Search the semantic cache and return a top candidate (or None).
    Filters by kb_version (if present) so cache invalidation can be achieved by bumping KB_VERSION.
    Runs as a synchronous function (like in app.py) to keep usage simple.
    """
    try:
        # ensure collection exists (infer dim from vector if possible)
        dim = len(vector) if vector and isinstance(vector, (list, tuple)) else 0
        try:
            _ensure_cache_collection(dim)
        except Exception:
            pass

        query_filter = None
        v = kb_version if kb_version is not None else KB_VERSION
        if v is not None:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(must=[FieldCondition(key="kb_version", match=MatchValue(value=v))])

        res = qc.search(
            collection_name=QDRANT_CACHE_COLLECTION,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter
        )
        if not res:
            return None
        top = res[0]
        if getattr(top, "score", None) and top.score >= score_threshold:
            return {"id": getattr(top, "id", None), "payload": top.payload, "score": top.score}
    except Exception as e:
        print("semantic_cache_search error:", e)
        return None
    return None
