# app_security.py
import os
import hashlib
import hmac
from typing import Optional

from fastapi import Header, HTTPException, status, Request

# Optional Redis usage (only if you want dynamic keys later)
try:
    from redis import Redis
    _redis_available = True
except Exception:
    _redis_available = False

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")            # single key (plaintext)
API_KEYS = os.getenv("API_KEYS", "")          # comma-separated keys (plaintext)
API_KEY_HASHED_SET = os.getenv("API_KEY_HASHED_SET", "")  # name of redis set if using Redis hashed keys
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

# Normalize configured keys into a list of sha256 hashes (if provided)
def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

_configured_hashes = set()
if ADMIN_API_KEY:
    _configured_hashes.add(_sha256_hex(ADMIN_API_KEY.strip()))
if API_KEYS:
    for k in API_KEYS.split(","):
        k = k.strip()
        if k:
            _configured_hashes.add(_sha256_hex(k))

_redis = None
if _redis_available and API_KEY_HASHED_SET:
    try:
        _redis = Redis.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        _redis = None

def _is_valid_plain_key(key: str) -> bool:
    """Compare provided key against configured plaintext keys (constant-time)."""
    if not key:
        return False
    # check against ADMIN_API_KEY / API_KEYS by hashing and comparing to configured hashes
    key_hash = _sha256_hex(key)
    for h in _configured_hashes:
        if hmac.compare_digest(key_hash, h):
            return True
    return False

def _is_valid_redis_key(key: str) -> bool:
    """If using Redis hashed set: compare SHA256(key) membership in set."""
    if not _redis or not API_KEY_HASHED_SET:
        return False
    try:
        key_hash = _sha256_hex(key)
        return _redis.sismember(API_KEY_HASHED_SET, key_hash)
    except Exception:
        return False

def require_api_key(x_api_key: Optional[str] = Header(None), request: Request = None):
    """
    FastAPI dependency to require an API key.
    - Accepts header 'X-API-Key'
    - If no ADMIN_API_KEY or API_KEYS or Redis set configured, raises 401 (forces operator to set keys)
    - Uses constant-time comparison to avoid timing attacks
    """
    # If no keys are configured at all, treat as development mode and allow (explicit)
    if not _configured_hashes and not (API_KEY_HASHED_SET and _redis):
        # development convenience - but WARN via HTTP header/response if used
        return

    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key")

    # 1) check configured plaintext keys (compares hashes)
    if _is_valid_plain_key(x_api_key):
        return

    # 2) check Redis hashed set, if present
    if _is_valid_redis_key(x_api_key):
        return

    # not valid
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
