# helpers.py
import os
import uuid
import hashlib
import hmac
import base64
import secrets
from typing import Tuple
from fastapi import Header, HTTPException

# import the one true get_async_db from your db package
from db import get_async_db    # keep using the project's DB helper

# PBKDF2 parameters
_PBKDF2_ITER = 200_000
_SALT_BYTES = 16
_DERIVED_KEY_BYTES = 32  # 256-bit

def _b64(s: bytes) -> str:
    return base64.b64encode(s).decode("ascii")

def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))

def _derive(password: str, salt: bytes, iterations: int = _PBKDF2_ITER) -> bytes:
    """Return raw derived key bytes for given password+salt."""
    if password is None:
        raise ValueError("password required")
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen=_DERIVED_KEY_BYTES)

def hash_password(plain_password: str) -> str:
    """
    Hash a password using PBKDF2-HMAC-SHA256.
    Output format: pbkdf2_sha256$<iters>$<salt_b64>$<dk_b64>
    """
    if plain_password is None:
        raise ValueError("plain_password is required")
    salt = secrets.token_bytes(_SALT_BYTES)
    dk = _derive(plain_password, salt, _PBKDF2_ITER)
    return f"pbkdf2_sha256${_PBKDF2_ITER}${_b64(salt)}${_b64(dk)}"

def verify_password(plain_password: str, password_hash: str) -> bool:
    """
    Verify password. Supports:
      - our PBKDF2 format: pbkdf2_sha256$iters$salt_b64$dk_b64
      - fallback: if passlib is installed and the hash doesn't match our format,
        attempt to verify via passlib (backwards-compat).
    Returns False on any error.
    """
    if not password_hash or plain_password is None:
        return False

    try:
        if password_hash.startswith("pbkdf2_sha256$"):
            parts = password_hash.split("$")
            if len(parts) != 4:
                return False
            _, iters_s, salt_b64, dk_b64 = parts
            iters = int(iters_s)
            salt = _unb64(salt_b64)
            expected_dk = _unb64(dk_b64)
            actual = _derive(plain_password, salt, iters)
            return hmac.compare_digest(actual, expected_dk)

        # Fallback: try passlib (existing bcrypt hashes)
        try:
            from passlib.context import CryptContext
            # Only attempt verify — do not change hashing behaviour here
            pwd_ctx = CryptContext(schemes=["bcrypt", "bcrypt_sha256"], deprecated="auto")
            return pwd_ctx.verify(plain_password, password_hash)
        except Exception:
            return False

    except Exception:
        return False

def gen_user_id() -> str:
    return str(uuid.uuid4())

# admin api key comes from env (set ADMIN_API_KEY in .env or system env)
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "dev-key-change-me")

async def require_admin_key(x_api_key: str = Header(None, alias="X-API-Key")):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key")
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid X-API-Key")
    return True
