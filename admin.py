# admin.py
import os
import json
import shutil
import subprocess
import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sqlalchemy
from fastapi import APIRouter, Depends, Path as FPath, HTTPException, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse

# imported helpers/deps from your project
from helpers import require_admin_key, get_async_db, hash_password, gen_user_id, ADMIN_API_KEY
from app_security import require_api_key   # existing API-key check used elsewhere
from database import create_table, add_module, get_modules, delete_module

router = APIRouter(prefix="/admin", tags=["admin"])
public_admin_router = APIRouter()  # no prefix -> serves /admin-login

# env-backed config defaults (same as app.py)
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
ADMIN_INGEST_SCRIPT = os.getenv("ADMIN_INGEST_SCRIPT", "ingest.py")
STATUS_DIR = os.getenv("INGEST_STATUS_DIR", "data/ingest")
STATUS_FILE = os.path.join(STATUS_DIR, "ingest_status.json")
INGEST_LOG = os.path.join(STATUS_DIR, "ingest.log")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "kb_chunks")


# -----------------------
# Users endpoints (your existing code)
# -----------------------
@router.get("/users")
async def admin_list_users(ok: bool = Depends(require_admin_key), db = Depends(get_async_db)):
    try:
        q = await db.execute(sqlalchemy.text(
            "SELECT id, email, name, is_active, created_at FROM users ORDER BY created_at DESC"
        ))
        rows = q.fetchall()
        users = []
        for r in rows:
            users.append({
                "id": r[0],
                "email": r[1],
                "name": r[2],
                "is_active": bool(r[3]) if r[3] is not None else False,
                "created_at": r[4].isoformat() if hasattr(r[4], "isoformat") else r[4]
            })
        return JSONResponse(users)
    except Exception:
        return JSONResponse({"error": "db error"}, status_code=500)


@router.post("/users")
async def admin_create_user(body: dict, ok: bool = Depends(require_admin_key), db = Depends(get_async_db)):
    email = (body.get("email") if isinstance(body, dict) else None) or None
    password = (body.get("password") if isinstance(body, dict) else None) or None
    name = (body.get("name") if isinstance(body, dict) else "") or ""
    is_active = body.get("is_active", True) if isinstance(body, dict) else True

    if not email:
        return JSONResponse({"ok": False, "error": "email required"}, status_code=400)
    if not password:
        return JSONResponse({"ok": False, "error": "password required"}, status_code=400)

    try:
        q = await db.execute(sqlalchemy.text("SELECT 1 FROM users WHERE email = :email"), {"email": email})
        if q.first():
            return JSONResponse({"ok": False, "error": "email already exists"}, status_code=409)
    except Exception:
        return JSONResponse({"ok": False, "error": "db error"}, status_code=500)

    uid = gen_user_id()
    pw_hash = hash_password(password)
    now = datetime.datetime.utcnow()
    try:
        await db.execute(sqlalchemy.text(
            "INSERT INTO users (id, email, name, password_hash, is_active, created_at) "
            "VALUES (:id, :email, :name, :password_hash, :is_active, :created_at)"
        ), {
            "id": uid,
            "email": email,
            "name": name,
            "password_hash": pw_hash,
            "is_active": bool(is_active),
            "created_at": now
        })
        await db.commit()
        return JSONResponse({"ok": True, "id": uid})
    except Exception:
        return JSONResponse({"ok": False, "error": "db error"}, status_code=500)


@router.put("/users/{user_id}")
async def admin_update_user(user_id: str = FPath(...), body: dict = None, ok: bool = Depends(require_admin_key), db = Depends(get_async_db)):
    if not isinstance(body, dict):
        return JSONResponse({"ok": False, "error": "invalid body"}, status_code=400)

    allowed = {}
    if "email" in body and body["email"]:
        allowed["email"] = body["email"]
    if "name" in body:
        allowed["name"] = body["name"] or ""
    if "is_active" in body:
        allowed["is_active"] = bool(body["is_active"])
    if "password" in body and body["password"]:
        allowed["password_hash"] = hash_password(body["password"])

    if not allowed:
        return JSONResponse({"ok": False, "error": "nothing to update"}, status_code=400)

    set_parts = []
    params = {"id": user_id}
    for k, v in allowed.items():
        set_parts.append(f"{k} = :{k}")
        params[k] = v

    sql = "UPDATE users SET " + ", ".join(set_parts) + " WHERE id = :id"
    try:
        await db.execute(sqlalchemy.text(sql), params)
        await db.commit()
        return JSONResponse({"ok": True})
    except Exception:
        return JSONResponse({"ok": False, "error": "db error"}, status_code=500)


@router.delete("/users/{user_id}")
async def admin_delete_user(user_id: str = FPath(...), ok: bool = Depends(require_admin_key), db = Depends(get_async_db)):
    try:
        await db.execute(sqlalchemy.text("DELETE FROM users WHERE id = :id"), {"id": user_id})
        await db.commit()
        return JSONResponse({"ok": True})
    except Exception:
        return JSONResponse({"ok": False, "error": "db error"}, status_code=500)


# -----------------------
# Public admin login page (keeps /admin-login)
# -----------------------
@public_admin_router.get("/admin-login", include_in_schema=False)
def admin_login_page():
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
        sessionStorage.setItem('dev_api_key', key);
        log('Key validated. Fetching admin interface...');
        const rc = await fetch('/admin/content', { headers: { 'X-API-Key': key } });
        if(!rc.ok){
          log('Failed to fetch admin content: ' + rc.status);
          alert('Failed to fetch admin interface (check server logs)');
          return;
        }
        const html = await rc.text();
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


# -----------------------
# Admin auth + content endpoints
# -----------------------
@router.post("/auth", include_in_schema=False)
async def admin_auth(body: Dict):
    key = (body.get("key") if isinstance(body, dict) else None) or None
    if not key:
        return JSONResponse({"ok": False, "error": "key required"}, status_code=400)
    if key != ADMIN_API_KEY:
        return JSONResponse({"ok": False, "error": "invalid key"}, status_code=401)
    return JSONResponse({"ok": True})


@router.get("/content", include_in_schema=False)
async def admin_content(ok: bool = Depends(require_admin_key)):
    p = Path("public") / "admin.html"
    if not p.exists():
        raise HTTPException(status_code=404, detail="admin.html not found.")
    return FileResponse(str(p))


# -----------------------
# Modules / Upload / Ingest / Delete collection
# -----------------------
@router.get("/modules", response_model=List[str], include_in_schema=False)
async def get_modules_list(ok: bool = Depends(require_admin_key)):
    try:
        modules = get_modules()
        return modules
    except Exception:
        return JSONResponse({"ok": False, "error": "db error"}, status_code=500)


@router.post("/modules", include_in_schema=False)
async def add_new_module(module_name: str = Form(...), ok: bool = Depends(require_admin_key)):
    try:
        existing_modules = get_modules()
        if module_name in existing_modules:
            raise HTTPException(status_code=400, detail="Module already exists")
        add_module(module_name)
        _append_admin_log(f"Module added: {module_name}")
        _write_admin_status({"pid": None, "status": "module_added", "module": module_name})
        return JSONResponse({"ok": True, "module": module_name})
    except HTTPException:
        raise
    except Exception:
        return JSONResponse({"ok": False, "error": "db error"}, status_code=500)


@router.delete("/modules/{module_name}", include_in_schema=False)
async def delete_existing_module(module_name: str = FPath(...), ok: bool = Depends(require_admin_key)):
    try:
        modules = get_modules()
        if module_name not in modules:
            raise HTTPException(status_code=404, detail="Module not found")
        delete_module(module_name)
        _append_admin_log(f"Module deleted: {module_name}")
        _write_admin_status({"pid": None, "status": "module_deleted", "module": module_name})
        return JSONResponse({"ok": True, "deleted_module": module_name})
    except HTTPException:
        raise
    except Exception:
        return JSONResponse({"ok": False, "error": "db error"}, status_code=500)


@router.post("/upload", dependencies=[Depends(require_api_key)])
async def admin_upload(request: Request, file: UploadFile = File(...), module: str = Form(...)):
    modules = get_modules()
    if module not in modules:
        raise HTTPException(status_code=404, detail=f"Module not found: {module}")

    target_dir = Path(DOCS_DIR) / module
    target_dir.mkdir(parents=True, exist_ok=True)

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


@router.post("/ingest", dependencies=[Depends(require_api_key)])
async def admin_ingest(body: Dict = None):
    module = None
    if body:
        module = body.get("module") or body.get("module_name")

    if not module:
        raise HTTPException(status_code=400, detail="Missing module in request body")

    modules = get_modules()
    if module not in modules:
        raise HTTPException(status_code=404, detail=f"Module not found: {module}")

    script = ADMIN_INGEST_SCRIPT
    if not os.path.exists(script):
        return JSONResponse({"ok": False, "error": f"ingest script not found at {script}"}, status_code=500)

    cmd = [shutil.which("python") or "python", script]
    env = os.environ.copy()
    env["INGEST_MODULE"] = module

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        _append_admin_log(f"Started ingest subprocess pid={proc.pid} module={module} cmd={' '.join(cmd)}")
        _write_admin_status({"pid": proc.pid, "status": "started", "module": module, "cmd": cmd})
        return JSONResponse({"ok": True, "pid": proc.pid, "cmd": cmd, "module": module})
    except Exception as e:
        _append_admin_log(f"Failed to start ingest subprocess: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.post("/delete_collection", dependencies=[Depends(require_api_key)])
async def admin_delete_collection(request: Request, body: Dict):
    if not body or not body.get("confirm"):
        raise HTTPException(status_code=400, detail="Missing confirm in body")

    module = body.get("module")
    confirm_all = bool(body.get("confirm_all"))

    try:
        qc = request.app.state.qc
        cols_resp = qc.get_collections()
        cols = [c.name for c in getattr(cols_resp, "collections", [])]

        result = {"ok": True, "module": module, "collections_before": cols, "deleted_collections": [], "deleted_points_summary": {}}

        if module:
            if QDRANT_COLLECTION not in cols:
                result["ok"] = False
                result["message"] = f"Collection {QDRANT_COLLECTION} not found"
                _append_admin_log(result["message"])
                return JSONResponse(result)

            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                filt = Filter(must=[FieldCondition(key="module", match=MatchValue(value=module))])
            except Exception:
                filt = {"must": [{"key": "module", "match": {"value": module}}]}

            PointsSelector = None
            try:
                from qdrant_client.http.models import PointsSelector as _PS
                PointsSelector = _PS
            except Exception:
                try:
                    from qdrant_client.models import PointsSelector as _PS
                    PointsSelector = _PS
                except Exception:
                    PointsSelector = None

            try:
                if PointsSelector is not None:
                    ps = PointsSelector(filter=filt)
                    res = qc.delete(collection_name=QDRANT_COLLECTION, points_selector=ps)
                else:
                    res = qc.delete(collection_name=QDRANT_COLLECTION, filter=filt)
                _append_admin_log(f"Deleted points for module={module} from collection {QDRANT_COLLECTION} -> {res}")
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
            deleted = []
            errors = {}
            for col in [QDRANT_COLLECTION] + (os.getenv("QDRANT_CACHE_COLLECTION", "kb_cache"), ):
                try:
                    if col in cols:
                        qc.delete_collection(collection_name=col)
                        deleted.append(col)
                        _append_admin_log(f"Deleted collection {col}")
                except Exception as e:
                    errors[col] = str(e)
                    _append_admin_log(f"Failed deleting collection {col}: {e}")
            result["deleted_collections"] = deleted
            result["errors"] = errors
            return JSONResponse(result)

        raise HTTPException(status_code=400, detail="Specify 'module' or 'confirm_all': true")
    except Exception as e:
        _append_admin_log(f"Unexpected error while deleting collections: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# -----------------------
# Ingest status + helpers
# -----------------------
def _is_pid_alive(pid: int) -> bool:
    try:
        try:
            import psutil
            return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
        except Exception:
            if os.name == "posix":
                os.kill(pid, 0)
                return True
            return True
    except OSError:
        return False
    except Exception:
        return True


@router.get("/ingest_status", dependencies=[Depends(require_api_key)])
async def admin_ingest_status():
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

    tail = ""
    try:
        if os.path.exists(INGEST_LOG):
            with open(INGEST_LOG, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
                tail = "\n".join(lines[-200:])
    except Exception:
        tail = ""

    st["pid_alive"] = bool(alive)
    st["log_tail"] = tail
    return JSONResponse(st)


def _append_admin_log(msg: str):
    try:
        os.makedirs(os.path.dirname(INGEST_LOG), exist_ok=True)
        with open(INGEST_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.datetime.utcnow().isoformat()}Z] {msg}\n")
    except Exception:
        # avoid crashing admin functions if logging fails
        pass


def _write_admin_status(data: Dict):
    try:
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
        payload = dict(data)
        payload["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        with open(STATUS_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


# convenience for importing log helper from other modules
def admin_append_log(msg: str):
    _append_admin_log(msg)
