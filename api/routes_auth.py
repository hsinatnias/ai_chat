# api/routes_auth.py (minimal)
from fastapi import APIRouter, Depends, Response, HTTPException, Form
from db import get_async_db
from db.models import User  # if you have a users table
from auth.sessions import create_session, set_session_cookie

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/login")
async def login(response: Response, email: str = Form(...), db=Depends(get_async_db)):
    # find user by email (adjust query / password handling as needed)
    q = await db.execute("SELECT id FROM users WHERE email = :email", {"email": email})
    row = q.first()
    if not row:
        raise HTTPException(status_code=401, detail="invalid credentials")
    user_id = row[0]
    sid = await create_session(user_id)
    set_session_cookie(response, sid)
    return {"ok": True, "session_id": sid}

@router.post("/logout")
async def logout(response: Response, session = Depends(current_session)):
    sid = session.get("session_id")
    if sid:
        await destroy_session(sid)
    # clear cookie
    response.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return {"ok": True}
