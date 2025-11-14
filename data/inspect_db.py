# inspect_db.py
import sqlite3, os
from pprint import pprint

DB_PATH = os.getenv("DATABASE_URL_PATH", "")
if not DB_PATH:
    # fallback if you used DATABASE_URL like sqlite+aiosqlite:///data/app.db
    env = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///data/app.db")
    if env.startswith("sqlite+aiosqlite:///"):
        DB_PATH = env.split("sqlite+aiosqlite:///")[-1]
    else:
        DB_PATH = "data/app.db"
DB_PATH = os.path.abspath(DB_PATH)

if not os.path.exists(DB_PATH):
    print("DB file not found:", DB_PATH)
    raise SystemExit(1)

print("Using DB:", DB_PATH)
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")
tables = cur.fetchall()
print("\n--- Tables ---")
for t in tables:
    print("-", t["name"])

def show_table(name, limit=5):
    try:
        cur.execute(f"SELECT COUNT(*) AS c FROM {name}")
        cnt = cur.fetchone()["c"]
    except Exception as e:
        print(f"\n{name}: cannot count ({e})")
        return
    print(f"\n--- {name} rows={cnt} sample:")
    try:
        cur.execute(f"SELECT * FROM {name} LIMIT {limit}")
        rows = cur.fetchall()
        for r in rows:
            pprint(dict(r))
    except Exception as e:
        print("  (error selecting sample)", e)

for name in ["conversations", "messages", "message_feedback"]:
    if any(t["name"] == name for t in tables):
        show_table(name)

cur.close()
conn.close()
print("\nDone.")
