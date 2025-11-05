#ingest.py
import os
import glob
import re
import hashlib
import json
import base64
import zlib
import time
from typing import List, Dict
from pathlib import Path
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langdetect import detect
from bs4 import BeautifulSoup
from pypdf import PdfReader
from docx import Document as DocxDocument

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None

# --- config (tweak via environment) ---
COLL = os.getenv("QDRANT_COLLECTION", "kb_chunks")
QURL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
EMB  = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
INGEST_MODULE = os.getenv("INGEST_MODULE", "").strip()  # populated by admin_ingest
if INGEST_MODULE:
    # restrict docs dir to a subfolder for this module
    DOCS_DIR = os.path.join(DOCS_DIR, INGEST_MODULE)
KB_VERSION = int(os.getenv("KB_VERSION", "1"))

USE_E5 = EMB.lower().startswith("intfloat/multilingual-e5")

# New config for batching / excerpt / chunk dumps
BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH", "64"))
EXCERPT_CHARS = int(os.getenv("PAYLOAD_EXCERPT_CHARS", "1200"))   # visible excerpt
MAX_PAYLOAD_CHARS = int(os.getenv("MAX_PAYLOAD_CHARS", "4000"))   # cap full text before compressing
DOC_DUMP_DIR = os.getenv("DOC_DUMP_DIR", "")  # not used in integrated mode but kept for compatibility

# ingest status/logging
INGEST_STATUS_DIR = os.getenv("INGEST_STATUS_DIR", "data/ingest")
os.makedirs(INGEST_STATUS_DIR, exist_ok=True)
STATUS_FILE = os.path.join(INGEST_STATUS_DIR, "ingest_status.json")
INGEST_LOG = os.path.join(INGEST_STATUS_DIR, "ingest.log")

def _write_status(data: dict):
    try:
        data["updated_at"] = datetime.utcnow().isoformat() + "Z"
        with open(STATUS_FILE, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _append_log(msg: str):
    try:
        with open(INGEST_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.utcnow().isoformat()}Z] {msg}\n")
    except Exception:
        pass

# ---- embedding setup ----
st = SentenceTransformer(EMB)
def embed(texts: List[str]) -> List[List[float]]:
    texts_ = [("passage: " + t) if USE_E5 else t for t in texts]
    return st.encode(texts_, normalize_embeddings=True).tolist()

try:
    from sudachipy import tokenizer, dictionary
    _tok = dictionary.Dictionary().create()
    _mode = tokenizer.Tokenizer.SplitMode.C
    def jp_sentences(text: str) -> List[str]:
        sents = re.split(r'[。！？]\s*', text)
        return [s.strip() for s in sents if s.strip()]
except Exception:
    def jp_sentences(text: str) -> List[str]:
        return [text]

def read_pdf(path: Path) -> List[Dict]:
    reader = PdfReader(str(path))
    pages = []
    for i, p in enumerate(reader.pages):
        txt = p.extract_text() or ""
        pages.append({"page": i+1, "text": txt})
    return pages

def read_docx(path: Path) -> str:
    d = DocxDocument(str(path))
    return "\n".join([p.text for p in d.paragraphs])

def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n")

def chunk_text(text: str, lang_hint: str, max_chars=700, overlap=100) -> List[str]:
    if lang_hint.startswith("ja") or re.search(r'[\u3040-\u30ff\u3400-\u9fff]', text):
        units = jp_sentences(text)
    else:
        units = re.split(r'(?<=[\.\?\!])\s+|\n{2,}', text)
    chunks, buf = [], ""
    for u in units:
        u = u.strip()
        if not u:
            continue
        if len(buf) + len(u) <= max_chars:
            buf += (("\n" if buf else "") + u)
        else:
            if buf: chunks.append(buf)
            buf = u[-max_chars:] if len(u) > max_chars else u
    if buf: chunks.append(buf)
    out = []
    for i, c in enumerate(chunks):
        prev_tail = chunks[i-1][-overlap:] if i>0 else ""
        out.append((prev_tail + c)[-max_chars:])
    return out

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# --- helpers: compress / decompress helpers for payload ---
def compress_text_for_payload(s: str) -> str:
    if not s:
        return ""
    b = s.encode("utf-8")
    c = zlib.compress(b, level=6)
    return base64.b64encode(c).decode("ascii")

def decompress_text_from_payload(b64: str) -> str:
    if not b64:
        return ""
    try:
        c = base64.b64decode(b64.encode("ascii"))
        return zlib.decompress(c).decode("utf-8", errors="ignore")
    except Exception:
        return ""

def safe_doc_id(path: Path) -> str:
    stem = path.stem.replace(" ", "_")
    short = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{short}"

def make_point(id_int: int, vector: list, payload: dict) -> PointStruct:
    return PointStruct(id=id_int, vector=vector, payload=payload)

def upsert_points_in_batches(qc: QdrantClient, points: List[PointStruct], batch_size: int = BATCH_SIZE, collection: str = COLL):
    total = len(points)
    for i in range(0, total, batch_size):
        batch = points[i:i+batch_size]
        try:
            qc.upsert(collection_name=collection, points=batch)
            msg = f"Upserted batch {i//batch_size + 1} ({len(batch)} points)"
            print(msg)
            _append_log(msg)
            # update status after each batch (best-effort)
            try:
                _write_status({"pid": os.getpid(), "status": "running", "progress": f"upserted batch {i//batch_size + 1}", "last_batch_count": len(batch)})
            except Exception:
                pass
        except Exception as e:
            err = f"ERROR upserting batch: {e}"
            print(err)
            _append_log(err)
            raise

# --- sanitize helper (keeps your previous sanitation) ---
def sanitize_document_text(text: str) -> str:
    if not text:
        return ""
    txt = text
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"&nbsp;|&quot;|&lt;|&gt;|&amp;", " ", txt)
    patterns = [
        r"メッセージ本文（任意）は、送信画面で入力します",
        r"※ご利用のメールソフトまたはブラウザによっては、.*",
        r"管理者サイトで作業をする際の推奨環境は、.*",
        r"推奨画面解像度は、.*",
        r"お礼メールについて",
    ]
    for pat in patterns:
        txt = re.sub(pat, " ", txt, flags=re.MULTILINE)
    txt = re.sub(r"\r\n", "\n", txt)
    txt = re.sub(r"\n\s*\n+", "\n\n", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    return txt.strip()

# ---- main ingest flow (integrated: store full text compressed into payload) ----
if __name__ == "__main__":
    # initial status/log
    _write_status({"pid": os.getpid(), "status": "starting", "progress": "initializing", "message": "Ingest starting"})
    _append_log(f"Ingest started (pid={os.getpid()})")

    try:
        qc = QdrantClient(url=QURL)
        points_buffer: List[PointStruct] = []
        pid = 1

        files = [Path(p) for p in glob.glob(os.path.join(DOCS_DIR, "**", "*.*"), recursive=True)
                 if Path(p).suffix.lower() in (".pdf",".docx",".txt",".md",".markdown",".htm",".html")]
        if not files:
            msg = "No docs found under ./docs"
            print(msg)
            _append_log(msg)
            _write_status({"pid": os.getpid(), "status": "idle", "progress": "no_files", "message": msg})
            exit(0)

        file_count = len(files)
        _write_status({"pid": os.getpid(), "status": "running", "progress": f"found {file_count} files", "files_total": file_count})
        _append_log(f"Found {file_count} files to ingest")

        processed = 0
        for f in files:
            processed += 1
            _append_log(f"Processing file {processed}/{file_count}: {f}")
            _write_status({"pid": os.getpid(), "status": "running", "progress": f"processing {processed}/{file_count}", "current_file": str(f), "files_total": file_count})
            ext = f.suffix.lower()
            if ext == ".pdf":
                items = read_pdf(f)
            elif ext == ".docx":
                items = [{"text": read_docx(f), "page": None}]
            elif ext in (".md", ".markdown"):
                items = [{"text": read_md(f), "page": None}]
            elif ext in (".htm", ".html"):
                items = [{"text": read_html(f), "page": None}]
            else:
                items = [{"text": read_txt(f), "page": None}]

            doc_title = f.stem
            doc_id = safe_doc_id(f)

            # Build a short concatenated preview for doc-level language detection
            raw_full_text = ""
            for itx in items:
                raw_full_text += (itx.get("text") or "") + "\n\n"
            try:
                doc_lang = detect(raw_full_text[:4000])
            except Exception:
                doc_lang = "ja" if "ja" in f.parts else "en"

            for it in items:
                raw_text = (it.get("text") or "")
                text = sanitize_document_text(raw_text)
                if not text:
                    continue

                lang = doc_lang

                chunks = chunk_text(text, lang_hint=lang, max_chars=700, overlap=100)
                if not chunks:
                    continue

                vecs = embed(chunks)

                # ensure collection exists with correct vector size
                cols = [c.name for c in qc.get_collections().collections]
                if COLL not in cols:
                    qc.create_collection(COLL, vectors_config=VectorParams(size=len(vecs[0]), distance=Distance.COSINE))
                    _append_log(f"Created collection {COLL} with vector size {len(vecs[0])}")

                for idx, (c, v) in enumerate(zip(chunks, vecs)):
                    # Prepare payload: excerpt + compressed full chunk text (capped)
                    excerpt = (c[:EXCERPT_CHARS] + "...") if len(c) > EXCERPT_CHARS else c
                    # Cap the stored full text to MAX_PAYLOAD_CHARS to avoid massive payloads
                    full_text_for_payload = c if len(c) <= MAX_PAYLOAD_CHARS else c[:MAX_PAYLOAD_CHARS] + "..."
                    compressed = compress_text_for_payload(full_text_for_payload)

                    payload = {
                        "text_excerpt": excerpt,
                        "full_text_z": compressed,
                        "doc_title": doc_title,
                        "source_path": str(f),
                        "lang": lang,
                        "page": it.get("page"),
                        "content_hash": sha1(c),
                        "kb_version": KB_VERSION,
                        "module": INGEST_MODULE or ""
                    }

                    points_buffer.append(make_point(pid, v, payload))
                    pid += 1

                    # When buffer reaches batch size, flush to Qdrant
                    if len(points_buffer) >= BATCH_SIZE:
                        _append_log(f"Flushing {len(points_buffer)} points to Qdrant (batch)")
                        upsert_points_in_batches(qc, points_buffer, batch_size=BATCH_SIZE, collection=COLL)
                        points_buffer = []
                        _write_status({"pid": os.getpid(), "status": "running", "progress": f"processed {processed}/{file_count}", "current_file": str(f), "buffer": 0})

        # Upsert any remaining points
        if points_buffer:
            _append_log(f"Flushing remaining {len(points_buffer)} points to Qdrant")
            upsert_points_in_batches(qc, points_buffer, batch_size=BATCH_SIZE, collection=COLL)
            points_buffer = []

        # finish ok
        _append_log("Ingest completed successfully")
        _write_status({"pid": os.getpid(), "status": "done", "progress": "completed", "message": "Ingest completed"})
        print("Ingest completed.")

    except Exception as e:
        _append_log("Ingest error: %s" % (str(e),))
        _write_status({"pid": os.getpid(), "status": "error", "progress": "failed", "error": str(e)})
        raise
