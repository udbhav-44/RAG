"""
This module provides an HTTP server for file uploads using FastAPI.
"""
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import time
import jwt
import re
import logging
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

UPLOAD_DIR = "user_uploads"  # Directory to store uploaded files
os.makedirs(UPLOAD_DIR, exist_ok=True)
UPLOAD_ROOT = Path(UPLOAD_DIR).resolve()
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
REQUIRE_AUTH = os.getenv("AUTH_REQUIRE_TOKEN", "false").lower() == "true"

def _sanitize_user_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", value.strip())
    if not cleaned:
        raise HTTPException(status_code=400, detail="Invalid user id.")
    return cleaned


def _get_user_id(request: Request) -> str:
    auth_header = request.headers.get("Authorization", "")
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "", 1).strip()
    if not token:
        if REQUIRE_AUTH:
            raise HTTPException(status_code=401, detail="Unauthorized.")
        return "anonymous"
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    user_id = payload.get("sub") or payload.get("email")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    return _sanitize_user_id(str(user_id))


def _user_root(request: Request) -> Path:
    user_id = _get_user_id(request)
    user_dir = (UPLOAD_ROOT / user_id).resolve()
    if UPLOAD_ROOT not in user_dir.parents and user_dir != UPLOAD_ROOT:
        raise HTTPException(status_code=400, detail="Invalid user directory.")
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def _safe_upload_path(root: Path, filename: str) -> Path:
    safe_name = Path(filename).name
    target = (root / safe_name).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return target


def _file_payload(path: Path, owner: str) -> dict:
    stats = path.stat()
    return {
        "name": path.name,
        "size": stats.st_size,
        "modified_at": datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc).isoformat(),
        "owner": owner,
    }


@app.get("/uploads")
async def list_uploads(request: Request):
    files = []
    user_root = _user_root(request)
    if not user_root.exists():
        return {"files": files}
    owner = user_root.name
    for entry in user_root.iterdir():
        if entry.is_file():
            files.append(_file_payload(entry, owner))
    files.sort(key=lambda item: item["modified_at"], reverse=True)
    return {"files": files}


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...), filename: str = Form(None)):
    """
    Uploads a file to the server.

    Args:
        file (UploadFile): The file to be uploaded.
        filename (str): The name of the file.

    Returns:
        dict: A dictionary containing a message and the filename.
    """
    try:
        # Use the provided filename if available, otherwise use the original filename
        original_filename = filename or file.filename
        
        # Check if a file with the same name already exists
        user_root = _user_root(request)
        file_path = _safe_upload_path(user_root, original_filename)
        if file_path.exists():
            # If it exists, add a timestamp to make it unique
            timestamp = int(time.time())
            name, extension = os.path.splitext(original_filename)
            new_filename = f"{name}_{timestamp}{extension}"
            file_path = _safe_upload_path(user_root, new_filename)
        else:
            new_filename = original_filename

        # Save the file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        return {"message": "File uploaded successfully", "filename": new_filename}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.delete("/upload/{filename}")
async def delete_upload(request: Request, filename: str):
    user_root = _user_root(request)
    target = _safe_upload_path(user_root, filename)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    target.unlink()
    return {"message": "File deleted successfully", "filename": target.name}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
