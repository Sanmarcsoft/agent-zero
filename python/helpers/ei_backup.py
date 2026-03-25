# Implements MHH-EI specification: "Language Enabled Emotional Intelligence
# and Theory of Mind Algorithms" by Sean Webb (AGPL-3.0)
# Webb Equation of Emotion: EP ∆ P = ER
#
# GPG-encrypted backup of emotional data from ChromaDB to git repo.

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone

import chromadb

EI_CHROMADB_HOST = os.environ.get("EI_CHROMADB_HOST", "10.0.0.12")
EI_CHROMADB_PORT = int(os.environ.get("EI_CHROMADB_PORT", "18000"))
EI_IDENTITY_REPO_PATH = os.environ.get(
    "EI_IDENTITY_REPO_PATH",
    "/var/lib/agent-zero/ei-identity"
)
EI_BACKUP_GPG_KEY = os.environ.get("EI_BACKUP_GPG_KEY", "")


def _get_gpg_key_id() -> str:
    """Retrieve GPG key ID for backup encryption from pass or env."""
    if EI_BACKUP_GPG_KEY:
        return EI_BACKUP_GPG_KEY

    try:
        result = subprocess.run(
            ["pass", "show", "ei-identity/backup-key"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    # Fallback: use default GPG key
    try:
        result = subprocess.run(
            ["gpg", "--list-secret-keys", "--keyid-format=long"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "sec" in line and "/" in line:
                    return line.split("/")[1].split(" ")[0]
    except Exception:
        pass

    return ""


def _export_collection(collection_name: str, user_id: str) -> list[dict]:
    """Export all documents for a user from a ChromaDB collection."""
    client = chromadb.HttpClient(host=EI_CHROMADB_HOST, port=EI_CHROMADB_PORT)
    try:
        coll = client.get_collection(name=collection_name)
    except Exception:
        return []

    results = coll.get(
        where={"user_id": user_id},
        include=["metadatas", "documents"],
    )

    items = []
    if results and results["ids"]:
        for i, doc_id in enumerate(results["ids"]):
            item = {
                "id": doc_id,
                "document": results["documents"][i] if results["documents"] else "",
            }
            if results["metadatas"] and results["metadatas"][i]:
                item.update(results["metadatas"][i])
            items.append(item)

    return items


def _gpg_encrypt(data: str, recipient: str) -> bytes:
    """Encrypt data string with GPG."""
    result = subprocess.run(
        ["gpg", "--encrypt", "--recipient", recipient, "--trust-model", "always", "--armor"],
        input=data.encode("utf-8"),
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GPG encryption failed: {result.stderr.decode()}")
    return result.stdout


def _gpg_decrypt(encrypted_data: bytes) -> str:
    """Decrypt GPG-encrypted data."""
    result = subprocess.run(
        ["gpg", "--decrypt"],
        input=encrypted_data,
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GPG decryption failed: {result.stderr.decode()}")
    return result.stdout.decode("utf-8")


def backup_user(user_id: str, repo_path: str | None = None) -> dict:
    """
    Export all emotional data for a user from ChromaDB, GPG-encrypt, and save to git repo.

    Returns dict with status and file paths.
    """
    repo_path = repo_path or EI_IDENTITY_REPO_PATH
    gpg_key = _get_gpg_key_id()
    if not gpg_key:
        return {"status": "error", "message": "No GPG key found for backup encryption"}

    backup_dir = os.path.join(repo_path, "backups", user_id)
    os.makedirs(backup_dir, exist_ok=True)

    collections = {
        "self_map": "ei_self_maps",
        "emotion_log": "ei_emotion_log",
        "tom_profiles": "ei_tom_profiles",
    }

    files_written = []
    for name, coll_name in collections.items():
        data = _export_collection(coll_name, user_id)
        json_str = json.dumps(data, indent=2, default=str)
        encrypted = _gpg_encrypt(json_str, gpg_key)

        gpg_path = os.path.join(backup_dir, f"{name}.json.gpg")
        with open(gpg_path, "wb") as f:
            f.write(encrypted)
        files_written.append(gpg_path)

    # Git commit and push
    try:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        subprocess.run(
            ["git", "add", "-A"],
            cwd=repo_path,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            ["git", "commit", "-m", f"Backup emotional data for {user_id} at {timestamp}"],
            cwd=repo_path,
            capture_output=True,
            timeout=30,
        )
        subprocess.run(
            ["git", "push"],
            cwd=repo_path,
            capture_output=True,
            timeout=60,
        )
    except Exception:
        pass  # Git failures are non-fatal — data is still saved locally

    return {
        "status": "ok",
        "user_id": user_id,
        "files": files_written,
        "gpg_key": gpg_key,
    }


def restore_user(user_id: str, repo_path: str | None = None) -> dict:
    """
    Decrypt backup files and re-import to ChromaDB.

    Returns dict with status and counts.
    """
    repo_path = repo_path or EI_IDENTITY_REPO_PATH
    backup_dir = os.path.join(repo_path, "backups", user_id)

    if not os.path.isdir(backup_dir):
        return {"status": "error", "message": f"No backup directory for {user_id}"}

    client = chromadb.HttpClient(host=EI_CHROMADB_HOST, port=EI_CHROMADB_PORT)

    collections = {
        "self_map": "ei_self_maps",
        "emotion_log": "ei_emotion_log",
        "tom_profiles": "ei_tom_profiles",
    }

    restored = {}
    for name, coll_name in collections.items():
        gpg_path = os.path.join(backup_dir, f"{name}.json.gpg")
        if not os.path.exists(gpg_path):
            restored[name] = 0
            continue

        with open(gpg_path, "rb") as f:
            encrypted = f.read()

        decrypted = _gpg_decrypt(encrypted)
        items = json.loads(decrypted)

        if not items:
            restored[name] = 0
            continue

        coll = client.get_or_create_collection(name=coll_name)

        # Delete existing data for this user
        try:
            existing = coll.get(where={"user_id": user_id})
            if existing["ids"]:
                coll.delete(ids=existing["ids"])
        except Exception:
            pass

        # Insert restored data
        ids = []
        documents = []
        metadatas = []
        for item in items:
            item_id = item.pop("id", item.get("item_id", ""))
            doc = item.pop("document", "")
            if not item_id:
                continue
            ids.append(item_id)
            documents.append(doc)
            metadatas.append(item)

        if ids:
            coll.add(ids=ids, documents=documents, metadatas=metadatas)

        restored[name] = len(ids)

    return {
        "status": "ok",
        "user_id": user_id,
        "restored": restored,
    }
