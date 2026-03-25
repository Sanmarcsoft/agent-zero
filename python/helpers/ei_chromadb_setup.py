# Implements MHH-EI specification: "Language Enabled Emotional Intelligence
# and Theory of Mind Algorithms" by Sean Webb (AGPL-3.0)
# Webb Equation of Emotion: EP ∆ P = ER
#
# One-time setup script for ChromaDB collections used by the EI system.
# Run: python -m python.helpers.ei_chromadb_setup

from __future__ import annotations

import os
import chromadb

EI_CHROMADB_HOST = os.environ.get("EI_CHROMADB_HOST", "10.0.0.12")
EI_CHROMADB_PORT = int(os.environ.get("EI_CHROMADB_PORT", "18000"))

COLLECTIONS = {
    "ei_self_maps": {
        "description": "Per-user {self} map identity attachments (quadrant, power_level, valence)",
    },
    "ei_emotion_log": {
        "description": "Longitudinal Webb Equation emotion events",
    },
    "ei_tom_profiles": {
        "description": "Third-party Theory of Mind emotional models",
    },
}


def get_client() -> chromadb.HttpClient:
    return chromadb.HttpClient(host=EI_CHROMADB_HOST, port=EI_CHROMADB_PORT)


def setup_collections(client: chromadb.HttpClient | None = None) -> dict:
    """Create or retrieve all EI collections. Returns dict of collection objects."""
    if client is None:
        client = get_client()

    collections = {}
    for name, meta in COLLECTIONS.items():
        coll = client.get_or_create_collection(
            name=name,
            metadata={"description": meta["description"]},
        )
        collections[name] = coll
        print(f"Collection '{name}': {coll.count()} documents")

    return collections


def verify_connection(client: chromadb.HttpClient | None = None) -> bool:
    """Verify ChromaDB is reachable."""
    if client is None:
        client = get_client()
    try:
        client.heartbeat()
        return True
    except Exception as e:
        print(f"ChromaDB connection failed: {e}")
        return False


if __name__ == "__main__":
    print(f"Connecting to ChromaDB at {EI_CHROMADB_HOST}:{EI_CHROMADB_PORT}...")
    client = get_client()

    if not verify_connection(client):
        print("ERROR: Cannot reach ChromaDB. Check host/port.")
        exit(1)

    print("Connection OK. Setting up collections...")
    setup_collections(client)
    print("Done.")
