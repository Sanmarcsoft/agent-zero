# Implements MHH-EI specification: "Language Enabled Emotional Intelligence
# and Theory of Mind Algorithms" by Sean Webb (AGPL-3.0)
# Webb Equation of Emotion: EP ∆ P = ER

import json
import os
import uuid
from datetime import datetime, timezone

import chromadb

from helpers.tool import Tool, Response

EI_CHROMADB_HOST = os.environ.get("EI_CHROMADB_HOST", "10.0.0.12")
EI_CHROMADB_PORT = int(os.environ.get("EI_CHROMADB_PORT", "18000"))
COLLECTION_NAME = "ei_self_maps"

VALID_QUADRANTS = ["people", "accomplishments", "life_story", "ideas_likes"]
VALID_SOURCES = ["inferred", "user_stated", "agent_observed"]


def _get_collection():
    client = chromadb.HttpClient(host=EI_CHROMADB_HOST, port=EI_CHROMADB_PORT)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def _cache_key(user_id: str) -> str:
    return f"ei_self_map_{user_id}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SelfMapManager(Tool):

    async def execute(self, action="", user_id="", **kwargs):
        if not user_id:
            user_id = self._resolve_user_id()
        if not action:
            return Response(
                message="Error: 'action' is required. Valid actions: load, add, update, remove, list, restore",
                break_loop=False,
            )

        action = action.strip().lower()
        try:
            if action == "load":
                return await self._load(user_id)
            elif action == "add":
                return await self._add(user_id, **kwargs)
            elif action == "update":
                return await self._update(user_id, **kwargs)
            elif action == "remove":
                return await self._remove(user_id, **kwargs)
            elif action == "list":
                return await self._list(user_id)
            elif action == "restore":
                return await self._restore(user_id, **kwargs)
            else:
                return Response(
                    message=f"Unknown action '{action}'. Valid: load, add, update, remove, list, restore",
                    break_loop=False,
                )
        except Exception as e:
            return Response(message=f"self_map_manager error: {e}", break_loop=False)

    def _resolve_user_id(self) -> str:
        """Resolve user_id from agent context or default."""
        uid = self.agent.get_data("ei_user_id")
        if uid:
            return uid
        uid = self.agent.context.get_data("ei_user_id")
        if uid:
            return uid
        return "default"

    async def _load(self, user_id: str) -> Response:
        """Load entire {self} map for user from ChromaDB, cache in agent context."""
        collection = _get_collection()
        results = collection.get(
            where={"user_id": user_id},
            include=["metadatas", "documents"],
        )

        items = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                content = results["documents"][i] if results["documents"] else ""
                items.append({"id": doc_id, "content": content, **meta})

        # Cache in agent context
        self.agent.set_data(_cache_key(user_id), items)
        self.agent.set_data("ei_user_id", user_id)

        if not items:
            return Response(
                message=f"No {{self}} map found for user '{user_id}'. Map is empty — items will be added as the user reveals identity attachments.",
                break_loop=False,
            )

        summary = self._format_map_summary(items)
        return Response(
            message=f"Loaded {{self}} map for '{user_id}' ({len(items)} items):\n{summary}",
            break_loop=False,
        )

    async def _add(self, user_id: str, **kwargs) -> Response:
        """Add a new item to the {self} map."""
        label = kwargs.get("label", "")
        if not label:
            return Response(message="Error: 'label' is required for add.", break_loop=False)

        quadrant = kwargs.get("quadrant", "ideas_likes")
        if quadrant not in VALID_QUADRANTS:
            return Response(
                message=f"Error: quadrant must be one of {VALID_QUADRANTS}",
                break_loop=False,
            )

        power_level = int(kwargs.get("power_level", 5))
        power_level = max(1, min(10, power_level))
        valence = int(kwargs.get("valence", 0))
        valence = max(-10, min(10, valence))
        source = kwargs.get("source", "inferred")
        if source not in VALID_SOURCES:
            source = "inferred"
        associations = kwargs.get("associations", "")

        item_id = str(uuid.uuid4())
        now = _now_iso()

        metadata = {
            "user_id": user_id,
            "item_id": item_id,
            "label": label,
            "quadrant": quadrant,
            "power_level": power_level,
            "valence": valence,
            "associations": associations,
            "source": source,
            "created_at": now,
            "updated_at": now,
        }

        collection = _get_collection()
        collection.add(
            ids=[item_id],
            documents=[f"{label} ({quadrant}, power={power_level}, valence={valence})"],
            metadatas=[metadata],
        )

        # Update cache
        cached = self.agent.get_data(_cache_key(user_id)) or []
        cached.append({"id": item_id, **metadata})
        self.agent.set_data(_cache_key(user_id), cached)

        return Response(
            message=f"Added '{label}' to {{self}} map: quadrant={quadrant}, power={power_level}, valence={valence}, source={source}",
            break_loop=False,
        )

    async def _update(self, user_id: str, **kwargs) -> Response:
        """Update an existing {self} map item."""
        item_id = kwargs.get("item_id", "")
        if not item_id:
            return Response(message="Error: 'item_id' is required for update.", break_loop=False)

        collection = _get_collection()
        existing = collection.get(ids=[item_id], include=["metadatas", "documents"])
        if not existing["ids"]:
            return Response(message=f"Item '{item_id}' not found.", break_loop=False)

        meta = existing["metadatas"][0]
        if meta.get("user_id") != user_id:
            return Response(message="Error: item belongs to a different user.", break_loop=False)

        # Apply updates
        for field in ["label", "quadrant", "power_level", "valence", "associations", "source"]:
            if field in kwargs and kwargs[field] != "":
                val = kwargs[field]
                if field == "power_level":
                    val = max(1, min(10, int(val)))
                elif field == "valence":
                    val = max(-10, min(10, int(val)))
                elif field == "quadrant" and val not in VALID_QUADRANTS:
                    continue
                meta[field] = val

        meta["updated_at"] = _now_iso()

        label = meta.get("label", "")
        doc_text = f"{label} ({meta.get('quadrant')}, power={meta.get('power_level')}, valence={meta.get('valence')})"

        collection.update(ids=[item_id], documents=[doc_text], metadatas=[meta])

        # Refresh cache
        cached = self.agent.get_data(_cache_key(user_id)) or []
        for i, item in enumerate(cached):
            if item.get("item_id") == item_id or item.get("id") == item_id:
                cached[i] = {"id": item_id, **meta}
                break
        self.agent.set_data(_cache_key(user_id), cached)

        return Response(
            message=f"Updated '{label}': power={meta.get('power_level')}, valence={meta.get('valence')}",
            break_loop=False,
        )

    async def _remove(self, user_id: str, **kwargs) -> Response:
        """Remove an item from the {self} map."""
        item_id = kwargs.get("item_id", "")
        if not item_id:
            return Response(message="Error: 'item_id' is required for remove.", break_loop=False)

        collection = _get_collection()
        existing = collection.get(ids=[item_id], include=["metadatas"])
        if not existing["ids"]:
            return Response(message=f"Item '{item_id}' not found.", break_loop=False)

        meta = existing["metadatas"][0]
        if meta.get("user_id") != user_id:
            return Response(message="Error: item belongs to a different user.", break_loop=False)

        label = meta.get("label", item_id)
        collection.delete(ids=[item_id])

        # Update cache
        cached = self.agent.get_data(_cache_key(user_id)) or []
        cached = [item for item in cached if item.get("item_id") != item_id and item.get("id") != item_id]
        self.agent.set_data(_cache_key(user_id), cached)

        return Response(
            message=f"Removed '{label}' from {{self}} map.",
            break_loop=False,
        )

    async def _list(self, user_id: str) -> Response:
        """List all {self} map items for user."""
        cached = self.agent.get_data(_cache_key(user_id))
        if cached is None:
            # Not yet loaded — load from ChromaDB
            return await self._load(user_id)

        if not cached:
            return Response(
                message=f"{{self}} map for '{user_id}' is empty.",
                break_loop=False,
            )

        summary = self._format_map_summary(cached)
        return Response(
            message=f"{{self}} map for '{user_id}' ({len(cached)} items):\n{summary}",
            break_loop=False,
        )

    async def _restore(self, user_id: str, **kwargs) -> Response:
        """Restore {self} map from decrypted backup JSON."""
        backup_data = kwargs.get("backup_data", "")
        if not backup_data:
            return Response(
                message="Error: 'backup_data' required — provide decrypted JSON array of {self} map items.",
                break_loop=False,
            )

        try:
            items = json.loads(backup_data) if isinstance(backup_data, str) else backup_data
        except json.JSONDecodeError as e:
            return Response(message=f"Error parsing backup data: {e}", break_loop=False)

        if not isinstance(items, list):
            return Response(message="Error: backup_data must be a JSON array.", break_loop=False)

        collection = _get_collection()

        # Delete existing items for this user
        existing = collection.get(where={"user_id": user_id})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        # Insert backup items
        ids = []
        documents = []
        metadatas = []
        for item in items:
            item_id = item.get("item_id", str(uuid.uuid4()))
            label = item.get("label", "unknown")
            ids.append(item_id)
            documents.append(
                f"{label} ({item.get('quadrant', 'ideas_likes')}, "
                f"power={item.get('power_level', 5)}, valence={item.get('valence', 0)})"
            )
            meta = {
                "user_id": user_id,
                "item_id": item_id,
                "label": label,
                "quadrant": item.get("quadrant", "ideas_likes"),
                "power_level": int(item.get("power_level", 5)),
                "valence": int(item.get("valence", 0)),
                "associations": item.get("associations", ""),
                "source": item.get("source", "inferred"),
                "created_at": item.get("created_at", _now_iso()),
                "updated_at": item.get("updated_at", _now_iso()),
            }
            metadatas.append(meta)

        if ids:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)

        self.agent.set_data(_cache_key(user_id), [{"id": i, **m} for i, m in zip(ids, metadatas)])

        return Response(
            message=f"Restored {len(ids)} {{self}} map items for '{user_id}' from backup.",
            break_loop=False,
        )

    def _format_map_summary(self, items: list) -> str:
        """Format {self} map items grouped by quadrant."""
        by_quadrant: dict[str, list] = {}
        for item in items:
            q = item.get("quadrant", "unknown")
            by_quadrant.setdefault(q, []).append(item)

        lines = []
        for quadrant in VALID_QUADRANTS:
            q_items = by_quadrant.get(quadrant, [])
            if not q_items:
                continue
            lines.append(f"\n**{quadrant.replace('_', ' ').title()}:**")
            # Sort by power level descending
            q_items.sort(key=lambda x: int(x.get("power_level", 0)), reverse=True)
            for item in q_items:
                valence_str = f"+{item.get('valence')}" if int(item.get("valence", 0)) > 0 else str(item.get("valence", 0))
                lines.append(
                    f"  - {item.get('label')} (power={item.get('power_level')}, valence={valence_str})"
                )
        return "\n".join(lines)
