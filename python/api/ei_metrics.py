# Implements MHH-EI specification: "Language Enabled Emotional Intelligence
# and Theory of Mind Algorithms" by Sean Webb (AGPL-3.0)
# Webb Equation of Emotion: EP ∆ P = ER
#
# API endpoint for EI metrics dashboard.

import os
import json
from collections import Counter
from datetime import datetime, timezone

from python.helpers.api import ApiHandler, Input, Output
from flask import Request, Response

EI_CHROMADB_HOST = os.environ.get("EI_CHROMADB_HOST", "10.0.0.12")
EI_CHROMADB_PORT = int(os.environ.get("EI_CHROMADB_PORT", "18000"))


def _get_client():
    import chromadb
    return chromadb.HttpClient(host=EI_CHROMADB_HOST, port=EI_CHROMADB_PORT)


class EIMetrics(ApiHandler):

    @classmethod
    def requires_auth(cls) -> bool:
        return True

    @classmethod
    def get_methods(cls) -> list[str]:
        return ["GET", "POST"]

    async def process(self, input: Input, request: Request) -> Output:
        section = input.get("section", "all")
        user_id = input.get("user_id", "")

        try:
            client = _get_client()
        except Exception as e:
            return {"error": f"ChromaDB connection failed: {e}"}

        result = {}

        if section in ("all", "overview"):
            result["overview"] = self._get_overview(client, user_id)

        if section in ("all", "self_map"):
            result["self_map"] = self._get_self_map(client, user_id)

        if section in ("all", "emotion_log"):
            result["emotion_log"] = self._get_emotion_log(client, user_id)

        if section in ("all", "tom_profiles"):
            result["tom_profiles"] = self._get_tom_profiles(client, user_id)

        if section in ("all", "trends"):
            result["trends"] = self._get_trends(client, user_id)

        return result

    def _get_overview(self, client, user_id):
        """High-level counts and stats."""
        overview = {
            "self_map_items": 0,
            "emotion_events": 0,
            "tom_profiles": 0,
            "unique_users": set(),
            "collections_healthy": True,
        }

        for coll_name in ["ei_self_maps", "ei_emotion_log", "ei_tom_profiles"]:
            try:
                coll = client.get_collection(coll_name)
                count = coll.count()
                if coll_name == "ei_self_maps":
                    overview["self_map_items"] = count
                elif coll_name == "ei_emotion_log":
                    overview["emotion_events"] = count
                elif coll_name == "ei_tom_profiles":
                    overview["tom_profiles"] = count

                # Get unique users
                if count > 0:
                    docs = coll.get(include=["metadatas"], limit=min(count, 1000))
                    if docs["metadatas"]:
                        for m in docs["metadatas"]:
                            uid = m.get("user_id", "")
                            if uid:
                                overview["unique_users"].add(uid)
            except Exception:
                overview["collections_healthy"] = False

        overview["unique_users"] = list(overview["unique_users"])
        return overview

    def _get_self_map(self, client, user_id):
        """Self map breakdown by quadrant, power distribution, top items."""
        try:
            coll = client.get_collection("ei_self_maps")
            where = {"user_id": user_id} if user_id else None
            results = coll.get(
                where=where,
                include=["metadatas"],
                limit=500,
            )
        except Exception as e:
            return {"error": str(e)}

        if not results["metadatas"]:
            return {"items": [], "quadrants": {}, "power_distribution": [], "total": 0}

        items = results["metadatas"]
        quadrant_counts = Counter()
        quadrant_avg_power = {}
        power_distribution = Counter()
        valence_distribution = {"positive": 0, "negative": 0, "neutral": 0}

        for item in items:
            q = item.get("quadrant", "unknown")
            power = int(item.get("power_level", 0))
            valence = int(item.get("valence", 0))

            quadrant_counts[q] += 1
            quadrant_avg_power.setdefault(q, []).append(power)
            power_distribution[power] += 1

            if valence > 0:
                valence_distribution["positive"] += 1
            elif valence < 0:
                valence_distribution["negative"] += 1
            else:
                valence_distribution["neutral"] += 1

        # Compute averages
        quadrant_stats = {}
        for q, powers in quadrant_avg_power.items():
            quadrant_stats[q] = {
                "count": quadrant_counts[q],
                "avg_power": round(sum(powers) / len(powers), 1),
                "max_power": max(powers),
            }

        # Top items by power
        top_items = sorted(items, key=lambda x: int(x.get("power_level", 0)), reverse=True)[:10]
        top_items_clean = [
            {
                "label": i.get("label", ""),
                "quadrant": i.get("quadrant", ""),
                "power_level": int(i.get("power_level", 0)),
                "valence": int(i.get("valence", 0)),
                "source": i.get("source", ""),
            }
            for i in top_items
        ]

        return {
            "total": len(items),
            "quadrants": quadrant_stats,
            "power_distribution": dict(sorted(power_distribution.items())),
            "valence_distribution": valence_distribution,
            "top_items": top_items_clean,
        }

    def _get_emotion_log(self, client, user_id):
        """Emotion event breakdown: groups, severities, timeline."""
        try:
            coll = client.get_collection("ei_emotion_log")
            where = {"user_id": user_id} if user_id else None
            results = coll.get(
                where=where,
                include=["metadatas"],
                limit=1000,
            )
        except Exception as e:
            return {"error": str(e)}

        if not results["metadatas"]:
            return {"total": 0, "groups": {}, "severities": {}, "timeline": [], "recent": []}

        events = results["metadatas"]
        group_counts = Counter()
        severity_counts = Counter()
        source_counts = Counter()
        timeline = []

        for ev in events:
            group = ev.get("emotion_group", "unknown")
            severity = ev.get("severity", 0)
            source = ev.get("source", "unknown")
            timestamp = ev.get("timestamp", "")

            group_counts[group] += 1
            severity_counts[int(severity)] += 1
            source_counts[source] += 1

            if timestamp:
                timeline.append({
                    "timestamp": timestamp,
                    "group": group,
                    "severity": int(severity),
                    "label": ev.get("affected_item_label", ""),
                })

        # Sort timeline by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        # Recent events (last 20)
        recent = sorted(events, key=lambda x: x.get("timestamp", ""), reverse=True)[:20]
        recent_clean = [
            {
                "perception": ev.get("perception", "")[:100],
                "emotion_group": ev.get("emotion_group", ""),
                "severity": int(ev.get("severity", 0)),
                "severity_label": ev.get("severity_label", ""),
                "affected_item": ev.get("affected_item_label", ""),
                "timestamp": ev.get("timestamp", ""),
            }
            for ev in recent
        ]

        # Compute average severity
        all_severities = [int(ev.get("severity", 0)) for ev in events if ev.get("severity")]
        avg_severity = round(sum(all_severities) / len(all_severities), 2) if all_severities else 0

        return {
            "total": len(events),
            "groups": dict(group_counts.most_common()),
            "severities": dict(sorted(severity_counts.items())),
            "sources": dict(source_counts.most_common()),
            "avg_severity": avg_severity,
            "timeline": timeline[-100:],  # last 100 for charting
            "recent": recent_clean,
        }

    def _get_tom_profiles(self, client, user_id):
        """Theory of Mind profile stats."""
        try:
            coll = client.get_collection("ei_tom_profiles")
            where = {"user_id": user_id} if user_id else None
            results = coll.get(
                where=where,
                include=["metadatas", "documents"],
                limit=500,
            )
        except Exception as e:
            return {"error": str(e)}

        if not results["metadatas"]:
            return {"total": 0, "profiles": [], "analyses": 0}

        profiles = []
        analyses = 0
        for i, meta in enumerate(results["metadatas"]):
            record_type = meta.get("record_type", "")
            if record_type == "profile":
                attachments = []
                try:
                    doc = results["documents"][i] if results["documents"] else ""
                    attachments = json.loads(doc) if doc else []
                except (json.JSONDecodeError, TypeError):
                    pass

                profiles.append({
                    "person_name": meta.get("person_display_name", meta.get("person_name", "")),
                    "relationship": meta.get("relationship", ""),
                    "attachment_count": len(attachments),
                    "created_at": meta.get("created_at", ""),
                })
            elif record_type == "analysis":
                analyses += 1

        return {
            "total": len(profiles),
            "analyses": analyses,
            "profiles": profiles,
        }

    def _get_trends(self, client, user_id):
        """Compute trends from emotion log data."""
        try:
            coll = client.get_collection("ei_emotion_log")
            where = {"user_id": user_id} if user_id else None
            results = coll.get(
                where=where,
                include=["metadatas"],
                limit=1000,
            )
        except Exception as e:
            return {"error": str(e)}

        if not results["metadatas"]:
            return {"daily": {}, "group_over_time": {}}

        events = results["metadatas"]

        # Group by date
        daily = {}
        group_by_day = {}
        for ev in events:
            ts = ev.get("timestamp", "")
            if not ts:
                continue
            day = ts[:10]  # YYYY-MM-DD
            daily[day] = daily.get(day, 0) + 1

            group = ev.get("emotion_group", "unknown")
            group_by_day.setdefault(day, Counter())[group] += 1

        # Convert counters to dicts
        group_over_time = {day: dict(counts) for day, counts in sorted(group_by_day.items())}

        return {
            "daily_event_count": dict(sorted(daily.items())),
            "group_over_time": group_over_time,
        }
