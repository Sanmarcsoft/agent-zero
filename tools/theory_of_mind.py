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
TOM_COLLECTION = "ei_tom_profiles"


def _get_tom_collection():
    client = chromadb.HttpClient(host=EI_CHROMADB_HOST, port=EI_CHROMADB_PORT)
    return client.get_or_create_collection(name=TOM_COLLECTION)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TheoryOfMind(Tool):
    """
    Theory of Mind tool using MHH-EI Perception Tracking.

    Core principle: If a person has a Perception about something, they know about it.
    If they lack the Perception, they lack knowledge.

    Models third-party emotional states by:
    1. Building/retrieving their embedded {self} map (what they care about)
    2. Tracking what Perceptions they have access to
    3. Running the Webb Equation from their perspective (their EP, their P → their ER)
    """

    async def execute(self, action="analyze", **kwargs):
        if not action:
            action = "analyze"

        action = action.strip().lower()
        try:
            if action == "analyze":
                return await self._analyze(**kwargs)
            elif action == "create_profile":
                return await self._create_profile(**kwargs)
            elif action == "update_profile":
                return await self._update_profile(**kwargs)
            elif action == "get_profile":
                return await self._get_profile(**kwargs)
            elif action == "perception_check":
                return await self._perception_check(**kwargs)
            else:
                return Response(
                    message=f"Unknown action '{action}'. Valid: analyze, create_profile, update_profile, get_profile, perception_check",
                    break_loop=False,
                )
        except Exception as e:
            return Response(message=f"theory_of_mind error: {e}", break_loop=False)

    async def _analyze(self, **kwargs) -> Response:
        """
        Analyze a third party's likely emotional state.

        Uses their embedded {self} map to run the Webb Equation from their perspective.
        """
        person_name = kwargs.get("person_name", "")
        if not person_name:
            return Response(
                message="Error: 'person_name' is required — who are we modeling?",
                break_loop=False,
            )

        situation = kwargs.get("situation", "")
        if not situation:
            return Response(
                message="Error: 'situation' is required — what is the event or scenario?",
                break_loop=False,
            )

        user_id = kwargs.get("user_id", "") or self.agent.get_data("ei_user_id") or "default"

        # Try to load existing profile
        collection = _get_tom_collection()
        results = collection.get(
            where={"$and": [
                {"user_id": user_id},
                {"person_name": person_name.lower()},
            ]},
            include=["metadatas", "documents"],
        )

        profile = None
        attachments = []
        if results["ids"]:
            profile = results["metadatas"][0]
            # Parse attachments from document
            try:
                attachments = json.loads(results["documents"][0])
            except (json.JSONDecodeError, TypeError):
                attachments = []

        # Resolve which attachment is affected
        affected_attachment = kwargs.get("affected_attachment", "")
        ep_power = int(kwargs.get("ep_power", 5))

        if affected_attachment and attachments:
            for att in attachments:
                if att.get("label", "").lower() == affected_attachment.lower():
                    ep_power = int(att.get("power_level", ep_power))
                    break

        # Perception tracking: does this person know about the situation?
        has_perception = str(kwargs.get("has_perception", "true")).lower() in ("true", "yes", "1")

        if not has_perception:
            return Response(
                message=(
                    f"**Theory of Mind: {person_name}**\n"
                    f"- Situation: {situation}\n"
                    f"- Perception: **{person_name} does NOT have this Perception**\n"
                    f"- Conclusion: {person_name} does not know about this event. "
                    f"They cannot have an emotional reaction to something they are unaware of. "
                    f"If/when they receive this Perception (told, witness, discover), "
                    f"then the Webb Equation will activate.\n"
                    f"- Key: Perception Tracking is the linchpin of Theory of Mind."
                ),
                break_loop=False,
            )

        # They have the Perception → run Webb Equation from their perspective
        valence_shift = kwargs.get("valence_shift", "negative")
        p_weight = int(kwargs.get("p_weight", 5))
        source = kwargs.get("source", "external")
        time_frame = kwargs.get("time_frame", "now")

        # Import the emotion computation functions
        from python.tools.emotional_analysis import select_emotion_group, compute_severity, EMOTION_GROUPS

        emotion_group = select_emotion_group(
            valence_shift=valence_shift,
            accepted=str(kwargs.get("accepted", "false")).lower() in ("true", "yes", "1"),
            source=source,
            time_frame=time_frame,
            perspective=kwargs.get("perspective", "internal"),
        )

        severity = compute_severity(ep_power, p_weight)
        group_data = EMOTION_GROUPS.get(emotion_group, EMOTION_GROUPS["confusion"])
        severity_label = group_data["levels"][severity - 1]

        # Confidence based on profile completeness
        if profile and attachments:
            confidence = "high" if len(attachments) >= 5 else "moderate"
        else:
            confidence = "low (no profile — using general assumptions)"

        # Build the response
        profile_note = ""
        if not profile:
            profile_note = (
                "\n\n*Note: No embedded {self} map exists for this person. "
                "Analysis uses default assumptions. Create a profile via "
                "`theory_of_mind(action='create_profile')` for more accurate modeling.*"
            )

        result = (
            f"**Theory of Mind: {person_name}**\n"
            f"- Situation: {situation}\n"
            f"- Perception: {person_name} **HAS** this Perception (they know about it)\n"
            f"- Affected attachment: {affected_attachment or 'general'} (estimated EP power={ep_power})\n"
            f"- Perception weight: {p_weight}/10\n"
            f"- **From {person_name}'s perspective:**\n"
            f"  - Emotion group: **{emotion_group}**\n"
            f"  - Severity: **{severity_label}** ({severity}/5)\n"
            f"- Confidence: {confidence}"
            f"{profile_note}"
        )

        # Store analysis in ChromaDB
        try:
            analysis_id = str(uuid.uuid4())
            collection.add(
                ids=[analysis_id],
                documents=[json.dumps({
                    "type": "analysis",
                    "person_name": person_name,
                    "situation": situation,
                    "emotion_group": emotion_group,
                    "severity": severity,
                })],
                metadatas=[{
                    "user_id": user_id,
                    "person_name": person_name.lower(),
                    "record_type": "analysis",
                    "emotion_group": emotion_group,
                    "severity": severity,
                    "timestamp": _now_iso(),
                }],
            )
        except Exception:
            pass

        return Response(message=result, break_loop=False)

    async def _create_profile(self, **kwargs) -> Response:
        """Create an embedded {self} map profile for a third party."""
        person_name = kwargs.get("person_name", "")
        if not person_name:
            return Response(message="Error: 'person_name' is required.", break_loop=False)

        user_id = kwargs.get("user_id", "") or self.agent.get_data("ei_user_id") or "default"
        relationship = kwargs.get("relationship", "")
        attachments_json = kwargs.get("attachments", "[]")

        try:
            attachments = json.loads(attachments_json) if isinstance(attachments_json, str) else attachments_json
        except json.JSONDecodeError:
            attachments = []

        collection = _get_tom_collection()

        # Check for existing
        existing = collection.get(
            where={"$and": [
                {"user_id": user_id},
                {"person_name": person_name.lower()},
                {"record_type": "profile"},
            ]},
        )
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        profile_id = str(uuid.uuid4())
        now = _now_iso()

        collection.add(
            ids=[profile_id],
            documents=[json.dumps(attachments)],
            metadatas=[{
                "user_id": user_id,
                "person_name": person_name.lower(),
                "person_display_name": person_name,
                "relationship": relationship,
                "record_type": "profile",
                "attachment_count": len(attachments),
                "created_at": now,
                "updated_at": now,
            }],
        )

        return Response(
            message=(
                f"Created Theory of Mind profile for '{person_name}' "
                f"(relationship: {relationship or 'unspecified'}, "
                f"{len(attachments)} known attachments). "
                f"Profile ID: {profile_id}"
            ),
            break_loop=False,
        )

    async def _update_profile(self, **kwargs) -> Response:
        """Update an existing ToM profile's attachments."""
        person_name = kwargs.get("person_name", "")
        if not person_name:
            return Response(message="Error: 'person_name' is required.", break_loop=False)

        user_id = kwargs.get("user_id", "") or self.agent.get_data("ei_user_id") or "default"

        collection = _get_tom_collection()
        existing = collection.get(
            where={"$and": [
                {"user_id": user_id},
                {"person_name": person_name.lower()},
                {"record_type": "profile"},
            ]},
            include=["metadatas", "documents"],
        )

        if not existing["ids"]:
            return Response(
                message=f"No profile found for '{person_name}'. Use action='create_profile' first.",
                break_loop=False,
            )

        # Merge new attachments
        try:
            current_attachments = json.loads(existing["documents"][0])
        except (json.JSONDecodeError, TypeError):
            current_attachments = []

        new_attachments_json = kwargs.get("attachments", "[]")
        try:
            new_attachments = json.loads(new_attachments_json) if isinstance(new_attachments_json, str) else new_attachments_json
        except json.JSONDecodeError:
            new_attachments = []

        # Merge by label
        by_label = {a.get("label", "").lower(): a for a in current_attachments}
        for att in new_attachments:
            by_label[att.get("label", "").lower()] = att
        merged = list(by_label.values())

        meta = existing["metadatas"][0]
        meta["attachment_count"] = len(merged)
        meta["updated_at"] = _now_iso()
        if kwargs.get("relationship"):
            meta["relationship"] = kwargs["relationship"]

        collection.update(
            ids=[existing["ids"][0]],
            documents=[json.dumps(merged)],
            metadatas=[meta],
        )

        return Response(
            message=f"Updated profile for '{person_name}': now {len(merged)} attachments.",
            break_loop=False,
        )

    async def _get_profile(self, **kwargs) -> Response:
        """Retrieve a ToM profile."""
        person_name = kwargs.get("person_name", "")
        if not person_name:
            return Response(message="Error: 'person_name' is required.", break_loop=False)

        user_id = kwargs.get("user_id", "") or self.agent.get_data("ei_user_id") or "default"

        collection = _get_tom_collection()
        results = collection.get(
            where={"$and": [
                {"user_id": user_id},
                {"person_name": person_name.lower()},
                {"record_type": "profile"},
            ]},
            include=["metadatas", "documents"],
        )

        if not results["ids"]:
            return Response(
                message=f"No Theory of Mind profile found for '{person_name}'.",
                break_loop=False,
            )

        meta = results["metadatas"][0]
        try:
            attachments = json.loads(results["documents"][0])
        except (json.JSONDecodeError, TypeError):
            attachments = []

        lines = [
            f"**Theory of Mind Profile: {meta.get('person_display_name', person_name)}**",
            f"- Relationship: {meta.get('relationship', 'unspecified')}",
            f"- Known attachments: {len(attachments)}",
        ]
        for att in sorted(attachments, key=lambda x: int(x.get("power_level", 0)), reverse=True):
            lines.append(
                f"  - {att.get('label')} ({att.get('quadrant', 'unknown')}, "
                f"power={att.get('power_level', '?')}, valence={att.get('valence', '?')})"
            )

        return Response(message="\n".join(lines), break_loop=False)

    async def _perception_check(self, **kwargs) -> Response:
        """
        Check whether a person has a specific Perception (knowledge of an event).

        This is the linchpin of Theory of Mind per MHH-EI:
        If they have the Perception → they know → emotions activate.
        If they lack the Perception → they don't know → no emotional reaction.
        """
        person_name = kwargs.get("person_name", "")
        event = kwargs.get("event", "")

        if not person_name or not event:
            return Response(
                message="Error: 'person_name' and 'event' are required.",
                break_loop=False,
            )

        # Gather evidence
        was_present = str(kwargs.get("was_present", "unknown")).lower()
        was_told = str(kwargs.get("was_told", "unknown")).lower()
        could_observe = str(kwargs.get("could_observe", "unknown")).lower()

        has_perception = False
        evidence = []

        if was_present in ("true", "yes"):
            has_perception = True
            evidence.append(f"{person_name} was physically present")
        if was_told in ("true", "yes"):
            has_perception = True
            evidence.append(f"{person_name} was told about it")
        if could_observe in ("true", "yes"):
            has_perception = True
            evidence.append(f"{person_name} could observe/sense it")

        if not evidence:
            evidence.append("Insufficient evidence to determine — ask clarifying questions")

        evidence_str = "; ".join(evidence)

        if has_perception:
            result = (
                f"**Perception Check: {person_name}**\n"
                f"- Event: {event}\n"
                f"- Has Perception: **YES**\n"
                f"- Evidence: {evidence_str}\n"
                f"- Conclusion: {person_name} knows about this event. "
                f"The Webb Equation is active — they will have an emotional reaction "
                f"based on how this event affects their {'{self}'} map attachments."
            )
        else:
            result = (
                f"**Perception Check: {person_name}**\n"
                f"- Event: {event}\n"
                f"- Has Perception: **NO** (or insufficient evidence)\n"
                f"- Evidence: {evidence_str}\n"
                f"- Conclusion: {person_name} does not know about this event. "
                f"No emotional reaction possible without Perception. "
                f"They would react only if/when they learn about it."
            )

        return Response(message=result, break_loop=False)
