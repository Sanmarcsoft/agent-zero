# Implements MHH-EI specification: "Language Enabled Emotional Intelligence
# and Theory of Mind Algorithms" by Sean Webb (AGPL-3.0)
# Webb Equation of Emotion: EP ∆ P = ER

import json
import os
import uuid
from datetime import datetime, timezone

import chromadb
from langchain_core.documents import Document

from python.helpers.tool import Tool, Response
from python.helpers.memory import Memory

EI_CHROMADB_HOST = os.environ.get("EI_CHROMADB_HOST", "10.0.0.12")
EI_CHROMADB_PORT = int(os.environ.get("EI_CHROMADB_PORT", "18000"))
EMOTION_LOG_COLLECTION = "ei_emotion_log"


# ── Emotion Group Definitions ──────────────────────────────────────────────
# Each group has 5 severity levels (1=lowest, 5=highest) per MHH-EI spec

EMOTION_GROUPS = {
    "fear": {
        "levels": ["concerned", "cautious", "afraid", "horror", "panic"],
        "description": "Devaluation threat, not accepted, internal source, present time",
    },
    "anger": {
        "levels": ["annoyed", "frustrated", "angry", "fury", "rage"],
        "description": "External attack on valuation, not accepted",
    },
    "sadness": {
        "levels": ["disappointed", "hurt", "sad", "grief", "despair"],
        "description": "Devaluation accepted, internalized loss",
    },
    "happiness": {
        "levels": ["satisfied", "pleased", "happy", "elated", "ecstatic"],
        "description": "Valuation meets or exceeds EP",
    },
    "disgust": {
        "levels": ["reticent", "distaste", "disgusted", "repulsed", "revulsion"],
        "description": "Violation of values/norms, something unwanted on {self}",
    },
    "worry": {
        "levels": ["distressed", "nervous", "worried", "distraught", "dread"],
        "description": "Fear projected into future time",
    },
    "regret": {
        "levels": ["mild_regret", "regret_2", "regret", "lament", "deplore"],
        "description": "Sadness projected into past time",
    },
    "pride": {
        "levels": ["pride_1", "pride_2", "pride_3", "pride_4", "pride_5"],
        "description": "Positive reflection on self-caused valuation increase",
    },
    "shame": {
        "levels": ["contrite", "sorry", "shame", "shame_4", "remorse"],
        "description": "Self-caused devaluation, internal perspective",
    },
    "embarrassment": {
        "levels": ["embarrassment_1", "embarrassment_2", "embarrassment_3", "embarrassment_4", "embarrassment_5"],
        "description": "Shame with external perspective awareness",
    },
    "flattery": {
        "levels": ["flattery_1", "flattery_2", "flattery_3", "flattery_4", "flattery_5"],
        "description": "External source introducing positive valuation",
    },
    "surprise": {
        "levels": ["surprise_1", "surprise_2", "surprise_3", "surprise_4", "surprise_5"],
        "description": "Expectation-defying event, out of the blue",
    },
    "stress": {
        "levels": ["stress_1", "stress_2", "stress_3", "stress_4", "stress_5"],
        "description": "Extended EP/P imbalance over time",
    },
    "relief": {
        "levels": ["relief_1", "relief_2", "relief_3", "relief_4", "relief_5"],
        "description": "Resolution of previous EP/P imbalance",
    },
    "envy": {
        "levels": ["envy_1", "envy_2", "envy_3", "envy_4", "envy_5"],
        "description": "Wanting an item from another's {self} map",
    },
    "love": {
        "levels": ["love_1", "love_2", "love_3", "love_4", "love_5"],
        "description": "Wanting experience/item for extended indefinite period",
    },
    "boredom": {
        "levels": ["boredom_1", "boredom_2", "boredom_3", "boredom_4", "boredom_5"],
        "description": "Prolonged {self}-unrelated perceptions",
    },
    "curiosity": {
        "levels": ["curiosity_1", "curiosity_2", "curiosity_3", "curiosity_4", "curiosity_5"],
        "description": "Investigating if new idea increases {self} valuation",
    },
    "confusion": {
        "levels": ["confusion_1", "confusion_2", "confusion_3", "confusion_4", "confusion_5"],
        "description": "Uncertain/undefined match with EP",
    },
    "positive_anticipation": {
        "levels": ["mild_anticipation", "anticipation_2", "anticipation", "edge_of_seat", "bated_breath"],
        "description": "Future valuation increase expected",
    },
    "negative_anticipation": {
        "levels": ["neg_anticipation_1", "neg_anticipation_2", "neg_anticipation_3", "neg_anticipation_4", "neg_anticipation_5"],
        "description": "Future devaluation expected",
    },
    "positive_rumination": {
        "levels": ["pos_rumination_1", "pos_rumination_2", "pos_rumination_3", "pos_rumination_4", "pos_rumination_5"],
        "description": "Reflecting on past valuation increase",
    },
    "negative_rumination": {
        "levels": ["neg_rumination_1", "neg_rumination_2", "neg_rumination_3", "neg_rumination_4", "neg_rumination_5"],
        "description": "Dwelling on past external attack, not resolved",
    },
}


def _get_emotion_log_collection():
    client = chromadb.HttpClient(host=EI_CHROMADB_HOST, port=EI_CHROMADB_PORT)
    return client.get_or_create_collection(name=EMOTION_LOG_COLLECTION)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_severity(ep_power: int, p_weight: int) -> int:
    """
    Compute emotion severity (1-5) from EP power level and Perception weight.

    Per MHH-EI spec:
    - EP power: how important the attachment is (1-10)
    - P weight: how serious/real the perception is (1-10)
    - Combined → severity 1-5
    """
    combined = (ep_power + p_weight) / 2.0
    if combined <= 2:
        return 1
    elif combined <= 4:
        return 2
    elif combined <= 6:
        return 3
    elif combined <= 8:
        return 4
    else:
        return 5


def select_emotion_group(
    valence_shift: str,
    accepted: bool,
    source: str,
    time_frame: str,
    perspective: str,
    is_norm_violation: bool = False,
    is_extended: bool = False,
    resolves_previous: bool = False,
    wants_from_other: bool = False,
    wants_indefinitely: bool = False,
    uncertain_match: bool = False,
    is_unrelated_prolonged: bool = False,
    is_investigating_new: bool = False,
    is_surprise: bool = False,
) -> str:
    """
    21-step decision algorithm from MHH-EI spec to determine emotion group.

    Args:
        valence_shift: "negative" or "positive"
        accepted: Whether the valence shift has been accepted/internalized
        source: "internal" | "external" | "value"
        time_frame: "past" | "now" | "future"
        perspective: "internal" | "external"
        is_norm_violation: Something unwanted on {self}
        is_extended: Extended imbalance over time
        resolves_previous: Relieves a prior imbalance
        wants_from_other: Wants item from another's {self}
        wants_indefinitely: Wants for extended period
        uncertain_match: Uncertain EP match
        is_unrelated_prolonged: Prolonged unrelated perceptions
        is_investigating_new: Investigating if new idea increases {self}
        is_surprise: Out-of-the-blue event

    Returns:
        Emotion group name string
    """
    # Step 5: Accepted negative shift
    if valence_shift == "negative" and accepted:
        if time_frame == "now" or time_frame == "present":
            return "sadness"
        elif time_frame == "past":
            return "regret"

    # Step 6: Unaccepted negative shift, internal source → threat
    if valence_shift == "negative" and not accepted and source == "internal":
        if time_frame in ("now", "present"):
            return "fear"
        elif time_frame == "future":
            return "worry"

    # Step 7: External attack, unaccepted
    if valence_shift == "negative" and not accepted and source == "external":
        if time_frame in ("now", "present"):
            return "anger"
        elif time_frame == "past":
            return "negative_rumination"

    # Step 8: Positive valuation
    if valence_shift == "positive":
        if time_frame in ("now", "present"):
            return "happiness"
        elif time_frame == "past":
            return "positive_rumination"
        elif time_frame == "future":
            return "positive_anticipation"

    # Step 9: Reflecting on positive self-caused valuation
    if valence_shift == "positive" and source == "value":
        return "pride"

    # Step 10: Norm violation / something unwanted
    if is_norm_violation:
        return "disgust"

    # Step 11: Self-caused devaluation with culpability
    if valence_shift == "negative" and source == "internal" and perspective == "internal":
        return "shame"

    # Step 12: Other-caused devaluation with external perspective
    if valence_shift == "negative" and perspective == "external":
        return "embarrassment"

    # Step 13: External positive valuation
    if valence_shift == "positive" and source == "external":
        return "flattery"

    # Step 14: Surprise
    if is_surprise:
        return "surprise"

    # Step 15: Extended imbalance
    if is_extended:
        return "stress"

    # Step 16: Resolution of prior imbalance
    if resolves_previous:
        return "relief"

    # Step 17: Wanting from others
    if wants_from_other:
        return "envy"

    # Step 18: Wanting indefinitely
    if wants_indefinitely:
        return "love"

    # Step 19: Uncertain match
    if uncertain_match:
        return "confusion"

    # Step 20: Prolonged unrelated perceptions
    if is_unrelated_prolonged:
        return "boredom"

    # Step 21: Investigating new idea
    if is_investigating_new:
        return "curiosity"

    # Default: need more information
    return "confusion"


class EmotionalAnalysis(Tool):

    async def execute(self, perception="", user_id="", **kwargs):
        """
        Analyze emotional content using the Webb Equation of Emotion.

        Args:
            perception: The perception event to analyze (what happened / what was said)
            user_id: User whose {self} map to reference
            affected_item_label: Which {self} map item is affected (optional)
            valence_shift: "positive" or "negative"
            accepted: Whether shift is accepted (true/false)
            source: "internal", "external", or "value"
            time_frame: "past", "now", "future"
            perspective: "internal" or "external"
            p_weight: Perception weight 1-10 (how serious/real)
            Additional kwargs for special conditions
        """
        if not perception:
            return Response(
                message="Error: 'perception' is required — describe the event or statement to analyze.",
                break_loop=False,
            )

        if not user_id:
            user_id = self.agent.get_data("ei_user_id") or "default"

        # Load {self} map from cache
        cache_key = f"ei_self_map_{user_id}"
        self_map = self.agent.get_data(cache_key) or []

        # Resolve affected item
        affected_label = kwargs.get("affected_item_label", "")
        ep_power = 5  # default if no item found
        item_valence = 0
        affected_item = None

        if affected_label and self_map:
            for item in self_map:
                if item.get("label", "").lower() == affected_label.lower():
                    affected_item = item
                    ep_power = int(item.get("power_level", 5))
                    item_valence = int(item.get("valence", 0))
                    break

        # Parse analysis parameters
        valence_shift = kwargs.get("valence_shift", "negative")
        accepted = str(kwargs.get("accepted", "false")).lower() in ("true", "yes", "1")
        source = kwargs.get("source", "external")
        time_frame = kwargs.get("time_frame", "now")
        perspective = kwargs.get("perspective", "internal")
        p_weight = int(kwargs.get("p_weight", 5))
        p_weight = max(1, min(10, p_weight))

        # Special condition flags
        is_norm_violation = str(kwargs.get("is_norm_violation", "false")).lower() in ("true", "yes", "1")
        is_extended = str(kwargs.get("is_extended", "false")).lower() in ("true", "yes", "1")
        resolves_previous = str(kwargs.get("resolves_previous", "false")).lower() in ("true", "yes", "1")
        wants_from_other = str(kwargs.get("wants_from_other", "false")).lower() in ("true", "yes", "1")
        wants_indefinitely = str(kwargs.get("wants_indefinitely", "false")).lower() in ("true", "yes", "1")
        uncertain_match = str(kwargs.get("uncertain_match", "false")).lower() in ("true", "yes", "1")
        is_surprise = str(kwargs.get("is_surprise", "false")).lower() in ("true", "yes", "1")

        # ── Webb Equation: EP ∆ P = ER ────────────────────────────────────
        # Step 1: Determine emotion group via decision algorithm
        emotion_group = select_emotion_group(
            valence_shift=valence_shift,
            accepted=accepted,
            source=source,
            time_frame=time_frame,
            perspective=perspective,
            is_norm_violation=is_norm_violation,
            is_extended=is_extended,
            resolves_previous=resolves_previous,
            wants_from_other=wants_from_other,
            wants_indefinitely=wants_indefinitely,
            uncertain_match=uncertain_match,
            is_surprise=is_surprise,
        )

        # Step 2: Compute severity
        severity = compute_severity(ep_power, p_weight)

        # Step 3: Get severity label
        group_data = EMOTION_GROUPS.get(emotion_group, EMOTION_GROUPS["confusion"])
        severity_label = group_data["levels"][severity - 1]

        # ── Build Webb Equation instance ───────────────────────────────────
        equation_id = str(uuid.uuid4())
        now = _now_iso()

        webb_equation = {
            "equation_id": equation_id,
            "user_id": user_id,
            "perception": perception,
            "affected_item_label": affected_label,
            "affected_item_id": affected_item.get("item_id", "") if affected_item else "",
            "ep_power": ep_power,
            "p_weight": p_weight,
            "valence_shift": valence_shift,
            "accepted": accepted,
            "source": source,
            "time_frame": time_frame,
            "perspective": perspective,
            "emotion_group": emotion_group,
            "severity": severity,
            "severity_label": severity_label,
            "timestamp": now,
        }

        # ── Persist to FAISS FRAGMENTS (session) ──────────────────────────
        try:
            db = await Memory.get(self.agent)
            faiss_text = (
                f"Webb Equation: {perception} → {emotion_group} ({severity_label}, severity {severity}/5). "
                f"Affected: {affected_label or 'general'}, EP power={ep_power}, P weight={p_weight}. "
                f"User: {user_id}."
            )
            await db.insert_text(
                faiss_text,
                {
                    "area": Memory.Area.FRAGMENTS.value,
                    "ei_type": "webb_equation",
                    "equation_id": equation_id,
                    "emotion_group": emotion_group,
                    "severity": str(severity),
                    "user_id": user_id,
                },
            )
        except Exception:
            pass  # FAISS write failure is non-fatal

        # ── Persist to ChromaDB ei_emotion_log (longitudinal) ─────────────
        try:
            collection = _get_emotion_log_collection()
            collection.add(
                ids=[equation_id],
                documents=[
                    f"{perception} → {emotion_group} ({severity_label})"
                ],
                metadatas=[{
                    "user_id": user_id,
                    "equation_id": equation_id,
                    "perception": perception[:500],  # truncate for metadata
                    "affected_item_label": affected_label,
                    "ep_power": ep_power,
                    "p_weight": p_weight,
                    "emotion_group": emotion_group,
                    "severity": severity,
                    "severity_label": severity_label,
                    "valence_shift": valence_shift,
                    "source": source,
                    "time_frame": time_frame,
                    "timestamp": now,
                }],
            )
        except Exception:
            pass  # ChromaDB write failure is non-fatal

        # ── Build response ─────────────────────────────────────────────────
        recommendation = self._generate_recommendation(emotion_group, severity, affected_label)

        result = (
            f"**Webb Equation of Emotion Analysis**\n"
            f"- Perception: {perception}\n"
            f"- Affected attachment: {affected_label or 'general/unidentified'}"
            f" (EP power={ep_power})\n"
            f"- Perception weight: {p_weight}/10\n"
            f"- Emotion group: **{emotion_group}**\n"
            f"- Severity: **{severity_label}** ({severity}/5)\n"
            f"- Recommendation: {recommendation}\n"
            f"- Equation ID: {equation_id}"
        )

        return Response(message=result, break_loop=False)

    def _generate_recommendation(self, emotion_group: str, severity: int, affected_label: str) -> str:
        """Generate contextual response recommendation based on emotion group and severity."""
        attachment_ref = f" regarding '{affected_label}'" if affected_label else ""

        if severity <= 2:
            intensity = "gentle acknowledgment"
        elif severity <= 3:
            intensity = "empathetic engagement"
        elif severity <= 4:
            intensity = "deep empathetic response with validation"
        else:
            intensity = "maximum compassion and careful, sustained support"

        recommendations = {
            "fear": f"Provide {intensity}{attachment_ref}. Acknowledge the threat without minimizing. Help assess realistic vs. catastrophic thinking.",
            "anger": f"Validate the anger{attachment_ref} with {intensity}. Acknowledge the perceived injustice. Avoid defensiveness.",
            "sadness": f"Offer {intensity}{attachment_ref}. Sit with the loss. Do not rush to problem-solving.",
            "happiness": f"Mirror and amplify the positive emotion{attachment_ref} with {intensity}. Celebrate with the user.",
            "disgust": f"Acknowledge the values violation{attachment_ref} with {intensity}. Validate boundaries.",
            "worry": f"Address future concerns{attachment_ref} with {intensity}. Help distinguish productive worry from rumination.",
            "regret": f"Provide {intensity} about past events{attachment_ref}. Help process without enabling self-blame loops.",
            "pride": f"Affirm the accomplishment{attachment_ref} with {intensity}. Reflect genuine appreciation.",
            "shame": f"Respond with {intensity}{attachment_ref}. Separate behavior from identity. Normalize human fallibility.",
            "embarrassment": f"Normalize the experience{attachment_ref} with {intensity}. Reduce perceived social threat.",
            "surprise": f"Help process the unexpected event{attachment_ref} with {intensity}. Allow time to integrate.",
            "stress": f"Acknowledge the sustained pressure{attachment_ref} with {intensity}. Explore practical relief options.",
            "relief": f"Celebrate the resolution{attachment_ref} with {intensity}. Acknowledge the weight that was lifted.",
            "envy": f"Explore the underlying desire{attachment_ref} with {intensity} without judgment.",
            "love": f"Honor the deep attachment{attachment_ref} with {intensity}.",
            "curiosity": f"Encourage exploration{attachment_ref} with {intensity}.",
            "confusion": f"Help clarify{attachment_ref} with {intensity}. Ask targeted questions.",
            "boredom": f"Acknowledge disengagement with {intensity}. Explore what would be meaningful.",
        }

        return recommendations.get(
            emotion_group,
            f"Respond with {intensity}{attachment_ref}. Gather more context about the emotional experience.",
        )
