# Implements MHH-EI specification: "Language Enabled Emotional Intelligence
# and Theory of Mind Algorithms" by Sean Webb (AGPL-3.0)
# Webb Equation of Emotion: EP ∆ P = ER

import json
import os

import chromadb

from python.helpers.tool import Tool, Response

EI_CHROMADB_HOST = os.environ.get("EI_CHROMADB_HOST", "10.0.0.12")
EI_CHROMADB_PORT = int(os.environ.get("EI_CHROMADB_PORT", "18000"))

# Emotion-to-visual mappings
EMOTION_COLOR_PALETTES = {
    "happiness": "warm golden tones, soft yellows and oranges, radiant light",
    "sadness": "muted blues, soft grays, gentle twilight hues",
    "anger": "deep reds, burnt sienna, intense contrast",
    "fear": "dark purples, shadowy blues, high contrast with sharp edges",
    "worry": "cool grays, muted lavender, overcast atmosphere",
    "pride": "rich golds, deep burgundy, regal warm tones",
    "shame": "desaturated earth tones, muted shadows",
    "disgust": "sickly greens, muddy browns, unsettling contrast",
    "surprise": "bright electric blues, vivid pops of color",
    "love": "warm rose, soft pink, gentle radiance",
    "curiosity": "vibrant teals, bright greens, sparkling highlights",
    "relief": "soft pastels, clearing sky blues, gentle warmth",
    "stress": "harsh fluorescent tones, jarring contrasts",
    "envy": "dark greens, shadowy emerald",
    "confusion": "swirling patterns, mixed desaturated hues",
    "boredom": "flat monochrome, minimal contrast",
}

EMOTION_EXPRESSIONS = {
    "happiness": "warm genuine smile, bright relaxed eyes, open posture",
    "sadness": "gentle downturned mouth, soft compassionate eyes, slight head tilt",
    "anger": "firm determined jaw, focused intense gaze, protective stance",
    "fear": "wide alert eyes, slightly tense shoulders, vigilant posture",
    "worry": "thoughtful furrowed brow, concerned attentive gaze",
    "pride": "confident slight smile, steady assured gaze, upright posture",
    "shame": "lowered gaze, humble posture, quiet presence",
    "curiosity": "bright inquisitive eyes, slightly tilted head, engaged lean forward",
    "love": "tender warm gaze, soft expression, open welcoming presence",
    "relief": "relaxed exhale expression, softened features, peaceful presence",
    "surprise": "raised eyebrows, open expression, alert engaged posture",
}

QUADRANT_MOTIFS = {
    "people": "warm interpersonal elements, connected figures, nurturing imagery, family symbols",
    "accomplishments": "structured architectural elements, achievement symbols, tools of craft, trophies of effort",
    "life_story": "flowing narrative elements, paths and journeys, time markers, photo album aesthetic",
    "ideas_likes": "abstract thought patterns, lightbulbs and sparks, books and symbols of knowledge",
}


class EIProfileImage(Tool):
    """
    Generate a text prompt for an Agent Zero profile image based on emotional data.

    Synthesizes the user's {self} map, recent emotions, and agent personality
    into a detailed image generation prompt suitable for DALL-E, Stable Diffusion, etc.
    """

    async def execute(self, user_id="", style="", **kwargs):
        if not user_id:
            user_id = self.agent.get_data("ei_user_id") or "default"

        # 1. Load {self} map
        self_map = self.agent.get_data(f"ei_self_map_{user_id}") or []

        # 2. Determine dominant quadrant
        quadrant_scores: dict[str, int] = {}
        highest_power_items = []
        for item in self_map:
            q = item.get("quadrant", "ideas_likes")
            power = int(item.get("power_level", 0))
            quadrant_scores[q] = quadrant_scores.get(q, 0) + power
            if power >= 7:
                highest_power_items.append(item)

        dominant_quadrant = max(quadrant_scores, key=quadrant_scores.get) if quadrant_scores else "ideas_likes"

        # 3. Get recent emotion logs
        dominant_emotion = "curiosity"  # default
        try:
            client = chromadb.HttpClient(host=EI_CHROMADB_HOST, port=EI_CHROMADB_PORT)
            coll = client.get_or_create_collection(name="ei_emotion_log")
            recent = coll.get(
                where={"user_id": user_id},
                include=["metadatas"],
                limit=20,
            )
            if recent["metadatas"]:
                # Count emotion groups
                emotion_counts: dict[str, int] = {}
                for meta in recent["metadatas"]:
                    eg = meta.get("emotion_group", "")
                    if eg:
                        emotion_counts[eg] = emotion_counts.get(eg, 0) + 1
                if emotion_counts:
                    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        except Exception:
            pass

        # 4. Build the prompt
        color_palette = EMOTION_COLOR_PALETTES.get(dominant_emotion, "balanced neutral tones with subtle warmth")
        expression = EMOTION_EXPRESSIONS.get(dominant_emotion, "calm attentive expression, present and engaged")
        motif = QUADRANT_MOTIFS.get(dominant_quadrant, "balanced universal symbols")

        # Highest-power attachment themes
        attachment_themes = []
        for item in sorted(highest_power_items, key=lambda x: int(x.get("power_level", 0)), reverse=True)[:3]:
            attachment_themes.append(item.get("label", ""))

        theme_str = ", ".join(attachment_themes) if attachment_themes else "universal human connection"

        style_directive = style or "digital art portrait, professional, modern"

        prompt = (
            f"A {style_directive} of an AI assistant avatar. "
            f"Color palette: {color_palette}. "
            f"Expression and posture: {expression}. "
            f"Visual motifs subtly incorporating: {motif}. "
            f"Thematic elements suggesting connection to: {theme_str}. "
            f"The overall mood is {dominant_emotion}, conveyed through lighting, "
            f"color temperature, and atmospheric elements. "
            f"The avatar appears approachable, intelligent, and emotionally aware. "
            f"Clean background with subtle gradient. High quality, detailed rendering."
        )

        result = {
            "prompt": prompt,
            "metadata": {
                "user_id": user_id,
                "dominant_emotion": dominant_emotion,
                "dominant_quadrant": dominant_quadrant,
                "highest_power_items": [i.get("label") for i in highest_power_items[:3]],
                "style": style_directive,
            },
        }

        return Response(
            message=(
                f"**Profile Image Prompt Generated**\n\n"
                f"Dominant emotion: {dominant_emotion}\n"
                f"Dominant quadrant: {dominant_quadrant}\n"
                f"Key themes: {theme_str}\n\n"
                f"**Prompt:**\n{prompt}\n\n"
                f"Use this prompt with any image generation API (DALL-E, Stable Diffusion, etc.) "
                f"to create the agent's profile avatar."
            ),
            break_loop=False,
        )
