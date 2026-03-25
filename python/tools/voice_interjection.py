# Voice interjection — allows Q to have Moneypenny (or other voices) speak
# in the conversation. Uses the TTS server's multi-voice support.

import base64
import json
import os
import urllib.request

from python.helpers.tool import Tool, Response

TTS_BASE = os.environ.get("TTS_URL", "http://10.0.0.96:8880/v1/audio/speech").rsplit("/v1", 1)[0]
TTS_ENDPOINT = f"{TTS_BASE}/v1/audio/speech"

VOICES = {
    "moneypenny": {
        "name": "Moneypenny",
        "description": "Miss Moneypenny — warm, professional, slightly teasing British secretary",
        "voice_id": "moneypenny",
    },
    "q": {
        "name": "Q",
        "description": "The Quartermaster — direct, technical, British authority",
        "voice_id": "q",
    },
}


def _synthesize(text: str, voice: str) -> str:
    """Call TTS server and return base64-encoded audio."""
    payload = json.dumps({
        "input": text,
        "voice": voice,
        "model": "tts-1",
        "response_format": "mp3",
    }).encode()
    req = urllib.request.Request(
        TTS_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return base64.b64encode(resp.read()).decode("utf-8")


class VoiceInterjection(Tool):
    """
    Let another character speak in the conversation.

    Q can call this to have Moneypenny deliver a message in her own voice,
    creating a natural multi-voice conversation. The audio is played in the
    browser alongside Q's responses.
    """

    async def execute(self, text="", voice="moneypenny", **kwargs):
        if not text:
            return Response(
                message="Error: 'text' required — what should the character say?",
                break_loop=False,
            )

        voice_id = voice.lower().strip()
        voice_info = VOICES.get(voice_id)
        if not voice_info:
            return Response(
                message=f"Unknown voice '{voice}'. Available: {', '.join(VOICES.keys())}",
                break_loop=False,
            )

        self.set_progress(f"{voice_info['name']} is speaking...")

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            audio_b64 = await loop.run_in_executor(None, _synthesize, text, voice_info["voice_id"])

            if not audio_b64:
                return Response(
                    message=f"{voice_info['name']} could not speak — TTS returned empty audio.",
                    break_loop=False,
                )

            # Return the audio as a special response that the frontend can play
            return Response(
                message=f"**{voice_info['name']}:** {text}",
                break_loop=False,
                additional={
                    "voice_interjection": True,
                    "speaker": voice_info["name"],
                    "voice_id": voice_id,
                    "audio": audio_b64,
                },
            )

        except Exception as e:
            return Response(
                message=f"{voice_info['name']} failed to speak: {e}",
                break_loop=False,
            )
