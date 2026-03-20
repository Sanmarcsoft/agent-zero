# avatar_bridge.py — orchestrates the avatar animation pipeline
#
# Pipeline: text → Percy TTS (Qwen3-TTS) → viseme extraction → WebSocket broadcast
# Scaleway LivePortrait: face preparation from reference photo
#
# Environment variables:
#   PERCY_TTS_URL      — Qwen3-TTS endpoint (default http://ai.matthewstevens.org:8880/v1)
#   PERCY_TTS_KEY      — API key for Percy TTS (default prime-mouth)
#   LIVEPORTRAIT_URL   — Scaleway LivePortrait endpoint (default http://localhost:8090)

from __future__ import annotations

import asyncio
import base64
import io
import math
import os
import struct
import time
import wave
from typing import Any

import httpx

from python.helpers.print_style import PrintStyle

# ── configuration ────────────────────────────────────────────────────────────

PERCY_TTS_URL = os.environ.get("PERCY_TTS_URL", "http://10.0.0.96:8880/v1")
PERCY_TTS_KEY = os.environ.get("PERCY_TTS_KEY", "prime-mouth")

# Percy TTS returns MP3 (audio/mpeg). We need PCM for viseme analysis.
# Use subprocess to convert MP3→WAV via ffmpeg if available, else use
# the MP3 directly for playback and fall back to amplitude-from-mp3.
_HAS_FFMPEG: bool | None = None
LIVEPORTRAIT_URL = os.environ.get("LIVEPORTRAIT_URL", "http://localhost:8090")

# ── viseme mapping tables ────────────────────────────────────────────────────
# Maps amplitude bands and zero-crossing rates to approximate mouth shapes.
# These are the same viseme names used in avatar-test.html.

VISEME_SET = ["rest", "aa", "ee", "ih", "oh", "oo", "mm", "ff", "th", "ss"]

# Amplitude thresholds (normalized 0-1) for viseme selection.
# Higher amplitude → wider mouth opening.  Shape is refined by spectral
# characteristics approximated via zero-crossing rate.
_AMP_SILENT = 0.02
_AMP_LOW = 0.10
_AMP_MID = 0.30
_AMP_HIGH = 0.55

# ── emotion definitions (matches avatar-test.html EMO table) ─────────────────

EMOTION_TYPES = [
    "neutral", "happiness", "sadness", "anger", "fear", "surprise",
    "disgust", "worry", "pride", "shame", "curiosity", "love",
]


class AvatarBridge:
    """Orchestrates the full avatar animation pipeline.

    Coordinates Percy TTS synthesis, viseme extraction from audio,
    LivePortrait face preparation, and packages everything for
    WebSocket delivery to the avatar renderer.
    """

    def __init__(
        self,
        percy_url: str | None = None,
        percy_key: str | None = None,
        liveportrait_url: str | None = None,
    ) -> None:
        self.percy_url = percy_url or PERCY_TTS_URL
        self.percy_key = percy_key or PERCY_TTS_KEY
        self.liveportrait_url = liveportrait_url or LIVEPORTRAIT_URL

    # ── TTS synthesis ────────────────────────────────────────────────────

    async def synthesize_percy(self, text: str) -> tuple[bytes, bytes]:
        """Call Qwen3-TTS with the Percy voice.

        Makes two requests (in parallel when possible):
          1. MP3 for browser playback (smaller, fast)
          2. WAV for viseme analysis (PCM, 24kHz mono s16)

        Returns (mp3_bytes, wav_bytes). If WAV request fails, wav_bytes
        will be empty and viseme extraction falls back to text timing.
        """
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text")

        url = f"{self.percy_url.rstrip('/')}/audio/speech"
        headers = {"Content-Type": "application/json"}
        clean_text = text.strip()

        PrintStyle.standard(f"[AvatarBridge] Synthesizing TTS for {len(clean_text)} chars...")
        t0 = time.monotonic()

        # Request both formats concurrently
        async with httpx.AsyncClient(timeout=60.0) as client:
            mp3_task = client.post(url, json={"input": clean_text, "response_format": "mp3"}, headers=headers)
            wav_task = client.post(url, json={"input": clean_text, "response_format": "wav"}, headers=headers)

            mp3_resp, wav_resp = await asyncio.gather(mp3_task, wav_task, return_exceptions=True)

        elapsed = time.monotonic() - t0

        # Handle MP3 response
        if isinstance(mp3_resp, Exception):
            raise RuntimeError(f"Percy TTS MP3 request failed: {mp3_resp}")
        if mp3_resp.status_code != 200:
            raise RuntimeError(f"Percy TTS failed ({mp3_resp.status_code}): {mp3_resp.text[:500]}")
        mp3_bytes = mp3_resp.content

        # Handle WAV response (non-fatal if it fails)
        wav_bytes = b""
        if isinstance(wav_resp, Exception):
            PrintStyle.error(f"[AvatarBridge] WAV request failed: {wav_resp}")
        elif wav_resp.status_code != 200:
            PrintStyle.error(f"[AvatarBridge] WAV request returned {wav_resp.status_code}")
        else:
            wav_bytes = wav_resp.content
            # Validate WAV header
            if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF":
                PrintStyle.error("[AvatarBridge] WAV response is not valid WAV")
                wav_bytes = b""

        PrintStyle.standard(
            f"[AvatarBridge] TTS complete: {len(mp3_bytes)} MP3 + {len(wav_bytes)} WAV "
            f"in {elapsed:.2f}s"
        )

        if len(mp3_bytes) < 100:
            raise RuntimeError("Percy TTS returned too little audio data")

        return mp3_bytes, wav_bytes

    # ── viseme extraction ────────────────────────────────────────────────

    def extract_visemes(self, audio_wav: bytes) -> list[dict[str, Any]]:
        """Extract viseme timing from WAV audio using amplitude envelope analysis.

        Analyzes the audio in fixed-duration windows, computing amplitude (RMS)
        and a simple zero-crossing rate to approximate spectral characteristics.
        Maps each window to a viseme shape with timing information.

        Returns a list of dicts: {viseme: str, start_ms: int, duration_ms: int}
        """
        if not audio_wav or len(audio_wav) < 44:
            return [{"viseme": "rest", "start_ms": 0, "duration_ms": 100}]

        # Parse WAV header
        buf = io.BytesIO(audio_wav)
        try:
            with wave.open(buf, "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)
        except Exception as e:
            PrintStyle.error(f"[AvatarBridge] Failed to parse WAV: {e}")
            return [{"viseme": "rest", "start_ms": 0, "duration_ms": 100}]

        if n_frames == 0:
            return [{"viseme": "rest", "start_ms": 0, "duration_ms": 100}]

        # Convert raw PCM to float samples (mono, -1.0 to 1.0)
        samples = _pcm_to_float_mono(raw_data, sample_width, n_channels)
        if not samples:
            return [{"viseme": "rest", "start_ms": 0, "duration_ms": 100}]

        # Analysis parameters
        window_ms = 50  # 50ms windows for viseme timing
        window_samples = max(1, int(sample_rate * window_ms / 1000))
        hop_samples = window_samples  # non-overlapping windows

        visemes: list[dict[str, Any]] = []
        total_samples = len(samples)
        pos = 0
        time_ms = 0

        while pos < total_samples:
            end = min(pos + window_samples, total_samples)
            window = samples[pos:end]
            actual_duration_ms = int(len(window) * 1000 / sample_rate)

            if len(window) < 4:
                visemes.append({
                    "viseme": "rest",
                    "start_ms": time_ms,
                    "duration_ms": actual_duration_ms,
                })
                break

            # Compute RMS amplitude (normalized)
            rms = math.sqrt(sum(s * s for s in window) / len(window))

            # Compute zero-crossing rate (rough spectral proxy)
            crossings = sum(
                1 for i in range(1, len(window))
                if (window[i] >= 0) != (window[i - 1] >= 0)
            )
            zcr = crossings / len(window)  # 0.0 to ~0.5

            # Map to viseme
            viseme = _amplitude_to_viseme(rms, zcr)

            visemes.append({
                "viseme": viseme,
                "start_ms": time_ms,
                "duration_ms": actual_duration_ms,
            })

            pos += hop_samples
            time_ms += actual_duration_ms

        # Merge consecutive identical visemes for efficiency
        visemes = _merge_consecutive_visemes(visemes)

        # Ensure we end on rest
        if visemes and visemes[-1]["viseme"] != "rest":
            last = visemes[-1]
            visemes.append({
                "viseme": "rest",
                "start_ms": last["start_ms"] + last["duration_ms"],
                "duration_ms": 100,
            })

        PrintStyle.standard(
            f"[AvatarBridge] Extracted {len(visemes)} viseme segments "
            f"from {time_ms}ms audio"
        )
        return visemes

    # ── face preparation (Scaleway LivePortrait) ─────────────────────────

    async def prepare_face(self, image_bytes: bytes) -> dict[str, Any]:
        """Send a reference photo to Scaleway LivePortrait /prepare endpoint.

        Returns the face model JSON from the server, which contains landmarks,
        face encoding, and other data needed for real-time animation.
        """
        if not image_bytes:
            raise ValueError("Cannot prepare face from empty image data")

        url = f"{self.liveportrait_url.rstrip('/')}/prepare"

        PrintStyle.standard("[AvatarBridge] Preparing face model via LivePortrait...")
        t0 = time.monotonic()

        async with httpx.AsyncClient(timeout=120.0) as client:
            # Detect image type from magic bytes
            content_type = "image/jpeg"
            if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                content_type = "image/png"
            elif image_bytes[:2] == b"\xff\xd8":
                content_type = "image/jpeg"
            elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
                content_type = "image/webp"

            response = await client.post(
                url,
                files={"image": ("avatar.png", image_bytes, content_type)},
            )

            if response.status_code != 200:
                error_detail = response.text[:500]
                raise RuntimeError(
                    f"LivePortrait /prepare failed ({response.status_code}): {error_detail}"
                )

            face_model = response.json()

        elapsed = time.monotonic() - t0
        PrintStyle.standard(
            f"[AvatarBridge] Face model prepared in {elapsed:.2f}s "
            f"(keys: {list(face_model.keys()) if isinstance(face_model, dict) else 'N/A'})"
        )
        return face_model

    # ── full pipeline ────────────────────────────────────────────────────

    async def process_response(
        self,
        text: str,
        emotion_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Full avatar animation pipeline: TTS -> visemes -> package for WebSocket.

        Args:
            text: The text to speak/animate.
            emotion_state: Dict with 'emotion_type' (str) and 'emotion_severity' (int 1-5).

        Returns:
            A dict ready for WebSocket broadcast with keys:
                - audio_b64: base64-encoded WAV audio
                - visemes: list of viseme timing dicts
                - emotion: validated emotion state
                - text: original text
                - duration_ms: total audio duration in milliseconds
        """
        emotion_type = emotion_state.get("emotion_type", "neutral")
        if emotion_type not in EMOTION_TYPES:
            emotion_type = "neutral"

        emotion_severity = emotion_state.get("emotion_severity", 3)
        emotion_severity = max(1, min(5, int(emotion_severity)))

        # Step 1: Synthesize speech with Percy voice (returns MP3 + WAV)
        mp3_bytes, wav_bytes = await self.synthesize_percy(text)

        # Step 2: Extract viseme timing from audio
        if wav_bytes:
            visemes = self.extract_visemes(wav_bytes)
        else:
            # Fallback: estimate visemes from text timing
            visemes = _estimate_visemes_from_text(text)

        # Calculate total duration from visemes
        duration_ms = 0
        if visemes:
            last = visemes[-1]
            duration_ms = last["start_ms"] + last["duration_ms"]

        # Step 3: Encode MP3 audio for WebSocket transport
        # Browser can play MP3 natively — more efficient than WAV
        audio_b64 = base64.b64encode(mp3_bytes).decode("utf-8")

        # Package for WebSocket
        result = {
            "type": "animate",
            "audio_b64": audio_b64,
            "audio_format": "mp3",
            "visemes": visemes,
            "emotion": {
                "type": emotion_type,
                "severity": emotion_severity,
            },
            "text": text,
            "duration_ms": duration_ms,
            "timestamp": int(time.time() * 1000),
        }

        PrintStyle.standard(
            f"[AvatarBridge] Pipeline complete: {len(text)} chars → "
            f"{len(audio_wav)} bytes audio → {len(visemes)} visemes → "
            f"{duration_ms}ms duration, emotion={emotion_type}({emotion_severity})"
        )

        return result


# ── internal helpers ─────────────────────────────────────────────────────────

def _pcm_to_float_mono(
    raw_data: bytes,
    sample_width: int,
    n_channels: int,
) -> list[float]:
    """Convert raw PCM bytes to a list of mono float samples in [-1.0, 1.0]."""

    if sample_width == 1:
        # 8-bit unsigned PCM
        fmt = "B"
        max_val = 128.0
        offset = 128
    elif sample_width == 2:
        # 16-bit signed PCM (most common)
        fmt = "<h"
        max_val = 32768.0
        offset = 0
    elif sample_width == 3:
        # 24-bit signed PCM — handle manually
        fmt = None
        max_val = 8388608.0
        offset = 0
    elif sample_width == 4:
        # 32-bit signed PCM
        fmt = "<i"
        max_val = 2147483648.0
        offset = 0
    else:
        PrintStyle.error(f"[AvatarBridge] Unsupported sample width: {sample_width}")
        return []

    samples: list[float] = []
    frame_size = sample_width * n_channels

    if fmt is None and sample_width == 3:
        # 24-bit PCM: read 3 bytes at a time
        for i in range(0, len(raw_data), frame_size):
            # Take first channel only for mono conversion
            b = raw_data[i : i + 3]
            if len(b) < 3:
                break
            # Sign-extend 24-bit to 32-bit
            value = struct.unpack("<i", b + (b"\xff" if b[2] & 0x80 else b"\x00"))[0]
            samples.append(value / max_val)
    else:
        struct_size = struct.calcsize(fmt)
        for i in range(0, len(raw_data), frame_size):
            # Take first channel only for mono conversion
            b = raw_data[i : i + struct_size]
            if len(b) < struct_size:
                break
            value = struct.unpack(fmt, b)[0]
            if offset:
                value -= offset
            samples.append(value / max_val)

    return samples


def _amplitude_to_viseme(rms: float, zcr: float) -> str:
    """Map RMS amplitude and zero-crossing rate to a viseme.

    The heuristic uses amplitude to determine mouth openness and ZCR
    as a rough proxy for spectral content (fricatives have high ZCR,
    vowels have low ZCR, nasals/stops are in between).
    """
    # Silent or near-silent
    if rms < _AMP_SILENT:
        return "rest"

    # Very low amplitude — closed mouth shapes
    if rms < _AMP_LOW:
        if zcr > 0.25:
            return "ss"  # sibilant-like
        if zcr > 0.15:
            return "ff"  # fricative-like
        return "mm"  # nasal/closed

    # Low-mid amplitude
    if rms < _AMP_MID:
        if zcr > 0.25:
            return "th"  # dental fricative
        if zcr > 0.15:
            return "ee"  # front vowel (tighter)
        if zcr > 0.08:
            return "ih"  # mid vowel
        return "oh"  # back vowel

    # Mid-high amplitude
    if rms < _AMP_HIGH:
        if zcr > 0.20:
            return "ee"  # open front vowel
        if zcr > 0.10:
            return "oh"  # open back vowel
        return "aa"  # open vowel

    # High amplitude — wide open mouth
    if zcr > 0.15:
        return "oh"  # rounded open
    return "aa"  # maximally open


def _merge_consecutive_visemes(
    visemes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge adjacent viseme segments that have the same shape.

    Reduces the number of segments sent over WebSocket without
    losing timing accuracy.
    """
    if not visemes:
        return visemes

    merged: list[dict[str, Any]] = [visemes[0].copy()]

    for segment in visemes[1:]:
        prev = merged[-1]
        if segment["viseme"] == prev["viseme"]:
            # Extend the previous segment
            prev["duration_ms"] += segment["duration_ms"]
        else:
            merged.append(segment.copy())

    return merged


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    global _HAS_FFMPEG
    if _HAS_FFMPEG is None:
        import shutil
        _HAS_FFMPEG = shutil.which("ffmpeg") is not None
    return _HAS_FFMPEG


def _mp3_to_wav(mp3_bytes: bytes) -> bytes:
    """Convert MP3 bytes to WAV PCM bytes using ffmpeg.

    Returns empty bytes if ffmpeg is not available.
    """
    if not _check_ffmpeg():
        PrintStyle.standard("[AvatarBridge] ffmpeg not found — skipping MP3→WAV conversion")
        return b""

    import subprocess
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_f:
            mp3_f.write(mp3_bytes)
            mp3_path = mp3_f.name

        wav_path = mp3_path.replace(".mp3", ".wav")

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", mp3_path,
                "-ar", "24000",      # 24kHz sample rate
                "-ac", "1",          # mono
                "-sample_fmt", "s16",  # 16-bit PCM
                wav_path,
            ],
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            PrintStyle.error(f"[AvatarBridge] ffmpeg failed: {result.stderr[:200]}")
            return b""

        with open(wav_path, "rb") as f:
            wav_bytes = f.read()

        # Clean up
        os.unlink(mp3_path)
        os.unlink(wav_path)

        PrintStyle.standard(
            f"[AvatarBridge] MP3→WAV: {len(mp3_bytes)} → {len(wav_bytes)} bytes"
        )
        return wav_bytes

    except Exception as e:
        PrintStyle.error(f"[AvatarBridge] MP3→WAV conversion error: {e}")
        return b""


def _estimate_visemes_from_text(text: str) -> list[dict[str, Any]]:
    """Estimate viseme timing from text when audio analysis is unavailable.

    Uses a simple character-to-viseme mapping with ~80ms per phoneme,
    matching the timing in avatar-test.html's testLip() function.
    """
    char_to_viseme = {
        "a": "aa", "e": "ee", "i": "ih", "o": "oh", "u": "oo",
        "m": "mm", "n": "mm", "f": "ff", "v": "ff",
        "t": "th", "d": "th", "s": "ss", "z": "ss",
        "p": "mm", "b": "mm", "w": "oo", "r": "oh",
        "l": "th", "k": "ih", "g": "ih", "h": "aa",
        "j": "ee", "y": "ee", "c": "ss", "x": "ss",
        "q": "oo",
    }

    visemes: list[dict[str, Any]] = []
    time_ms = 0
    phoneme_ms = 80

    for char in text.lower():
        vis = char_to_viseme.get(char, "rest")
        if char == " ":
            vis = "rest"
            duration = 40  # shorter pause for spaces
        else:
            duration = phoneme_ms

        visemes.append({
            "viseme": vis,
            "start_ms": time_ms,
            "duration_ms": duration,
        })
        time_ms += duration

    # End on rest
    visemes.append({
        "viseme": "rest",
        "start_ms": time_ms,
        "duration_ms": 100,
    })

    return _merge_consecutive_visemes(visemes)
