# avatar_handler.py — WebSocket handler for real-time avatar animation
#
# Namespace: /avatar (derived from filename avatar_handler.py)
#
# Events handled:
#   avatar_animate  — receives text + emotion, runs TTS pipeline, broadcasts animation
#   avatar_prepare  — receives image bytes, forwards to LivePortrait for face prep
#   avatar_emotion  — real-time emotion update (no TTS, just emotion change)
#
# Outbound events broadcast to all connected avatar clients:
#   avatar_animate_result — animation package (audio + visemes + emotion)
#   avatar_prepare_result — face model data from LivePortrait
#   avatar_emotion_update — emotion state change notification

from __future__ import annotations

import base64
import time
import traceback
from typing import Any

from python.helpers.avatar_bridge import AvatarBridge, EMOTION_TYPES
from python.helpers.print_style import PrintStyle
from python.helpers.websocket import WebSocketHandler, WebSocketResult


class AvatarHandler(WebSocketHandler):
    """WebSocket handler for the avatar animation pipeline.

    Connected clients (avatar-test.html or production avatar UIs) receive
    real-time animation packages containing audio, viseme timing, and
    emotion state updates.
    """

    @classmethod
    def get_event_types(cls) -> list[str]:
        return ["avatar_animate", "avatar_prepare", "avatar_emotion"]

    @classmethod
    def requires_auth(cls) -> bool:
        # Avatar UI may run on local ASUS laptop without session auth
        return False

    @classmethod
    def requires_csrf(cls) -> bool:
        return False

    async def on_connect(self, sid: str) -> None:
        PrintStyle.standard(f"[AvatarHandler] Client connected: {sid}")

    async def on_disconnect(self, sid: str) -> None:
        PrintStyle.standard(f"[AvatarHandler] Client disconnected: {sid}")

    async def process_event(
        self,
        event_type: str,
        data: dict[str, Any],
        sid: str,
    ) -> dict[str, Any] | WebSocketResult | None:
        correlation_id = data.get("correlationId")

        if event_type == "avatar_animate":
            return await self._handle_animate(data, sid, correlation_id)
        elif event_type == "avatar_prepare":
            return await self._handle_prepare(data, sid, correlation_id)
        elif event_type == "avatar_emotion":
            return await self._handle_emotion(data, sid, correlation_id)
        else:
            return self.result_error(
                code="UNKNOWN_EVENT",
                message=f"Unknown event type: {event_type}",
                correlation_id=correlation_id,
            )

    # ── avatar_animate: full TTS + viseme pipeline ──────────────────────

    async def _handle_animate(
        self,
        data: dict[str, Any],
        sid: str,
        correlation_id: str | None,
    ) -> WebSocketResult:
        text = data.get("text", "").strip()
        if not text:
            return self.result_error(
                code="MISSING_TEXT",
                message="'text' is required for avatar animation",
                correlation_id=correlation_id,
            )

        emotion_type = data.get("emotion_type", "neutral")
        if emotion_type not in EMOTION_TYPES:
            emotion_type = "neutral"

        emotion_severity = data.get("emotion_severity", 3)
        try:
            emotion_severity = max(1, min(5, int(emotion_severity)))
        except (TypeError, ValueError):
            emotion_severity = 3

        PrintStyle.standard(
            f"[AvatarHandler] animate request from {sid}: "
            f"{len(text)} chars, emotion={emotion_type}({emotion_severity})"
        )

        try:
            bridge = AvatarBridge()
            result = await bridge.process_response(
                text=text,
                emotion_state={
                    "emotion_type": emotion_type,
                    "emotion_severity": emotion_severity,
                },
            )

            # Broadcast animation to all connected avatar clients
            await self.broadcast(
                "avatar_animate_result",
                result,
                correlation_id=correlation_id,
            )

            return self.result_ok(
                data={
                    "status": "broadcast_sent",
                    "duration_ms": result.get("duration_ms", 0),
                    "viseme_count": len(result.get("visemes", [])),
                    "audio_size": len(result.get("audio_b64", "")),
                },
                correlation_id=correlation_id,
            )

        except Exception as e:
            PrintStyle.error(f"[AvatarHandler] animate error: {e}")
            PrintStyle.error(traceback.format_exc())
            return self.result_error(
                code="ANIMATE_FAILED",
                message=f"Avatar animation failed: {str(e)}",
                correlation_id=correlation_id,
            )

    # ── avatar_prepare: face model preparation via LivePortrait ─────────

    async def _handle_prepare(
        self,
        data: dict[str, Any],
        sid: str,
        correlation_id: str | None,
    ) -> WebSocketResult:
        image_b64 = data.get("image_b64", "")
        if not image_b64:
            return self.result_error(
                code="MISSING_IMAGE",
                message="'image_b64' is required for face preparation",
                correlation_id=correlation_id,
            )

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            return self.result_error(
                code="INVALID_IMAGE",
                message="'image_b64' is not valid base64",
                correlation_id=correlation_id,
            )

        if len(image_bytes) < 100:
            return self.result_error(
                code="INVALID_IMAGE",
                message="Image data too small to be a valid image",
                correlation_id=correlation_id,
            )

        PrintStyle.standard(
            f"[AvatarHandler] prepare request from {sid}: "
            f"{len(image_bytes)} bytes image"
        )

        try:
            bridge = AvatarBridge()
            face_model = await bridge.prepare_face(image_bytes)

            # Broadcast face model to all connected clients
            prepare_result = {
                "type": "face_prepared",
                "face_model": face_model,
                "timestamp": int(time.time() * 1000),
            }
            await self.broadcast(
                "avatar_prepare_result",
                prepare_result,
                correlation_id=correlation_id,
            )

            return self.result_ok(
                data={
                    "status": "face_prepared",
                    "model_keys": list(face_model.keys()) if isinstance(face_model, dict) else [],
                },
                correlation_id=correlation_id,
            )

        except Exception as e:
            PrintStyle.error(f"[AvatarHandler] prepare error: {e}")
            PrintStyle.error(traceback.format_exc())
            return self.result_error(
                code="PREPARE_FAILED",
                message=f"Face preparation failed: {str(e)}",
                correlation_id=correlation_id,
            )

    # ── avatar_emotion: real-time emotion state update ──────────────────

    async def _handle_emotion(
        self,
        data: dict[str, Any],
        sid: str,
        correlation_id: str | None,
    ) -> WebSocketResult:
        emotion_type = data.get("emotion_type", "neutral")
        if emotion_type not in EMOTION_TYPES:
            return self.result_error(
                code="INVALID_EMOTION",
                message=f"Unknown emotion_type '{emotion_type}'. "
                        f"Valid: {', '.join(EMOTION_TYPES)}",
                correlation_id=correlation_id,
            )

        emotion_severity = data.get("emotion_severity", 3)
        try:
            emotion_severity = max(1, min(5, int(emotion_severity)))
        except (TypeError, ValueError):
            emotion_severity = 3

        PrintStyle.standard(
            f"[AvatarHandler] emotion update from {sid}: "
            f"{emotion_type}({emotion_severity})"
        )

        # Broadcast emotion change to all connected avatar clients
        emotion_update = {
            "type": "emotion_update",
            "emotion": {
                "type": emotion_type,
                "severity": emotion_severity,
            },
            "timestamp": int(time.time() * 1000),
        }
        await self.broadcast(
            "avatar_emotion_update",
            emotion_update,
            correlation_id=correlation_id,
        )

        return self.result_ok(
            data={
                "status": "emotion_broadcast",
                "emotion_type": emotion_type,
                "emotion_severity": emotion_severity,
            },
            correlation_id=correlation_id,
        )
