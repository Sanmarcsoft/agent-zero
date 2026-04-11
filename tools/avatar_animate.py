# Avatar real-time animation tool.
# Synthesizes speech via SanMarcSoft TTS (Qwen3-TTS), extracts visemes,
# and broadcasts animation data to connected avatar clients via WebSocket.

import os
import time
import traceback

from helpers.tool import Tool, Response
from helpers.avatar_bridge import AvatarBridge, EMOTION_TYPES
from helpers.print_style import PrintStyle


class AvatarAnimate(Tool):
    """Animate the avatar with speech and emotion in real-time.

    Uses the Q voice (Qwen3-TTS) for speech synthesis, extracts
    amplitude-based visemes for lip sync, and delivers the animation
    package to connected avatar clients via WebSocket broadcast.
    """

    async def execute(self, **kwargs) -> Response:
        text = self.args.get("text", "").strip()
        emotion_type = self.args.get("emotion_type", "neutral").strip().lower()
        emotion_severity_raw = self.args.get("emotion_severity", "3")

        # ── validate inputs ──────────────────────────────────────────────

        if not text:
            return Response(
                message="Error: 'text' is required — the text for the avatar to speak.",
                break_loop=False,
            )

        if emotion_type not in EMOTION_TYPES:
            return Response(
                message=(
                    f"Error: Unknown emotion_type '{emotion_type}'. "
                    f"Valid options: {', '.join(EMOTION_TYPES)}"
                ),
                break_loop=False,
            )

        try:
            emotion_severity = max(1, min(5, int(emotion_severity_raw)))
        except (TypeError, ValueError):
            emotion_severity = 3

        # ── run animation pipeline ───────────────────────────────────────

        self.set_progress("Synthesizing speech with Q voice...")

        try:
            bridge = AvatarBridge()
            t0 = time.monotonic()

            result = await bridge.process_response(
                text=text,
                emotion_state={
                    "emotion_type": emotion_type,
                    "emotion_severity": emotion_severity,
                },
            )

            elapsed = time.monotonic() - t0

            self.set_progress("Broadcasting animation to avatar...")

            # ── broadcast via WebSocket ──────────────────────────────────
            # Access the WebSocket manager through the agent context to
            # broadcast the animation package to all connected avatar clients.

            broadcast_sent = False
            try:
                ws_manager = _get_websocket_manager()
                if ws_manager is not None:
                    import asyncio
                    await ws_manager.broadcast(
                        "/avatar",
                        "avatar_animate_result",
                        result,
                        handler_id="python.tools.avatar_animate.AvatarAnimate",
                    )
                    broadcast_sent = True
                    PrintStyle.standard(
                        "[AvatarAnimate] Animation broadcast sent to /avatar namespace"
                    )
            except Exception as ws_err:
                PrintStyle.error(
                    f"[AvatarAnimate] WebSocket broadcast failed: {ws_err}"
                )
                # Not fatal — the result is still returned to the agent

            # ── build response ───────────────────────────────────────────

            duration_ms = result.get("duration_ms", 0)
            viseme_count = len(result.get("visemes", []))
            audio_size_kb = len(result.get("audio_b64", "")) * 3 / 4 / 1024

            status_parts = [
                f"Avatar animation ready ({elapsed:.1f}s pipeline):",
                f"  Speech: {duration_ms}ms audio ({audio_size_kb:.0f} KB)",
                f"  Lip sync: {viseme_count} viseme segments",
                f"  Emotion: {emotion_type} (severity {emotion_severity}/5)",
            ]

            if broadcast_sent:
                status_parts.append("  Status: Broadcast sent to avatar clients")
            else:
                status_parts.append(
                    "  Status: Animation packaged (no WebSocket clients connected)"
                )

            return Response(
                message="\n".join(status_parts),
                break_loop=False,
            )

        except Exception as e:
            PrintStyle.error(f"[AvatarAnimate] Pipeline error: {e}")
            PrintStyle.error(traceback.format_exc())
            return Response(
                message=f"Avatar animation error: {e}",
                break_loop=False,
            )


def _get_websocket_manager():
    """Attempt to retrieve the shared WebSocket manager instance.

    Returns None if the manager is not available (e.g., server not started
    or running in a context without WebSocket support).
    """
    try:
        from python.helpers.websocket_manager import WebSocketManager
        # The manager is typically stored on the app or accessed via a
        # module-level reference. Try the common patterns used in agent-zero.
        import python.helpers.runtime as runtime
        manager = getattr(runtime, "_websocket_manager", None)
        if manager is not None and isinstance(manager, WebSocketManager):
            return manager
    except ImportError:
        pass

    try:
        # Fallback: check if there is a global reference via the server module
        from web_ui import get_ws_manager  # type: ignore
        return get_ws_manager()
    except (ImportError, AttributeError):
        pass

    return None
