# Avatar video generation tool using EchoMimicV2 server.
# Sends reference image + TTS audio to the GPU server and returns animated MP4.

import os
import tempfile
import httpx

from helpers.tool import Tool, Response

ECHOMIMIC_URL = os.environ.get("ECHOMIMIC_URL", "http://10.0.0.96:8090")


class AvatarGenerate(Tool):
    """Generate animated avatar video from a reference image and audio."""

    async def execute(self, **kwargs):
        audio_path = kwargs.get("audio_path", "")
        image_path = kwargs.get("image_path", "")
        pose_style = kwargs.get("pose_style", "01")
        length = int(kwargs.get("length", 120))
        steps = int(kwargs.get("steps", 6))

        if not audio_path:
            return Response(
                message="Error: 'audio_path' required — path to WAV file from TTS.",
                break_loop=False,
            )
        if not image_path:
            return Response(
                message="Error: 'image_path' required — path to avatar reference image.",
                break_loop=False,
            )

        if not os.path.exists(audio_path):
            return Response(message=f"Audio file not found: {audio_path}", break_loop=False)
        if not os.path.exists(image_path):
            return Response(message=f"Image file not found: {image_path}", break_loop=False)

        # Check server health
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                health = await client.get(f"{ECHOMIMIC_URL}/health")
                if health.status_code != 200:
                    return Response(
                        message=f"EchoMimicV2 server not ready: {health.text}",
                        break_loop=False,
                    )
        except Exception as e:
            return Response(
                message=f"Cannot reach EchoMimicV2 server at {ECHOMIMIC_URL}: {e}",
                break_loop=False,
            )

        # Send generation request
        self.set_progress("Generating avatar animation...")
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                with open(image_path, "rb") as img_f, open(audio_path, "rb") as aud_f:
                    resp = await client.post(
                        f"{ECHOMIMIC_URL}/generate",
                        files={
                            "image": ("avatar.png", img_f, "image/png"),
                            "audio": ("speech.wav", aud_f, "audio/wav"),
                        },
                        data={
                            "pose_style": pose_style,
                            "length": str(length),
                            "steps": str(steps),
                        },
                    )

                if resp.status_code != 200:
                    return Response(
                        message=f"EchoMimicV2 generation failed ({resp.status_code}): {resp.text[:200]}",
                        break_loop=False,
                    )

                # Save the MP4 to a temp file
                output_dir = tempfile.mkdtemp(prefix="avatar_")
                output_path = os.path.join(output_dir, "avatar_animation.mp4")
                with open(output_path, "wb") as f:
                    f.write(resp.content)

                frames = resp.headers.get("X-EchoMimic-Frames", "?")
                seed = resp.headers.get("X-EchoMimic-Seed", "?")

                return Response(
                    message=(
                        f"Avatar animation generated: {output_path}\n"
                        f"Frames: {frames}, Pose: {pose_style}, Seed: {seed}\n"
                        f"Size: {len(resp.content) / 1024:.0f} KB"
                    ),
                    break_loop=False,
                )

        except httpx.TimeoutException:
            return Response(
                message="EchoMimicV2 generation timed out (>5 min). Try fewer frames or steps.",
                break_loop=False,
            )
        except Exception as e:
            return Response(message=f"Avatar generation error: {e}", break_loop=False)
