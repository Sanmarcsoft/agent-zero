## Avatar Animation Generator (EchoMimicV2)
generate animated talking-head video from a reference image and audio
uses GPU server running EchoMimicV2 for lip-synced face animation

### avatar_generate
Sends reference image + audio to EchoMimicV2 server, returns MP4 video
- audio_path: path to WAV audio file (from TTS output)
- image_path: path to avatar reference image (PNG/JPEG, frontal portrait)
- pose_style: gesture style (default "01"), options: 01, 02, 03, 04, fight, good, salute, ultraman
- length: max frames (default 120 = 5s at 24fps)
- steps: inference steps (6=fast, 20=quality)

usage:
~~~json
{
    "thoughts": ["Generate an animated video of my avatar speaking this response."],
    "headline": "Generating avatar animation",
    "tool_name": "avatar_generate",
    "tool_args": {
        "audio_path": "/tmp/tts_output.wav",
        "image_path": "/a0/agents/agent0/avatar/profile.png",
        "pose_style": "01",
        "length": "120",
        "steps": "6"
    }
}
~~~
