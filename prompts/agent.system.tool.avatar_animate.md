## Avatar Animation (Real-Time)
animate the avatar with speech and emotion in real-time via WebSocket
uses Q voice (Qwen3-TTS) for speech synthesis and amplitude-based viseme extraction for lip sync

### avatar_animate
Synthesizes speech audio, extracts lip-sync visemes, and broadcasts animation data to connected avatar clients via WebSocket.
- text: the text to speak (required) — the avatar will speak these words with lip sync
- emotion_type: emotional expression to display (default "neutral"), one of: neutral, happiness, sadness, anger, fear, surprise, disgust, worry, pride, shame, curiosity, love
- emotion_severity: intensity of the emotion from 1 (subtle) to 5 (extreme), default 3

The avatar renderer receives the audio, viseme timing, and emotion state via WebSocket and animates in real-time. No video file is generated — animation happens client-side using MediaPipe face landmarks and mesh warping.

usage:
~~~json
{
    "thoughts": ["I should animate my avatar to speak this response with appropriate emotion."],
    "headline": "Animating avatar response",
    "tool_name": "avatar_animate",
    "tool_args": {
        "text": "Hello! It's great to see you. How can I help you today?",
        "emotion_type": "happiness",
        "emotion_severity": "3"
    }
}
~~~

emotion examples:
~~~json
{
    "thoughts": ["The user shared bad news. I should express empathy through my avatar."],
    "headline": "Expressing empathy",
    "tool_name": "avatar_animate",
    "tool_args": {
        "text": "I'm sorry to hear that. That must be really difficult for you.",
        "emotion_type": "sadness",
        "emotion_severity": "2"
    }
}
~~~

~~~json
{
    "thoughts": ["This is an exciting discovery! Let me share it enthusiastically."],
    "headline": "Sharing exciting news",
    "tool_name": "avatar_animate",
    "tool_args": {
        "text": "Wow, that's an incredible finding! The implications are enormous!",
        "emotion_type": "surprise",
        "emotion_severity": "4"
    }
}
~~~
