## Service Health & Kickstart
check status and restart dependent services: TTS, STT, Ollama, ChromaDB, SearXNG, Moneypenny
use proactively when services fail or at session start if something seems broken

### service_health
- action: status | restart | kickstart
- service: tts | stt | ollama | chromadb | searxng | moneypenny (optional for status, required for restart)

Check all services:
~~~json
{
    "thoughts": ["Let me check if all my backend services are healthy."],
    "headline": "Checking service health",
    "tool_name": "service_health",
    "tool_args": {
        "action": "status"
    }
}
~~~

Restart a specific service:
~~~json
{
    "thoughts": ["The TTS server appears to be down. Let me restart it."],
    "headline": "Restarting TTS service",
    "tool_name": "service_health",
    "tool_args": {
        "action": "restart",
        "service": "tts"
    }
}
~~~

Kickstart all down services:
~~~json
{
    "thoughts": ["Multiple services seem unavailable. I will kickstart everything."],
    "headline": "Kickstarting all services",
    "tool_name": "service_health",
    "tool_args": {
        "action": "kickstart"
    }
}
~~~
