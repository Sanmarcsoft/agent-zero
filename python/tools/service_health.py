# Service health check and kickstart tool for Agent Zero.
# Allows Q to check and restart dependent services.

import subprocess
import asyncio
import urllib.request
import json

from python.helpers.tool import Tool, Response


SERVICES = {
    "tts": {
        "name": "TTS (Qwen3-TTS)",
        "health_url": "http://10.0.0.96:8880/health",
        "restart_cmd": "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 matthewstevens@10.0.0.96 'cd /opt/local/ai-services/Qwen3-TTS && nohup /opt/homebrew/bin/python3.14 tts_server.py > /tmp/tts.log 2>&1 &'",
    },
    "stt": {
        "name": "STT (Whisper)",
        "health_url": "http://10.0.0.96:8002/health",
        "restart_cmd": "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 matthewstevens@10.0.0.96 'cd /opt/local/ai-services/stt-server && nohup ./venv/bin/python3 stt_server.py > /tmp/stt.log 2>&1 &'",
    },
    "ollama": {
        "name": "Ollama (Embeddings & LLM)",
        "health_url": "http://10.0.0.96:11434/",
        "restart_cmd": "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 matthewstevens@10.0.0.96 'open -a Ollama 2>/dev/null || /usr/local/bin/ollama serve &'",
    },
    "chromadb": {
        "name": "ChromaDB (Vector DB)",
        "health_url": "http://10.0.0.12:18000/api/v2/heartbeat",
        "restart_cmd": "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 matt@10.0.0.12 'PATH=/usr/local/bin:$PATH docker restart ragmnt-chroma-rag-1'",
    },
    "searxng": {
        "name": "SearXNG (Web Search)",
        "health_url": "http://10.0.0.12:8081/",
        "restart_cmd": "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 matt@10.0.0.12 'PATH=/usr/local/bin:$PATH docker restart ragmnt-searxng-rag-1'",
    },
    "moneypenny": {
        "name": "Moneypenny (MCP Server)",
        "health_url": "http://moneypenny:3100/health",
        "restart_cmd": "docker restart moneypenny 2>/dev/null || ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 matt@10.0.0.112 'docker restart moneypenny'",
    },
}


def _check_health(url: str, timeout: int = 5) -> dict:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = {"raw": body[:200]}
            return {"status": "ok", "code": resp.status, "data": data}
    except Exception as e:
        return {"status": "down", "error": str(e)}


def _restart_service(cmd: str) -> str:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return "Restart command sent successfully."
        return f"Restart returned code {result.returncode}: {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return "Restart command timed out (30s)."
    except Exception as e:
        return f"Restart failed: {e}"


class ServiceHealth(Tool):
    async def execute(self, action="status", service="", **kwargs):
        action = (action or "status").strip().lower()

        if action == "status":
            return await self._status(service)
        elif action == "restart":
            return await self._restart(service)
        elif action == "kickstart":
            return await self._kickstart_all()
        else:
            return Response(
                message="Actions: status, restart, kickstart. Services: " + ", ".join(SERVICES.keys()),
                break_loop=False,
            )

    async def _status(self, service: str) -> Response:
        loop = asyncio.get_event_loop()
        targets = {service: SERVICES[service]} if service in SERVICES else SERVICES

        lines = ["**Service Health Report**\n"]
        all_ok = True
        for key, svc in targets.items():
            result = await loop.run_in_executor(None, _check_health, svc["health_url"])
            icon = "OK" if result["status"] == "ok" else "DOWN"
            if result["status"] != "ok":
                all_ok = False
            detail = ""
            if result["status"] == "ok" and isinstance(result.get("data"), dict):
                d = result["data"]
                extras = []
                if "voices" in d:
                    extras.append(f"voices: {d['voices']}")
                if "tools" in d:
                    extras.append(f"tools: {d['tools']}")
                if "model" in d:
                    extras.append(f"model: {d['model']}")
                if extras:
                    detail = " (" + ", ".join(extras) + ")"
            elif result["status"] != "ok":
                detail = f" — {result.get('error', 'unknown')}"
            lines.append(f"- {svc['name']}: **{icon}**{detail}")

        summary = "All services healthy." if all_ok else "Some services need attention."
        lines.append(f"\n{summary}")
        return Response(message="\n".join(lines), break_loop=False)

    async def _restart(self, service: str) -> Response:
        if service not in SERVICES:
            return Response(
                message=f"Unknown service '{service}'. Available: {', '.join(SERVICES.keys())}",
                break_loop=False,
            )
        svc = SERVICES[service]
        self.set_progress(f"Restarting {svc['name']}...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _restart_service, svc["restart_cmd"])
        return Response(
            message=f"**{svc['name']}**: {result}",
            break_loop=False,
        )

    async def _kickstart_all(self) -> Response:
        self.set_progress("Kickstarting all services...")
        loop = asyncio.get_event_loop()
        lines = ["**Kickstarting all services...**\n"]

        for key, svc in SERVICES.items():
            health = await loop.run_in_executor(None, _check_health, svc["health_url"])
            if health["status"] == "ok":
                lines.append(f"- {svc['name']}: already running, skipped")
            else:
                result = await loop.run_in_executor(None, _restart_service, svc["restart_cmd"])
                lines.append(f"- {svc['name']}: was DOWN, {result}")

        lines.append("\nWaiting 10 seconds for services to start...")
        await asyncio.sleep(10)

        # Recheck
        lines.append("\n**Post-kickstart status:**")
        for key, svc in SERVICES.items():
            health = await loop.run_in_executor(None, _check_health, svc["health_url"])
            icon = "OK" if health["status"] == "ok" else "STILL DOWN"
            lines.append(f"- {svc['name']}: {icon}")

        return Response(message="\n".join(lines), break_loop=False)
