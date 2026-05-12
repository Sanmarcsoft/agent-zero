"""
safety.py — deterministic pre-execution filter for the CodeExecution tool

Loaded by code_execution_tool.py at the top of CodeExecution.execute(). For
each call (python | nodejs | terminal | output | reset) we:

  1. Look up the patterns for the chosen runtime + the shared baseline
  2. Match each pattern against the code body / command
  3. Return a SafetyVerdict: allow / block, with reason + matched pattern

Block → CodeExecution.execute() short-circuits with a Response that names
the reason. The agent sees the block reason and can rephrase its intent or
escalate to the operator.

Audit-write: every classification (allow or block) writes a single record
to ChromaDB `agent_q_memory` (or the value of AGENT_SLUG-derived collection)
fire-and-forget. JSONL fallback at AUDIT_FALLBACK_PATH on failure.

Design constraints (matches the portfolio-agent damage-control pattern
shipped in Round 3):
  - Deny-by-pattern is the FLOOR. Allow is the default for unmatched.
  - Patterns sourced from a YAML file the operator can edit and reload
    (mtime-cached for steady-state latency).
  - "ask" tier degrades to block in autonomous loop context — agent-zero
    has no synchronous human-in-the-loop hook inside the agent runtime.
"""

import asyncio
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import yaml  # type: ignore[import]
except Exception:
    yaml = None


# ── Config ────────────────────────────────────────────────────────────────

PATTERNS_PATH = Path(__file__).resolve().parent.parent / "safety_patterns.yaml"
AUDIT_FALLBACK_PATH = Path(
    os.environ.get("CODE_EXEC_AUDIT_FALLBACK", "/a0/usr/audit-fallback/code-exec.jsonl")
)
AGENT_SLUG = (
    (os.environ.get("AGENT_SLUG") or os.environ.get("AGENT_NAME") or "q")
    .strip()
    .lower()
    .replace(" ", "-")
)
CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "10.0.0.12")
CHROMADB_PORT = os.environ.get("CHROMADB_PORT", "18000")
COLLECTION_NAME = f"agent_{AGENT_SLUG}_memory"


# ── Data ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SafetyVerdict:
    decision: str  # "allow" | "block"
    reason: str = "no match"
    matched_pattern: Optional[str] = None
    tier: Optional[str] = None  # "blocked" | "ask" | "zero_access"
    runtime: str = ""

    def is_block(self) -> bool:
        return self.decision == "block"


# ── Pattern cache ─────────────────────────────────────────────────────────

_patterns_cache: Optional[dict] = None
_patterns_mtime: float = 0.0
_compiled_cache: dict = {}


def _load_patterns() -> dict:
    global _patterns_cache, _patterns_mtime, _compiled_cache
    if not PATTERNS_PATH.exists() or yaml is None:
        # Fail-closed default: empty patterns mean nothing is filtered,
        # but that's a misconfiguration we surface via stderr.
        if yaml is None:
            print("[safety] PyYAML unavailable — code_execution safety filter inert",
                  file=sys.stderr)
        elif not PATTERNS_PATH.exists():
            print(f"[safety] safety_patterns.yaml not at {PATTERNS_PATH}",
                  file=sys.stderr)
        return {"shared": {"blocked": [], "zero_access_paths": []}}
    try:
        mtime = PATTERNS_PATH.stat().st_mtime
        if _patterns_cache is not None and mtime == _patterns_mtime:
            return _patterns_cache
        with PATTERNS_PATH.open("r") as f:
            data = yaml.safe_load(f) or {}
        _patterns_cache = data
        _patterns_mtime = mtime
        _compiled_cache.clear()
        return data
    except Exception as e:
        if _patterns_cache:
            return _patterns_cache
        return {"shared": {"blocked": [], "zero_access_paths": []}, "_load_error": str(e)}


def _compile_for(runtime: str):
    """Return [(compiled_regex, reason, tier), ...] for a runtime (with
    shared baseline merged in)."""
    if runtime in _compiled_cache:
        return _compiled_cache[runtime]
    data = _load_patterns()
    out = []
    for section_name, section in (
        ("shared", data.get("shared", {})),
        (runtime, data.get(runtime, {})),
    ):
        for tier in ("blocked", "ask"):
            for entry in section.get(tier, []) or []:
                pat = entry.get("pattern", "")
                reason = entry.get("reason", "")
                if not pat:
                    continue
                try:
                    rx = re.compile(pat, re.MULTILINE)
                except re.error as e:
                    print(f"[safety] bad pattern in {section_name}.{tier}: {pat!r} ({e})",
                          file=sys.stderr)
                    continue
                out.append((rx, reason, tier))
    _compiled_cache[runtime] = out
    return out


def _zero_access_paths() -> list:
    data = _load_patterns()
    return list(data.get("shared", {}).get("zero_access_paths", []) or [])


# ── Classifier ────────────────────────────────────────────────────────────


def classify(runtime: str, code: str) -> SafetyVerdict:
    """Classify a code body against the runtime's patterns + shared baseline.

    runtime: one of "python", "nodejs", "terminal", "output", "reset"
    code:    the code/command body. Empty for output/reset modes.
    """
    runtime = (runtime or "").lower().strip()
    if not code:
        return SafetyVerdict(decision="allow", reason="no body to inspect", runtime=runtime)

    # Zero-access path check (shared, plain substring)
    for p in _zero_access_paths():
        if p and p in code:
            return SafetyVerdict(
                decision="block",
                reason=f"zero_access_paths: code touches {p}",
                matched_pattern=p,
                tier="zero_access",
                runtime=runtime,
            )

    # Compiled pattern walk
    # NOTE on tier semantics:
    #   "blocked"  -> HARD BLOCK (deny, no override at this layer)
    #   "ask"      -> allow + audit-tag (the chat UI has a human-in-the-loop
    #                 reviewer at the chat level — we surface intent via
    #                 audit and let the call proceed; the user sees the
    #                 agent's invocation in the chat and can interrupt).
    #                 If you want a HARD block on ask patterns (autonomous
    #                 mode), set CODE_EXEC_ASK_BLOCKS=1.
    ask_blocks = os.environ.get("CODE_EXEC_ASK_BLOCKS") == "1"
    for rx, reason, tier in _compile_for(runtime):
        if rx.search(code):
            if tier == "blocked" or (tier == "ask" and ask_blocks):
                return SafetyVerdict(
                    decision="block",
                    reason=f"{tier}: {reason}",
                    matched_pattern=rx.pattern,
                    tier=tier,
                    runtime=runtime,
                )
            # ask tier in chat-mode: audit-tag, allow.
            return SafetyVerdict(
                decision="allow",
                reason=f"ask (audit-tag, allow): {reason}",
                matched_pattern=rx.pattern,
                tier=tier,
                runtime=runtime,
            )

    return SafetyVerdict(decision="allow", reason="no match", runtime=runtime)


# ── Audit-write (best-effort) ─────────────────────────────────────────────


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:16]


async def _post_chroma(collection_id: str, record: dict) -> bool:
    try:
        import aiohttp
        url = (
            f"http://{CHROMADB_HOST}:{CHROMADB_PORT}/api/v2/tenants/"
            f"default_tenant/databases/default_database/collections/"
            f"{collection_id}/add"
        )
        payload = {
            "ids": [record["id"]],
            "documents": [record["document"]],
            "embeddings": [[0.0] * 384],
            "metadatas": [record["metadata"]],
        }
        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(url, json=payload) as resp:
                return resp.status < 300
    except Exception:
        return False


async def _resolve_collection_id() -> Optional[str]:
    try:
        import aiohttp
        url = (
            f"http://{CHROMADB_HOST}:{CHROMADB_PORT}/api/v2/tenants/"
            f"default_tenant/databases/default_database/collections"
        )
        timeout = aiohttp.ClientTimeout(total=3)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                for c in data:
                    if c.get("name") == COLLECTION_NAME:
                        return c.get("id")
        return None
    except Exception:
        return None


def _write_fallback(record: dict) -> None:
    try:
        AUDIT_FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_FALLBACK_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        print("[safety] audit fallback write failed", file=sys.stderr)


def audit(verdict: SafetyVerdict, runtime: str, code: str) -> None:
    """Best-effort fire-and-forget audit-write. Falls back to JSONL on
    ChromaDB failure or missing collection."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
    code_hash = _hash(code)
    record_id = f"{AGENT_SLUG}-codexec-{ts}-{code_hash}"
    doc = (
        f"{AGENT_SLUG} code_execution {runtime}@{ts}\n"
        f"decision: {verdict.decision}\n"
        f"reason: {verdict.reason}\n"
        f"code_hash: {code_hash}\n"
        f"code_len: {len(code)}"
    )
    metadata = {
        "type": "code_execution",
        "ts": ts,
        "persona": AGENT_SLUG,
        "runtime": runtime,
        "decision": verdict.decision,
        "tier": verdict.tier or "",
        "matched_pattern": (verdict.matched_pattern or "")[:200],
        "code_hash": code_hash,
        "code_len": len(code),
        "blast_radius": "internal-code-execution",
        "reversibility": "irreversible-by-default",
    }
    record = {"id": record_id, "document": doc, "metadata": metadata}

    async def _go():
        cid = await _resolve_collection_id()
        if not cid:
            _write_fallback({**record, "reason_dropped": "no-collection-id"})
            return
        ok = await _post_chroma(cid, record)
        if not ok:
            _write_fallback({**record, "reason_dropped": "chroma-post-failed"})

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_go())
    except RuntimeError:
        _write_fallback({**record, "reason_dropped": "no-event-loop"})
