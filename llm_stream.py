"""Unified LLM streaming library — sync (requests) and async (httpx) generators
yielding text tokens from 7 providers: anthropic, openai, azure, google,
digitalocean, puter, bedrock.

    from llm_stream import stream_chat, astream_chat, resolve_model, MODEL_MAP
"""

import hashlib, hmac, json, logging, os, struct
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Generator, Optional
from urllib.parse import quote, urlparse

log = logging.getLogger(__name__)


class LLMStreamError(Exception):
    """Raised on provider configuration or protocol errors."""
    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"[{provider}] {message}")


# ── Model maps ────────────────────────────────────────────────────────────

MODEL_MAP: dict[str, dict[str, str]] = {
    # -- Anthropic Claude --
    "claude-opus-4.6":   {"anthropic": "claude-opus-4-6", "digitalocean": "anthropic-claude-opus-4.6", "bedrock": "us.anthropic.claude-opus-4-6-v1", "puter": "claude-opus-4-6"},
    "claude-sonnet-4.6": {"anthropic": "claude-sonnet-4-6", "digitalocean": "anthropic-claude-4.6-sonnet", "bedrock": "us.anthropic.claude-sonnet-4-6", "puter": "claude-sonnet-4-6"},
    "claude-haiku-4.5":  {"anthropic": "claude-haiku-4-5-20251001", "digitalocean": "anthropic-claude-haiku-4.5", "bedrock": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "puter": "claude-haiku-4-5-20251001"},
    "claude-opus-4.5":   {"anthropic": "claude-opus-4-5-20251101", "digitalocean": "anthropic-claude-opus-4.5", "bedrock": "us.anthropic.claude-opus-4-5-20251101-v1:0", "puter": "claude-opus-4-5-20251101"},
    "claude-sonnet-4.5": {"anthropic": "claude-sonnet-4-5-20250929", "digitalocean": "anthropic-claude-4.5-sonnet", "bedrock": "us.anthropic.claude-sonnet-4-5-20250929-v1:0", "puter": "claude-sonnet-4-5-20250929"},
    "claude-sonnet-4":   {"anthropic": "claude-sonnet-4-20250514", "digitalocean": "anthropic-claude-sonnet-4", "bedrock": "us.anthropic.claude-sonnet-4-20250514-v1:0"},
    # -- OpenAI --
    "gpt-4o":            {"openai": "gpt-4o", "digitalocean": "openai-gpt-4o", "azure": "gpt-4o", "puter": "gpt-4o"},
    "gpt-4o-mini":       {"openai": "gpt-4o-mini", "digitalocean": "openai-gpt-4o-mini", "azure": "gpt-4o-mini", "puter": "gpt-4o-mini"},
    "o4-mini":           {"openai": "o4-mini", "digitalocean": "openai-o3-mini", "puter": "o4-mini"},
    "gpt-5.3":           {"openai": "gpt-5.3-chat", "digitalocean": "openai-gpt-5.3-codex", "puter": "gpt-5.2-chat-latest"},
    # -- Meta Llama --
    "llama-3.3-70b":     {"digitalocean": "llama3.3-70b-instruct", "azure": "Llama-3.3-70B-Instruct", "puter": "openrouter:meta-llama/llama-3.3-70b-instruct"},
    "llama4-maverick-17b": {"bedrock": "us.meta.llama4-maverick-17b-instruct-v1:0"},
    "llama4-scout-17b":  {"bedrock": "us.meta.llama4-scout-17b-instruct-v1:0"},
    # -- Google --
    "gemini-2.5-flash":  {"google": "gemini-2.5-flash", "puter": "gemini-2.5-flash"},
    "gemini-2.5-pro":    {"google": "gemini-2.5-pro", "puter": "gemini-2.5-pro"},
    "gemini-3.1-pro-preview": {"google": "gemini-3.1-pro-preview"},
    # -- Other --
    "deepseek-r1":       {"digitalocean": "deepseek-r1-distill-llama-70b", "azure": "DeepSeek-R1", "puter": "deepseek-reasoner"},
    "grok-3":            {"azure": "grok-3", "puter": "grok-3"},
    # -- Bedrock-only (Amazon, Writer) --
    "nova-pro":          {"bedrock": "us.amazon.nova-pro-v1:0"},
    "nova-lite":         {"bedrock": "us.amazon.nova-lite-v1:0"},
    "nova-2-lite":       {"bedrock": "us.amazon.nova-2-lite-v1:0"},
    "palmyra-x4":        {"bedrock": "us.writer.palmyra-x4-v1:0"},
    "palmyra-x5":        {"bedrock": "us.writer.palmyra-x5-v1:0"},
}

# Derived from MODEL_MAP — single source of truth for Bedrock model IDs
BEDROCK_MODELS: dict[str, str] = {
    canonical: providers["bedrock"]
    for canonical, providers in MODEL_MAP.items()
    if "bedrock" in providers
}

AZURE_SERVERLESS = [
    "grok-3", "grok-3-mini", "DeepSeek-R1",
    "Meta-Llama-3.1-405B-Instruct", "Meta-Llama-3.1-8B-Instruct",
    "Llama-3.2-11B-Vision-Instruct", "Llama-3.2-90B-Vision-Instruct",
    "Llama-3.3-70B-Instruct", "Llama-4-Scout-17B-16E-Instruct",
    "Phi-4", "Cohere-command-r-plus-08-2024", "Cohere-command-r-08-2024",
    "Codestral-2501", "Ministral-3B",
]


# ── Config loading ────────────────────────────────────────────────────────

def _load_default_config() -> dict:
    """Auto-load config from ~/.config/ paths (CCA + speech-to-cli)."""
    config: dict[str, Any] = {}
    for path in ("~/.config/cloud-chat-assistant/config.json",
                 "~/.config/speech-to-cli/config.json"):
        try:
            with open(os.path.expanduser(path)) as f:
                config.update(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return config


def _cfg(config: Optional[dict]) -> dict:
    return config if config is not None else _load_default_config()


# ── Model resolution ──────────────────────────────────────────────────────

def resolve_model(canonical: str, provider: str) -> str:
    """Return provider-specific model ID for a canonical name. Falls back to canonical."""
    mapping = MODEL_MAP.get(canonical)
    if mapping:
        resolved = mapping.get(provider, canonical)
        if resolved != canonical:
            return resolved
        log.debug("No %s mapping for %s, using canonical", provider, canonical)
    return canonical


def _get_bedrock_model_id(model_name: str) -> Optional[str]:
    """Convert friendly model name to Bedrock model ID."""
    if model_name in BEDROCK_MODELS:
        return BEDROCK_MODELS[model_name]
    if "." in model_name or ":" in model_name:
        return model_name
    return None


# ── AWS SigV4 signing ─────────────────────────────────────────────────────

def _aws_sign_v4(method: str, url: str, payload: bytes,
                 access_key: str, secret_key: str,
                 region: str, service: str = "bedrock") -> dict[str, str]:
    """Sign an AWS request using Signature Version 4. Returns headers dict."""
    parsed = urlparse(url)
    host, uri = parsed.netloc, quote(parsed.path or "/", safe="/-_.~")
    t = datetime.now(timezone.utc)
    amz_date, date_stamp = t.strftime("%Y%m%dT%H%M%SZ"), t.strftime("%Y%m%d")

    h2s = {"host": host, "x-amz-date": amz_date, "content-type": "application/json"}
    signed_hdr = ";".join(sorted(h2s))
    canon_hdr = "".join(f"{k}:{v}\n" for k, v in sorted(h2s.items()))
    ph = hashlib.sha256(payload).hexdigest()

    canon_req = f"{method}\n{uri}\n\n{canon_hdr}\n{signed_hdr}\n{ph}"
    algo = "AWS4-HMAC-SHA256"
    scope = f"{date_stamp}/{region}/{service}/aws4_request"
    sts = f"{algo}\n{amz_date}\n{scope}\n{hashlib.sha256(canon_req.encode()).hexdigest()}"

    def _s(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    k = _s(_s(_s(_s(f"AWS4{secret_key}".encode(), date_stamp), region), service), "aws4_request")
    sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()

    return {"host": host, "x-amz-date": amz_date, "content-type": "application/json",
            "authorization": f"{algo} Credential={access_key}/{scope}, SignedHeaders={signed_hdr}, Signature={sig}"}


# ── Bedrock binary event-stream parser ────────────────────────────────────

def _parse_event_stream(data: bytes) -> Generator[tuple[str, dict], None, None]:
    """Parse AWS binary event-stream frames.

    Frame: [total_len:4BE][headers_len:4BE][prelude_crc:4][headers:N][payload:M][msg_crc:4]
    Yields (event_type, payload_dict).
    """
    off = 0
    while off + 12 <= len(data):
        total_len, headers_len, _ = struct.unpack_from(">III", data, off)
        if off + total_len > len(data):
            break
        hdr_start, hdr_end = off + 12, off + 12 + headers_len
        headers = _parse_event_headers(data[hdr_start:hdr_end])
        payload_bytes = data[hdr_end:off + total_len - 4]
        event_type = headers.get(":event-type", headers.get(":exception-type", "unknown"))
        payload_dict: dict = {}
        if payload_bytes:
            try:
                payload_dict = json.loads(payload_bytes)
            except (json.JSONDecodeError, ValueError):
                pass
        yield event_type, payload_dict
        off += total_len


def _parse_event_headers(data: bytes) -> dict[str, str]:
    """Parse event-stream header block into a dict."""
    headers: dict[str, str] = {}
    pos = 0
    while pos < len(data):
        name_len = data[pos]; pos += 1
        if pos + name_len > len(data): break
        name = data[pos:pos + name_len].decode(); pos += name_len
        if pos >= len(data): break
        hdr_type = data[pos]; pos += 1
        if hdr_type == 7:  # string
            if pos + 2 > len(data): break
            val_len = struct.unpack_from(">H", data, pos)[0]; pos += 2
            if pos + val_len > len(data): break
            headers[name] = data[pos:pos + val_len].decode(); pos += val_len
        else:
            break
    return headers


# ── SSE line parsers ──────────────────────────────────────────────────────

def _parse_sse_line(line: str, provider: str) -> Optional[str]:
    """Extract text token from an SSE line for any provider."""
    if not line or not line.startswith("data: "):
        return None
    data_str = line[6:]
    if data_str.strip() == "[DONE]":
        return None
    try:
        ev = json.loads(data_str)
    except (json.JSONDecodeError, ValueError):
        return None
    if provider == "anthropic":
        if ev.get("type") == "content_block_delta":
            return ev.get("delta", {}).get("text") or None
    elif provider == "google":
        cands = ev.get("candidates", [])
        if cands:
            parts = cands[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text") or None
    else:  # openai, azure, digitalocean, puter
        choices = ev.get("choices", [])
        if choices:
            return choices[0].get("delta", {}).get("content") or None
    return None


# ── Message preparation ──────────────────────────────────────────────────

def _prepare_messages(messages: list[dict], system_prompt: Optional[str]
                      ) -> tuple[Optional[str], list[dict]]:
    """Extract system prompt from messages if not provided. Returns (sys, cleaned)."""
    sys_parts, cleaned = [], []
    for msg in messages:
        if msg["role"] == "system":
            sys_parts.append(msg["content"])
        else:
            cleaned.append(msg)
    if system_prompt is None and sys_parts:
        system_prompt = "\n".join(sys_parts)
    return system_prompt, cleaned


def _openai_msgs(system_prompt: Optional[str], msgs: list[dict]) -> list[dict]:
    """Prepend system message to an OpenAI-format message list."""
    out: list[dict] = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})
    out.extend(msgs)
    return out


# ── Request builders ─────────────────────────────────────────────────────

def _build_request(provider: str, model: str, messages: list[dict],
                   system_prompt: Optional[str], cfg: dict
                   ) -> tuple[str, dict[str, str], dict]:
    """Build (url, headers, body) for a provider's streaming request."""
    sp, msgs = _prepare_messages(messages, system_prompt)
    mt = cfg.get("max_tokens", 1024)

    if provider == "anthropic":
        key = cfg.get("api_key") or cfg.get("llm_api_key", "")
        if not key: raise LLMStreamError(provider, "No API key configured")
        body: dict[str, Any] = {"model": model, "max_tokens": mt, "messages": msgs, "stream": True}
        if sp: body["system"] = sp
        return ("https://api.anthropic.com/v1/messages",
                {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                body)

    if provider == "openai":
        key = cfg.get("api_key") or cfg.get("llm_api_key", "")
        if not key: raise LLMStreamError(provider, "No API key configured")
        return ("https://api.openai.com/v1/chat/completions",
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                {"model": model, "messages": _openai_msgs(sp, msgs), "max_tokens": mt, "stream": True})

    if provider == "azure":
        ak, ep = cfg.get("api_key", ""), cfg.get("endpoint", "")
        if not ak or not ep: raise LLMStreamError(provider, "Azure AI not configured (need endpoint + api_key)")
        fm = _openai_msgs(sp, msgs)
        if model in AZURE_SERVERLESS:
            url = f"{ep}/models/chat/completions?api-version=2024-12-01-preview"
            body = {"model": model, "messages": fm, "max_tokens": mt, "stream": True}
        else:
            url = f"{ep}/openai/deployments/{model}/chat/completions?api-version=2024-12-01-preview"
            body = {"messages": fm, "max_tokens": mt, "stream": True}
        return url, {"api-key": ak, "Content-Type": "application/json"}, body

    if provider == "digitalocean":
        key = cfg.get("do_api_key") or cfg.get("api_key") or cfg.get("llm_api_key", "")
        if not key: raise LLMStreamError(provider, "No DigitalOcean API key configured")
        return ("https://inference.do-ai.run/v1/chat/completions",
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                {"model": model, "messages": _openai_msgs(sp, msgs), "max_completion_tokens": max(256, mt), "stream": True})

    if provider == "google":
        gk = cfg.get("google_api_key", "")
        if not gk: raise LLMStreamError(provider, "No Google API key configured")
        contents = [{"role": "model" if m["role"] == "assistant" else "user",
                      "parts": [{"text": m["content"]}]} for m in msgs]
        body = {"contents": contents}
        if sp: body["system_instruction"] = {"parts": [{"text": sp}]}
        return (f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={gk}",
                {"Content-Type": "application/json"}, body)

    if provider == "puter":
        key = cfg.get("puter_api_key") or cfg.get("puter_auth_token") or cfg.get("llm_api_key", "")
        if not key: raise LLMStreamError(provider, "No Puter auth token configured")
        return ("https://api.puter.com/puterai/openai/v1/chat/completions",
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                {"model": model, "messages": _openai_msgs(sp, msgs), "max_tokens": mt, "stream": True})

    raise LLMStreamError(provider, f"Unknown provider: {provider}")


def _build_bedrock_request(model: str, messages: list[dict],
                           system_prompt: Optional[str], cfg: dict
                           ) -> tuple[str, dict[str, str], bytes]:
    """Build Bedrock converse-stream request. Returns (url, signed_headers, payload_bytes)."""
    ak, sk = cfg.get("aws_access_key", ""), cfg.get("aws_secret_key", "")
    region = cfg.get("aws_region", "us-east-1")
    if not ak or not sk: raise LLMStreamError("bedrock", "AWS credentials not configured")
    model_id = _get_bedrock_model_id(model)
    if not model_id: raise LLMStreamError("bedrock", f"Unknown Bedrock model: {model}")

    sp, msgs = _prepare_messages(messages, system_prompt)
    conv_msgs = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in msgs]
    body: dict[str, Any] = {"modelId": model_id, "messages": conv_msgs,
                            "inferenceConfig": {"maxTokens": cfg.get("max_tokens", 2048),
                                                "temperature": cfg.get("temperature", 1.0)}}
    if sp: body["system"] = [{"text": sp}]

    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{quote(model_id, safe='')}/converse-stream"
    payload = json.dumps(body).encode()
    return url, _aws_sign_v4("POST", url, payload, ak, sk, region, "bedrock"), payload


# ── Bedrock frame iterator (shared by sync/async) ────────────────────────

def _extract_bedrock_tokens(buf: bytearray) -> Generator[str, None, bytearray]:
    """Consume complete frames from buf, yield text tokens, return remainder."""
    while len(buf) >= 12:
        total_len = struct.unpack_from(">I", buf, 0)[0]
        if len(buf) < total_len:
            break
        frame = bytes(buf[:total_len])
        del buf[:total_len]
        for event_type, payload_dict in _parse_event_stream(frame):
            if event_type == "contentBlockDelta":
                text = payload_dict.get("delta", {}).get("text", "")
                if text:
                    yield text
    return buf


# ── Sync API (requests) ──────────────────────────────────────────────────

def stream_chat(provider: str, model: str, messages: list[dict],
                system_prompt: Optional[str] = None,
                config: Optional[dict] = None) -> Generator[str, None, None]:
    """Sync generator yielding text tokens from an LLM provider.

    Uses ``requests`` for HTTP. Suitable for threaded callers (gnome-speaks).
    Raises LLMStreamError on config errors, requests.HTTPError on HTTP failures.
    """
    import requests
    c = _cfg(config)
    model = resolve_model(model, provider)

    if provider == "bedrock":
        url, headers, payload = _build_bedrock_request(model, messages, system_prompt, c)
        resp = requests.post(url, data=payload, headers=headers, timeout=60, stream=True)
        resp.raise_for_status()
        try:
            buf = bytearray()
            for chunk in resp.iter_content(chunk_size=4096):
                buf.extend(chunk)
                # buf is mutated in-place by _extract_bedrock_tokens
                for token in _extract_bedrock_tokens(buf):
                    yield token
        finally:
            resp.close()
        return

    url, headers, body = _build_request(provider, model, messages, system_prompt, c)
    resp = requests.post(url, headers=headers, json=body, timeout=60, stream=True)
    resp.raise_for_status()
    try:
        for line in resp.iter_lines(decode_unicode=True):
            token = _parse_sse_line(line, provider)
            if token:
                yield token
    finally:
        resp.close()


# ── Async API (httpx) ────────────────────────────────────────────────────

async def astream_chat(provider: str, model: str, messages: list[dict],
                       system_prompt: Optional[str] = None,
                       config: Optional[dict] = None,
                       client=None) -> AsyncGenerator[str, None]:
    """Async generator yielding text tokens from an LLM provider.

    Uses ``httpx`` for HTTP. Suitable for asyncio callers (CCA MCP server).
    Pass an existing ``httpx.AsyncClient`` via *client* to reuse connection
    pools; if None, a throwaway client is created per call.
    Raises LLMStreamError on config errors, httpx.HTTPStatusError on HTTP failures.
    """
    import httpx
    c = _cfg(config)
    model = resolve_model(model, provider)

    # Use caller's client if provided, otherwise create a throwaway one
    async def _get_client():
        if client:
            # Wrap in a no-op context manager
            class _Wrapper:
                async def __aenter__(self): return client
                async def __aexit__(self, *a): pass
            return _Wrapper()
        return httpx.AsyncClient(timeout=60.0)

    if provider == "bedrock":
        url, headers, payload = _build_bedrock_request(model, messages, system_prompt, c)
        async with await _get_client() as c_http:
            async with c_http.stream("POST", url, content=payload, headers=headers) as resp:
                resp.raise_for_status()
                buf = bytearray()
                async for chunk in resp.aiter_bytes():
                    buf.extend(chunk)
                    # buf is mutated in-place by _extract_bedrock_tokens
                    for token in _extract_bedrock_tokens(buf):
                        yield token
        return

    url, headers, body = _build_request(provider, model, messages, system_prompt, c)
    async with await _get_client() as c_http:
        async with c_http.stream("POST", url, headers=headers, json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                token = _parse_sse_line(line, provider)
                if token:
                    yield token
