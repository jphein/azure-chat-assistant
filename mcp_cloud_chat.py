#!/usr/bin/env python3
"""
Azure Chat Assistant MCP Server — standalone chat assistant powered by Azure AI models.

Uses Azure AI Foundry for LLM (GPT-5.3, grok-3, Llama, etc.) and delegates to the
azure-speech MCP server for TTS/STT when used alongside it, or works as a pure
chat tool on its own.

Refactored for asyncio and httpx for maximum efficiency and concurrency.

Tools:
  - chat:      Send a message to the LLM, get a response (with conversation history)
  - configure: View/change settings dynamically (model, API key, region, etc.)
  - reset:       Clear conversation history
  - clear_cache: Clear the in-memory response cache
  - models:    List available models and test connectivity
  """


import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import sqlite3
from urllib.parse import quote, urlparse
import httpx

from llm_stream import (
    astream_chat,
    resolve_model,
    BEDROCK_MODELS,
    _aws_sign_v4,
    _get_bedrock_model_id,
)

# ── Config ──────────────────────────────────────────────────────────────────

CONFIG_DIR = os.path.expanduser("~/.config/cloud-chat-assistant")
_OLD_CONFIG_DIR = os.path.expanduser("~/.config/azure-chat-assistant")

# Migrate from old location if needed
if os.path.exists(_OLD_CONFIG_DIR) and not os.path.exists(CONFIG_DIR):
    os.rename(_OLD_CONFIG_DIR, CONFIG_DIR)

CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
DB_PATH = os.path.join(CONFIG_DIR, "sessions.db")

DEFAULTS = {
    "api_key": "",
    "endpoint": "",
    "deployment": "gpt-5.3-chat",
    "model": "gpt-5.3-chat-2026-03-03",
    "model_type": "deployed",       # "deployed", "serverless", "codex", "bedrock", "google", "digitalocean", "puter"
    "reasoning_effort": "medium",    # for codex/reasoning models: low, medium, high, xhigh
    "max_completion_tokens": 2048,
    "temperature": 1.0,
    "system_prompt": "You are a helpful chat assistant. Keep responses concise and conversational.",
    "conversation_max_turns": 50,    # max history turns before auto-trimming
    "voice": "",                     # default TTS voice (empty = use speech config)
    "default_models": ["gpt-5.3-chat", "o4-mini", "grok-3", "DeepSeek-R1", "claude-sonnet-4.6", "gemini-3.1-pro-preview"],  # models for multi_chat when none specified
    "multi_chat_timeout": 15,        # per-model timeout in seconds for multi_chat
    "google_api_key": "",
    "google_project": "",
    "google_region": "global",
    # AWS Bedrock
    "aws_access_key": "",
    "aws_secret_key": "",
    "aws_region": "us-east-1",
    # DigitalOcean
    "do_api_key": "",
    # Puter
    "puter_api_key": "",
    # Multi-region Azure: additional endpoints beyond the primary
    "azure_endpoints": [],  # [{"endpoint": "https://...", "api_key": "..."}]
}

# ── Model Catalogs ──────────────────────────────────────────────────────────

# Azure AI Foundry models (verified 2026-03)
# Deployed models require explicit deployment in Azure portal
AZURE_DEPLOYED = [
    # OpenAI reasoning
    "o1", "o4-mini", "o1-mini", "o1-preview", "o3-mini",
    # OpenAI GPT
    "gpt-5.3-chat", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-35-turbo",
    # Embeddings
    "text-embedding-3-large", "text-embedding-3-small",
]
# Serverless models are always available (pay-per-use)
AZURE_SERVERLESS = [
    # xAI
    "grok-3", "grok-3-mini",
    # DeepSeek
    "DeepSeek-R1",
    # Meta Llama
    "Meta-Llama-3.1-405B-Instruct", "Meta-Llama-3.1-8B-Instruct",
    "Llama-3.2-11B-Vision-Instruct", "Llama-3.2-90B-Vision-Instruct",
    "Llama-3.3-70B-Instruct",
    "Llama-4-Scout-17B-16E-Instruct",
    # Microsoft
    "Phi-4",
    # Cohere
    "Cohere-command-r-plus-08-2024", "Cohere-command-r-08-2024",
    # Mistral
    "Codestral-2501", "Ministral-3B",
]

# Google Vertex AI models
GOOGLE_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-3.1-pro-preview"]

# DigitalOcean Serverless Inference models (via inference.do-ai.run)
DO_MODELS = [
    # Anthropic
    "anthropic-claude-opus-4.6", "anthropic-claude-4.6-sonnet",
    "anthropic-claude-opus-4.5", "anthropic-claude-4.5-sonnet", "anthropic-claude-haiku-4.5",
    "anthropic-claude-4.1-opus", "anthropic-claude-opus-4", "anthropic-claude-sonnet-4",
    # OpenAI
    "openai-gpt-5.4", "openai-gpt-5.3-codex", "openai-gpt-5.2", "openai-gpt-5.2-pro",
    "openai-gpt-5", "openai-gpt-5-mini", "openai-gpt-5-nano",
    "openai-gpt-4.1", "openai-gpt-4o", "openai-gpt-4o-mini",
    "openai-o3", "openai-o3-mini", "openai-o1",
    "openai-gpt-oss-120b", "openai-gpt-oss-20b",
    # Meta
    "llama3.3-70b-instruct", "llama3-8b-instruct",
    # Other
    "deepseek-r1-distill-llama-70b", "mistral-nemo-instruct-2407",
    "nvidia-nemotron-3-super-120b", "alibaba-qwen3-32b",
    "minimax-m2.5", "kimi-k2.5", "glm-5",
]
DO_ENDPOINT = "https://inference.do-ai.run/v1"

# Puter AI (OpenAI-compatible, free with auth token)
PUTER_MODELS = [
    "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929",
    "gpt-5.4-2026-03-05", "gpt-5.2-chat-latest", "o3", "o3-pro", "o4-mini",
    "gpt-4o", "gpt-4o-mini",
    "deepseek-chat", "deepseek-reasoner",
    "grok-4", "grok-4-fast", "grok-3",
    "gemini-2.5-pro", "gemini-2.5-flash",
    "mistral-large-latest",
]
PUTER_ENDPOINT = "https://api.puter.com/puterai/openai/v1"

# BEDROCK_MODELS imported from llm_stream (single source of truth)

# ── CLI Helpers ────────────────────────────────────────────────────────────

def _cli_available(name):
    """Check if a CLI tool is available."""
    return shutil.which(name) is not None

async def _run_cli(cmd, timeout=30):
    """Run CLI command async and return stdout or None on error."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode == 0:
            return stdout.decode().strip()
        return None
    except Exception:
        return None

async def _az_list_deployable_models(endpoint=None):
    """Use Azure CLI to list deployable OpenAI models for a given endpoint."""
    if not _cli_available("az"):
        return None
    endpoint = endpoint or CONFIG.get("endpoint", "")
    if not endpoint:
        return None
    resource_name = _endpoint_resource_name(endpoint)
    if not resource_name:
        return None
    # Get resource group (try env var first, then query Azure)
    rg = os.environ.get("AZURE_RESOURCE_GROUP", "")
    if not rg:
        # Try to find the resource group
        out = await _run_cli(["az", "resource", "list", "--name", resource_name, "--query", "[0].resourceGroup", "-o", "tsv"])
        if out:
            rg = out.strip()
    if not rg:
        return None
    # List models
    out = await _run_cli(["az", "cognitiveservices", "account", "list-models", "--name", resource_name, "--resource-group", rg, "-o", "json"])
    if not out:
        return None
    try:
        models = json.loads(out)
        # Filter to chat completion models, dedupe by name
        seen = set()
        deployable = []
        for m in models:
            name = m.get("name", "")
            if name in seen:
                continue
            seen.add(name)
            caps = m.get("capabilities", {})
            # Include chat/completion models, exclude embedding-only and rerank
            if "embed" in name.lower() or "rerank" in name.lower():
                continue
            if caps.get("chatCompletion") or caps.get("completion") or any(k in name.lower() for k in ["gpt", "llama", "phi", "claude", "mistral", "deepseek", "grok", "cohere", "gemini", "qwen", "jamba", "kimi"]) or name.startswith("o"):
                deployable.append(name)
        return deployable
    except Exception:
        return None

async def _az_list_deployed(endpoint=None):
    """Use Azure CLI to list currently deployed models for a given endpoint."""
    if not _cli_available("az"):
        return None
    endpoint = endpoint or CONFIG.get("endpoint", "")
    if not endpoint:
        return None
    resource_name = _endpoint_resource_name(endpoint)
    if not resource_name:
        return None
    rg = os.environ.get("AZURE_RESOURCE_GROUP", "")
    if not rg:
        out = await _run_cli(["az", "resource", "list", "--name", resource_name, "--query", "[0].resourceGroup", "-o", "tsv"])
        if out:
            rg = out.strip()
    if not rg:
        return None
    out = await _run_cli(["az", "cognitiveservices", "account", "deployment", "list", "--name", resource_name, "--resource-group", rg, "-o", "json"])
    if not out:
        return None
    try:
        deployments = json.loads(out)
        return [d.get("name", "") for d in deployments if d.get("name")]
    except Exception:
        return None

async def _aws_list_bedrock_models():
    """Use AWS CLI to list available Bedrock models."""
    if not _cli_available("aws"):
        return None
    region = CONFIG.get("aws_region", "us-east-1")
    out = await _run_cli(["aws", "bedrock", "list-foundation-models", "--region", region, "--output", "json"])
    if not out:
        return None
    try:
        data = json.loads(out)
        models = []
        for m in data.get("modelSummaries", []):
            model_id = m.get("modelId", "")
            # Include chat/text models, skip embedding-only
            if m.get("outputModalities") and "TEXT" in m.get("outputModalities", []):
                models.append(model_id)
        return models
    except Exception:
        return None

async def _gcloud_list_models():
    """Use gcloud to list Vertex AI models."""
    if not _cli_available("gcloud"):
        return None
    region = CONFIG.get("google_region", "us-east4")
    if region == "global":
        region = "us-east4"  # gcloud needs a specific region
    out = await _run_cli(["gcloud", "ai", "models", "list", f"--region={region}", "--format=json"])
    if not out:
        return None
    try:
        models = json.loads(out)
        return [m.get("displayName", m.get("name", "")) for m in models]
    except Exception:
        return None

CONFIG = {}
_conversation_history = []          # list of {"role": ..., "content": ...}
_cache = {}                         # simple in-memory cache for static responses
_model_status = {}                  # track last error/status per model
_stdout_lock = asyncio.Lock()
CURRENT_SESSION = "default"

# ── DB management ───────────────────────────────────────────────────────────

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                name TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_name) REFERENCES sessions(name) ON DELETE CASCADE
            )
        """)
        # Index for fast history lookups
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_name)")
        # Ensure default session exists
        conn.execute("INSERT OR IGNORE INTO sessions (name) VALUES ('default')")

def get_history(session_name):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                "SELECT role, content FROM messages WHERE session_name = ? ORDER BY id ASC",
                (session_name,)
            )
            return [{"role": r, "content": c} for r, c in cursor.fetchall()]
    except Exception:
        return []

def add_message(session_name, role, content):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO messages (session_name, role, content) VALUES (?, ?, ?)",
                (session_name, role, content)
            )
    except Exception:
        pass

def clear_session(session_name):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM messages WHERE session_name = ?", (session_name,))
    except Exception:
        pass

init_db()

# ── Config management ───────────────────────────────────────────────────────

ENV_MAP = {
    "api_key":        "AZURE_AI_API_KEY",
    "endpoint":       "AZURE_AI_ENDPOINT",
    "google_api_key": "GOOGLE_API_KEY",
    "google_project": "GOOGLE_PROJECT",
    "google_region":  "GOOGLE_REGION",
    "aws_access_key": "AWS_ACCESS_KEY_ID",
    "aws_secret_key": "AWS_SECRET_ACCESS_KEY",
    "aws_region":     "AWS_DEFAULT_REGION",
    "do_api_key":     "DIGITALOCEAN_API_KEY",
    "puter_api_key":  "PUTER_API_KEY",
}


def load_config():
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                disk = json.load(f)
            if "azure_ai" in disk:
                for k, v in disk["azure_ai"].items():
                    cfg[k] = v
            else:
                cfg.update(disk)
        except Exception:
            pass
    # Env vars take precedence over config file
    for cfg_key, env_key in ENV_MAP.items():
        val = os.environ.get(env_key, "")
        if val:
            cfg[cfg_key] = val
    # Fall back to ~/.aws/credentials if AWS keys still empty
    if not cfg.get("aws_access_key") or not cfg.get("aws_secret_key"):
        aws_creds = os.path.join(os.path.expanduser("~"), ".aws", "credentials")
        if os.path.exists(aws_creds):
            try:
                import configparser
                cp = configparser.ConfigParser()
                cp.read(str(aws_creds))
                profile = os.environ.get("AWS_PROFILE", "default")
                if cp.has_section(profile):
                    ak = cp.get(profile, "aws_access_key_id", fallback="")
                    sk = cp.get(profile, "aws_secret_access_key", fallback="")
                    if ak and sk:
                        cfg["aws_access_key"] = ak
                        cfg["aws_secret_key"] = sk
                        rg = cp.get(profile, "region", fallback="")
                        if rg and not cfg.get("aws_region"):
                            cfg["aws_region"] = rg
            except Exception:
                pass
    return cfg


def save_config():
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    disk = {}
    for k, v in CONFIG.items():
        if k.startswith("_"):
            continue
        if k in DEFAULTS and CONFIG[k] == DEFAULTS[k]:
            continue
        disk[k] = v
    for k in ("api_key", "endpoint", "deployment", "model", "model_type", "reasoning_effort", "google_api_key", "google_project", "google_region", "aws_access_key", "aws_secret_key", "aws_region", "do_api_key", "puter_api_key"):
        disk[k] = CONFIG.get(k, "")
    # Always persist azure_endpoints (even if empty, for clarity)
    azure_eps = CONFIG.get("azure_endpoints", [])
    if azure_eps:
        disk["azure_endpoints"] = azure_eps
    with open(CONFIG_PATH, "w") as f:
        json.dump(disk, f, indent=4)


CONFIG = load_config()

# Multi-region deployment routing: deployment_name -> {endpoint, api_key}
_deployment_map = {}


def _get_all_azure_endpoints():
    """Return list of {endpoint, api_key} dicts for all configured Azure resources."""
    endpoints = []
    primary_ep = CONFIG.get("endpoint", "")
    primary_key = CONFIG.get("api_key", "")
    if primary_ep and primary_key:
        endpoints.append({"endpoint": primary_ep, "api_key": primary_key})
    for extra in CONFIG.get("azure_endpoints", []):
        ep = extra.get("endpoint", "").rstrip("/")
        key = extra.get("api_key", "")
        if ep and key and ep != primary_ep:
            endpoints.append({"endpoint": ep, "api_key": key})
    return endpoints


def _endpoint_resource_name(endpoint):
    """Extract short resource name from Azure endpoint URL."""
    try:
        from urllib.parse import urlparse
        host = urlparse(endpoint).hostname
        return host.split(".")[0] if host else endpoint
    except Exception:
        return endpoint


def _google_base_url():
    """Build Vertex AI OpenAI-compatible base URL from project + region config."""
    project = CONFIG.get("google_project", "")
    region = CONFIG.get("google_region", "global")
    if not project:
        return ""
    return f"https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{region}/endpoints/openapi"


# ── AWS SigV4 Signing ───────────────────────────────────────────────────────
# _aws_sign_v4 and _get_bedrock_model_id imported from llm_stream.
# Thin wrapper preserves the original call signature for call_bedrock().

def _aws_sign(method, url, headers, payload, region, service="bedrock"):
    """Sign an AWS request using Signature Version 4 (delegates to llm_stream)."""
    access_key = CONFIG.get("aws_access_key", "")
    secret_key = CONFIG.get("aws_secret_key", "")
    if not access_key or not secret_key:
        return headers
    payload_bytes = payload.encode() if isinstance(payload, str) else payload
    signed = _aws_sign_v4(method, url, payload_bytes, access_key, secret_key, region, service)
    return {**headers, **signed}


async def call_bedrock(client: httpx.AsyncClient, messages, progress_token=None, model_name="claude-opus-4.5"):
    """Call AWS Bedrock using the Converse API. Returns (response_text, usage_dict, latency_ms)."""
    region = CONFIG.get("aws_region", "us-east-1")
    model_id = _get_bedrock_model_id(model_name)

    if not model_id:
        return f"Error: Unknown Bedrock model '{model_name}'", {}, 0
    if not CONFIG.get("aws_access_key") or not CONFIG.get("aws_secret_key"):
        return "Error: AWS credentials not configured. Set aws_access_key and aws_secret_key.", {}, 0

    system_content = []
    converse_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_content.append({"text": msg["content"]})
        else:
            converse_messages.append({"role": msg["role"], "content": [{"text": msg["content"]}]})

    body = {
        "modelId": model_id,
        "messages": converse_messages,
        "inferenceConfig": {"maxTokens": CONFIG.get("max_completion_tokens", 2048), "temperature": CONFIG.get("temperature", 1.0)}
    }
    if system_content:
        body["system"] = system_content

    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{quote(model_id, safe='')}/converse"
    payload = json.dumps(body)
    headers = _aws_sign("POST", url, {}, payload, region, "bedrock")

    if progress_token:
        await _send_progress(progress_token, 0.1, f"[{model_name}] Thinking...")

    t0 = time.perf_counter()
    try:
        resp = await client.post(url, content=payload, headers=headers, timeout=60.0)
        latency = (time.perf_counter() - t0) * 1000

        if resp.status_code == 200:
            _model_status[model_name] = "OK"
            data = resp.json()
            # Fast path: extract text directly
            try:
                blocks = data["output"]["message"]["content"]
                response_text = "".join(b["text"] for b in blocks if "text" in b)
            except (KeyError, TypeError):
                response_text = ""
            return response_text, data.get("usage", {}), latency
        elif resp.status_code == 429:
            _model_status[model_name] = "Rate Limited"
            return f"**[{model_name}]** Rate limit hit.", {}, 0
        else:
            _model_status[model_name] = f"Error {resp.status_code}"
            return f"Error {resp.status_code}: {resp.text[:300]}", {}, 0
    except Exception as e:
        return f"Error: {e}", {}, 0


# ── Streaming Bedrock via llm_stream ───────────────────────────────────────

async def call_bedrock_stream(client, messages, model_name="claude-opus-4.6"):
    """Streaming Bedrock call via llm_stream. Yields text tokens."""
    config = {k: CONFIG.get(k) for k in ("aws_access_key", "aws_secret_key", "aws_region")}
    config["max_tokens"] = CONFIG.get("max_completion_tokens", 2048)
    config["temperature"] = CONFIG.get("temperature", 1.0)
    async for token in astream_chat("bedrock", model_name, messages, config=config):
        yield token


# ── Codex Responses API ────────────────────────────────────────────────────

async def call_codex(client: httpx.AsyncClient, messages, progress_token, deployment, model):
    """Call Azure OpenAI model via the Responses API with streaming.

    Uses SSE streaming so agents get incremental output instead of waiting
    for the entire reasoning + generation to complete (which can take 60-120s
    for gpt-5.4-pro). Returns (response_text, usage_dict, latency_ms).
    """
    # Resolve endpoint via deployment map (multi-region) or primary
    ep_info = _deployment_map.get(deployment)
    ep = ep_info["endpoint"] if ep_info else CONFIG.get("endpoint", "")
    key = ep_info["api_key"] if ep_info else CONFIG.get("api_key", "")

    if not key or not ep:
        return "Error: api_key and endpoint required for codex models.", {}, 0

    # Responses API uses /openai/v1/responses (model in body)
    url = f"{ep}/openai/v1/responses"

    # Flatten messages into Responses API input format
    input_items = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            input_items.append({"role": "developer", "content": msg["content"]})
        else:
            input_items.append({"role": role, "content": msg["content"]})

    reasoning_effort = CONFIG.get("reasoning_effort", "medium")
    body = {
        "model": deployment,
        "input": input_items,
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": CONFIG.get("max_completion_tokens", 2048),
        "stream": True,
    }

    headers = {
        "api-key": key,
        "Content-Type": "application/json",
    }

    if progress_token:
        await _send_progress(progress_token, 0.1, f"[{model}] Thinking (codex streaming)...")

    t0 = time.perf_counter()
    response_text = ""
    usage = {}

    try:
        async with client.stream("POST", url, json=body, headers=headers, timeout=300.0) as resp:
            if resp.status_code == 404:
                # Try other endpoints
                fallback = await _try_endpoints_codex(client, deployment, model, messages, progress_token)
                if fallback:
                    resp_obj, _ = fallback
                    latency = (time.perf_counter() - t0) * 1000
                    data = resp_obj.json()
                    for item in data.get("output", []):
                        if item.get("type") == "message":
                            for block in item.get("content", []):
                                if block.get("type") == "output_text":
                                    response_text += block.get("text", "")
                    return response_text.strip() or "Error: No response from model.", data.get("usage", {}), latency
                return f"Error 404: Deployment '{deployment}' not found.", {}, 0

            if resp.status_code == 429:
                _model_status[model] = "Rate Limited"
                return f"**[{model}]** Rate limit hit.", {}, 0

            if resp.status_code != 200:
                await resp.aread()
                _model_status[model] = f"Error {resp.status_code}"
                return f"Error {resp.status_code}: {resp.text[:500]}", {}, 0

            # Parse SSE stream
            char_count = 0
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                # Accumulate output text deltas
                if event_type == "response.output_text.delta":
                    delta = event.get("delta", "")
                    if delta:
                        response_text += delta
                        char_count += len(delta)
                        # Progress update every ~200 chars
                        if progress_token and char_count % 200 < len(delta):
                            elapsed = time.perf_counter() - t0
                            await _send_progress(progress_token, min(0.9, 0.2 + char_count / 5000),
                                                 f"[{model}] Generating... ({elapsed:.0f}s, {char_count} chars)")

                # Capture usage from completed response
                elif event_type == "response.completed":
                    resp_data = event.get("response", {})
                    usage = resp_data.get("usage", {})

        latency = (time.perf_counter() - t0) * 1000
        _model_status[model] = "OK"
        if progress_token:
            await _send_progress(progress_token, 1.0, f"[{model}] Done ({latency:.0f}ms)")
        return response_text.strip() if response_text else "Error: No response from model.", usage, latency

    except Exception as e:
        return f"Error ({type(e).__name__}): {e}" if str(e) else f"Error: {type(e).__name__} (no details)", {}, 0


# ── Multi-endpoint fallback ────────────────────────────────────────────────

async def _try_endpoints_deployed(client, deployment, messages, body_builder, timeout=120.0):
    """Try a deployed model across all Azure endpoints. Returns (resp, endpoint_info) or None."""
    endpoints = _get_all_azure_endpoints()
    for ep_info in endpoints:
        ep, key = ep_info["endpoint"], ep_info["api_key"]
        url = f"{ep}/openai/deployments/{deployment}/chat/completions?api-version=2024-12-01-preview"
        headers = {"api-key": key, "Content-Type": "application/json"}
        body = body_builder()
        try:
            resp = await client.post(url, json=body, headers=headers, timeout=timeout)
            if resp.status_code != 404:
                # Cache this endpoint for future calls
                _deployment_map[deployment] = ep_info
                return resp, ep_info
        except Exception:
            continue
    return None

async def _try_endpoints_serverless(client, model, messages, body_builder, timeout=120.0):
    """Try a serverless model across all Azure endpoints. Returns (resp, endpoint_info) or None."""
    endpoints = _get_all_azure_endpoints()
    version = "2024-12-01-preview" if any(p in model.lower() for p in ("o1", "o4")) else "2024-05-01-preview"
    for ep_info in endpoints:
        ep, key = ep_info["endpoint"], ep_info["api_key"]
        url = f"{ep}/models/chat/completions?api-version={version}"
        headers = {"api-key": key, "Content-Type": "application/json"}
        body = body_builder()
        try:
            resp = await client.post(url, json=body, headers=headers, timeout=timeout)
            if resp.status_code != 404 and resp.status_code != 400:
                return resp, ep_info
        except Exception:
            continue
    return None

async def _try_endpoints_codex(client, deployment, model, messages, progress_token=None):
    """Try a codex/responses-API model across all Azure endpoints."""
    endpoints = _get_all_azure_endpoints()
    input_items = []
    for msg in messages:
        role = "developer" if msg["role"] == "system" else msg["role"]
        input_items.append({"role": role, "content": msg["content"]})

    reasoning_effort = CONFIG.get("reasoning_effort", "medium")
    body = {
        "model": deployment,
        "input": input_items,
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": CONFIG.get("max_completion_tokens", 2048),
    }

    for ep_info in endpoints:
        ep, key = ep_info["endpoint"], ep_info["api_key"]
        url = f"{ep}/openai/v1/responses"
        headers = {"api-key": key, "Content-Type": "application/json"}
        try:
            resp = await client.post(url, json=body, headers=headers, timeout=120.0)
            if resp.status_code != 404:
                _deployment_map[deployment] = ep_info
                return resp, ep_info
        except Exception:
            continue
    return None


# ── LLM call ────────────────────────────────────────────────────────────────

async def call_llm(client: httpx.AsyncClient, messages, progress_token=None, model_override=None, model_type_override=None):
    """Call Azure AI model with streaming, using a producer-consumer queue. Returns (response_text, usage_dict, latency_ms)."""
    api_key = CONFIG.get("api_key", "")
    endpoint = CONFIG.get("endpoint", "")

    deployment = model_override if model_override else CONFIG.get("deployment", "")
    model = model_override if model_override else CONFIG.get("model", deployment)
    model_type = model_type_override if model_type_override else CONFIG.get("model_type", "deployed")

    max_tokens = CONFIG.get("max_completion_tokens", 2048)
    temperature = CONFIG.get("temperature", 1.0)

    # Auto-detect Responses API models even if model_type is set to "deployed"
    _responses_api_models = ("codex", "deep-research", "gpt-5.4-pro", "gpt-5.4")
    if model_type == "deployed" and any(p in deployment.lower() for p in _responses_api_models):
        model_type = "codex"

    # Route to Bedrock if model type is bedrock (use deployment for model name)
    if model_type == "bedrock":
        return await call_bedrock(client, messages, progress_token, deployment)

    # Route to Codex Responses API
    if model_type == "codex":
        return await call_codex(client, messages, progress_token, deployment, model)

    # Route to Puter OpenAI-compatible endpoint
    if model_type == "puter":
        puter_key = CONFIG.get("puter_api_key", "")
        if not puter_key:
            return "Error: No Puter API key configured. Use configure tool to set puter_api_key.", {}, 0
        url = f"{PUTER_ENDPOINT}/chat/completions"
        body = {
            "messages": messages,
            "model": model,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {
            "Authorization": f"Bearer {puter_key}",
            "Content-Type": "application/json",
        }
    # Route to DigitalOcean Serverless Inference
    elif model_type == "digitalocean":
        do_key = CONFIG.get("do_api_key", "")
        if not do_key:
            return "Error: No DigitalOcean API key configured. Use configure tool to set do_api_key.", {}, 0
        url = f"{DO_ENDPOINT}/chat/completions"
        body = {
            "messages": messages,
            "model": model,
            "max_completion_tokens": max(256, max_tokens),  # DO enforces 256 minimum
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {
            "Authorization": f"Bearer {do_key}",
            "Content-Type": "application/json",
        }
    else:
        # Reasoning models (o1, o3, o4) use "developer" role instead of "system"
        is_reasoning = any(deployment.startswith(p) or model.startswith(p) for p in ("o1", "o3", "o4"))
        if is_reasoning:
            messages = [
                {"role": "developer" if m["role"] == "system" else m["role"], "content": m["content"]}
                for m in messages
            ]

        if model_type == "google":
            google_key = CONFIG.get("google_api_key", "")
            if not google_key:
                return "Error: No Google API key configured. Use configure tool to set google_api_key.", {}, 0
            google_ep = _google_base_url()
            if not google_ep:
                return "Error: No Google project configured. Use configure tool to set google_project.", {}, 0
            url = f"{google_ep}/chat/completions"
            body = {
                "messages": messages,
                "model": f"google/{model}",
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            headers = {
                "x-goog-api-key": google_key,
                "Content-Type": "application/json",
            }
        elif model_type == "deployed":
            # Multi-region: look up deployment map for the correct endpoint
            ep_info = _deployment_map.get(deployment)
            ep = ep_info["endpoint"] if ep_info else endpoint
            key = ep_info["api_key"] if ep_info else api_key
            if not key:
                return "Error: No API key configured. Use configure tool to set api_key.", {}, 0
            if not ep:
                return "Error: No endpoint configured.", {}, 0
            url = f"{ep}/openai/deployments/{deployment}/chat/completions?api-version=2024-12-01-preview"
            body = {
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            headers = {
                "api-key": key,
                "Content-Type": "application/json",
            }
        else:  # serverless
            if not api_key:
                return "Error: No API key configured. Use configure tool to set api_key.", {}, 0
            if not endpoint:
                return "Error: No endpoint configured.", {}, 0

            # o1 and o4 models on serverless endpoints require a newer api-version
            version = "2024-12-01-preview" if any(p in model.lower() for p in ("o1", "o4")) else "2024-05-01-preview"
            url = f"{endpoint}/models/chat/completions?api-version={version}"
            body = {
                "messages": messages,
                "model": model,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            headers = {
                "api-key": api_key,
                "Content-Type": "application/json",
            }

    is_reasoning = model_type != "digitalocean" and any(deployment.startswith(p) or model.startswith(p) for p in ("o1", "o3", "o4"))

    if is_reasoning:
        body["reasoning_effort"] = CONFIG.get("reasoning_effort", "high")
        body["max_completion_tokens"] = body.pop("max_tokens", body.get("max_completion_tokens", max_tokens))
        body.pop("stream", None)
        body.pop("stream_options", None)
    elif temperature != 1.0:
        body["temperature"] = temperature

    if progress_token:
        await _send_progress(progress_token, 0.1, f"[{model}] Thinking...")

    t0 = time.perf_counter()

    # Non-streaming path for reasoning models (o1, o3)
    if is_reasoning:
        try:
            resp = await client.post(url, json=body, headers=headers, timeout=120.0)
            latency = (time.perf_counter() - t0) * 1000

            # Endpoint fallback on 404 — model may be on a secondary Azure endpoint
            if resp.status_code == 404 and model_type in ("deployed", "serverless"):
                fb_fn = _try_endpoints_deployed if model_type == "deployed" else _try_endpoints_serverless
                fb = await fb_fn(client, deployment, messages, lambda: body, timeout=120.0)
                if fb:
                    resp, _ = fb
                    latency = (time.perf_counter() - t0) * 1000

            if resp.status_code == 429:
                _model_status[model] = "Rate Limited"
                return f"**[{model}]** is currently resting (Rate Limit hit).", {}, 0
            if resp.status_code != 200:
                _model_status[model] = f"Error {resp.status_code}"
                return f"Error {resp.status_code}: {resp.text[:500]}", {}, latency
            data = resp.json()
            _model_status[model] = "OK"
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            usage = data.get("usage", {})
            if progress_token:
                await _send_progress(progress_token, 1.0, f"[{model}] Done ({latency:.0f}ms)")
            return text if text else "Error: No response from model.", usage, latency
        except Exception as e:
            return f"Error: {e}", {}, 0

    queue = asyncio.Queue(maxsize=100)
    SENTINEL = object()
    
    # State shared between producer and consumer
    state = {
        "full_text": "",
        "usage": {},
        "ttft": 0,
        "error": None,
        "status_code": 200
    }

    async def producer():
        try:
            async with client.stream("POST", url, json=body, headers=headers, timeout=60.0) as resp:
                state["status_code"] = resp.status_code
                if resp.status_code == 200:
                    _model_status[model] = "OK"
                elif resp.status_code == 429:
                    _model_status[model] = "Rate Limited"
                    state["error"] = f"**[{model}]** is currently resting (Rate Limit hit). Please try another model or wait a moment."
                    await queue.put(SENTINEL)
                    return
                elif resp.status_code == 400:
                    body_text = await resp.aread()
                    _model_status[model] = "Error 400"
                    state["error"] = f"**[{model}]** Error 400: {body_text.decode()[:500]}"
                    await queue.put(SENTINEL)
                    return
                elif resp.status_code == 404 and model_type in ("deployed", "serverless"):
                    # Model not on this endpoint — mark for fallback retry
                    state["error"] = "_FALLBACK_404_"
                    await queue.put(SENTINEL)
                    return
                else:
                    body_text = await resp.aread()
                    _model_status[model] = f"Error {resp.status_code}"
                    state["error"] = f"Error {resp.status_code}: {body_text.decode()[:300]}"
                    await queue.put(SENTINEL)
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                        
                    await queue.put(chunk)
        except Exception as e:
            state["error"] = f"Error: {e}"
        finally:
            await queue.put(SENTINEL)

    async def consumer():
        last_progress = 0
        while True:
            chunk = await queue.get()
            if chunk is SENTINEL:
                queue.task_done()
                break
                
            if chunk.get("usage"):
                state["usage"] = chunk["usage"]

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if not state["ttft"]:
                        state["ttft"] = (time.perf_counter() - t0) * 1000
                    state["full_text"] += content

                    if progress_token:
                        now = time.perf_counter()
                        if now - last_progress > 0.3:
                            preview = state["full_text"][-80:] if len(state["full_text"]) > 80 else state["full_text"]
                            # Beautiful concurrent streaming: prefix with model name
                            asyncio.create_task(_send_progress(progress_token, 0.5, f"[{model}] {preview}"))
                            last_progress = now
            queue.task_done()

    # Run producer and consumer concurrently
    await asyncio.gather(producer(), consumer())

    latency = (time.perf_counter() - t0) * 1000

    # Streaming 404 fallback — retry across all Azure endpoints
    if state["error"] == "_FALLBACK_404_":
        fb_fn = _try_endpoints_deployed if model_type == "deployed" else _try_endpoints_serverless
        fb = await fb_fn(client, deployment if model_type == "deployed" else model, messages, lambda: body)
        if fb:
            resp, ep_info = fb
            latency = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                data = resp.json()
                _model_status[model] = "OK"
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                usage = data.get("usage", {})
                return text if text else "Error: No response from model.", usage, latency
            else:
                return f"Error {resp.status_code}: {resp.text[:500]}", {}, latency
        return f"Error 404: Model '{deployment}' not found on any configured Azure endpoint.", {}, 0

    if state["error"]:
        return state["error"], {}, 0

    if not state["full_text"]:
        return "Error: No response from model.", state["usage"], latency

    state["usage"]["_ttft_ms"] = round(state["ttft"])
    return state["full_text"], state["usage"], latency

# ── Conversation management ─────────────────────────────────────────────────

async def chat(client, user_message, progress_token=None, model_override=None, model_type_override=None, cached_history=None):
    """Send a message, get a response, maintain history, use cache, and fallback on rate limits."""
    global _cache, CURRENT_SESSION

    # Load history from DB (or use cached if provided by multi_chat)
    history = cached_history if cached_history is not None else get_history(CURRENT_SESSION)

    messages = []
    sys_prompt = CONFIG.get("system_prompt", "")
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # Generate cache key using hash (avoids serializing entire history)
    model_name = model_override if model_override else CONFIG.get("model", "")
    msg_hash = hashlib.md5(json.dumps(messages, separators=(',', ':')).encode()).hexdigest()
    cache_key = f"{model_name}:{msg_hash}"

    if cache_key in _cache:
        cached_resp, usage, latency = _cache[cache_key]
        if progress_token:
            await _send_progress(progress_token, 1.0, "Cache hit!")
        return f"{cached_resp} (cached)", usage, latency

    response, usage, latency = await call_llm(client, messages, progress_token, model_override, model_type_override)

    # Automatic Fallback Logic for Rate Limits (429)
    if "Rate Limit hit" in response and not model_override:
        fallback_model = "Meta-Llama-3.1-405B-Instruct"
        if progress_token:
            await _send_progress(progress_token, 0.2, f"Primary model resting. Summoning {fallback_model}...")
        
        fb_response, fb_usage, fb_latency = await call_llm(client, messages, progress_token, fallback_model, "serverless")
        
        if not fb_response.startswith("Error") and "Rate Limit hit" not in fb_response:
            response = f"{fb_response}\n\n*(Note: {model_name} was busy; {fallback_model} stepped in to answer.)*"
            usage = fb_usage
            latency += fb_latency
            model_name = fallback_model

    if not response.startswith("Error") and "Rate Limit hit" not in response:
        # Update cache on success
        _cache[cache_key] = (response, usage, latency)
        
        if not model_override:
            # Save to DB
            add_message(CURRENT_SESSION, "user", user_message)
            add_message(CURRENT_SESSION, "assistant", response)

    return response, usage, latency

async def multi_chat(client, user_message, models=None, progress_token=None):
    """Dispatch message to multiple models concurrently and return results as they arrive."""
    global CURRENT_SESSION

    # Fall back to configured defaults if no models specified
    if not models:
        models = CONFIG.get("default_models", DEFAULTS["default_models"])

    n = len(models)
    timeout = CONFIG.get("multi_chat_timeout", DEFAULTS["multi_chat_timeout"])
    tasks = {}
    t0 = time.perf_counter()

    # Progress: smooth time-based ticker that always moves, with jumps on model completion.
    # Creeps toward 90% proportional to elapsed/timeout, so the bar is always alive.
    # Jumps +5% instantly when a model finishes. Final 90→100% on completion.
    ticker_pct = 0
    models_done = 0

    async def _progress_ticker():
        """Background task: advance progress bar every 300ms. Always moves forward."""
        nonlocal ticker_pct
        while ticker_pct < 99:
            await asyncio.sleep(0.2)
            elapsed = time.perf_counter() - t0
            # Minimum +2% per tick so the bar never stalls
            target = ticker_pct + 2
            # Time-based creep toward 99%
            time_pct = min(int((elapsed / timeout) * 99), 99)
            target = max(target, time_pct)
            # Completion-based jump
            done_pct = int((models_done / n) * 99)
            target = max(target, done_pct)
            target = min(target, 99)
            if target > ticker_pct:
                ticker_pct = target
                elapsed_ms = elapsed * 1000
                await _send_progress(progress_token, ticker_pct / 100, f"⏳ {models_done}/{n} done ({elapsed_ms:.0f}ms)")

    if progress_token:
        await _send_progress(progress_token, 0.0, f"⏳ Querying {n} models...")
        ticker_task = asyncio.create_task(_progress_ticker())
    else:
        ticker_task = None

    # Load history once for all models (avoids N redundant DB reads)
    history = get_history(CURRENT_SESSION)

    def _on_model_done(fut):
        nonlocal models_done, ticker_pct
        models_done += 1
        # Immediately jump progress bar when a model completes
        if progress_token:
            done_pct = int((models_done / n) * 99)
            if done_pct > ticker_pct:
                ticker_pct = done_pct
                elapsed_ms = (time.perf_counter() - t0) * 1000
                asyncio.create_task(_send_progress(progress_token, ticker_pct / 100, f"⚡ {models_done}/{n} done ({elapsed_ms:.0f}ms)"))

    def _detect_model_type(model_name):
        ml = model_name.lower()
        if model_name in DO_MODELS:
            return "digitalocean"
        if model_name in PUTER_MODELS:
            return "puter"
        if "gemini" in ml:
            return "google"
        if model_name in BEDROCK_MODELS or "claude" in ml or "anthropic" in ml or "nova" in ml or "llama4" in ml or "palmyra" in ml:
            return "bedrock"
        # Responses API models (codex, deep-research, pro reasoning)
        if "codex" in ml or "deep-research" in ml or ml in ("gpt-5.4-pro",):
            return "codex"
        if any(p in ml for p in ("gpt", "o1", "o3", "o4")):
            return "deployed"
        return "serverless"

    for m in models:
        m_type = _detect_model_type(m)

        async def _call_model(model_name, model_type):
            resp, usage, lat = await chat(client, user_message, None, model_name, model_type, cached_history=history)
            return model_name, resp, usage, lat

        task = asyncio.create_task(_call_model(m, m_type))
        task.add_done_callback(_on_model_done)
        tasks[m] = task

    # Wait for all tasks with per-model timeout
    done, pending = await asyncio.wait(tasks.values(), timeout=timeout)

    # Cancel stragglers
    for task in pending:
        task.cancel()

    # Stop ticker
    if ticker_task:
        ticker_task.cancel()
        try:
            await ticker_task
        except asyncio.CancelledError:
            pass
        # Bridge the gap: ticker may have been mid-sleep, catch up to 90%
        await _send_progress(progress_token, 0.99, f"⏳ {len(done)}/{n} responded, collecting results...")

    # Collect results in model list order
    final_output = ""
    for m in models:
        task = tasks[m]
        if task in done:
            try:
                name, resp, usage, lat = task.result()
                final_output += f"**[{name}]** ({lat:.0f}ms)\n{resp}\n\n"
            except Exception as e:
                final_output += f"**[{m}]** (error)\n{e}\n\n"
        else:
            final_output += f"**[{m}]** (timed out after {timeout}s)\n_Skipped — exceeded {timeout}s timeout_\n\n"

    wall_time = (time.perf_counter() - t0) * 1000
    final_output += f"_Wall time: {wall_time:.0f}ms across {n} models ({len(done)} responded, {len(pending)} timed out)_"

    if progress_token:
        await _send_progress(progress_token, 1.0, f"✅ Complete — {len(done)} responded, {len(pending)} timed out")

    # Don't save multi_chat to session history — it pollutes individual model contexts
    # with other models' responses, causing echo/repetition on subsequent calls.

    return final_output.strip()

# ── MCP protocol ────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "chat",
        "description": "Send a message to the Azure AI assistant and get a response.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The message to send to the assistant."},
            },
            "required": ["message"],
        },
    },
    {
        "name": "multi_chat",
        "description": (
            "Send a message to multiple models concurrently and get a combined response. "
            "If models is omitted, uses the configured default_models list. "
            "Each response includes per-model latency and a wall-time summary. "
            "Slow models are skipped after the configured timeout (default 15s). "
            "VOICE FLOW: To read responses aloud, pass output to multi_speak with voice assignments by family: "
            "OpenAI (gpt-5.x, o1, o4)→en-US-DavisNeural, Claude→en-US-AvaNeural, "
            "Llama→en-US-AndrewNeural, DeepSeek→en-US-BrianNeural, Grok→en-US-GuyNeural, "
            "Gemini→en-US-AriaNeural, Cohere→en-US-JennyNeural, Phi/Mistral→en-US-EmmaNeural, "
            "Nova (AWS)→en-US-JasonNeural, DigitalOcean OSS→en-US-TonyNeural, Puter→en-US-SaraNeural."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The message to send."},
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of model names to query. Optional — defaults to configured default_models."
                }
            },
            "required": ["message"],
        },
    },
    {
        "name": "configure",
        "description": "View or change assistant settings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "api_key": {"type": "string"},
                "endpoint": {"type": "string"},
                "deployment": {"type": "string"},
                "model": {"type": "string"},
                "model_type": {"type": "string", "enum": ["deployed", "serverless", "codex", "bedrock", "google", "digitalocean", "puter"]},
                "aws_access_key": {"type": "string", "description": "AWS Access Key ID for Bedrock."},
                "aws_secret_key": {"type": "string", "description": "AWS Secret Access Key for Bedrock."},
                "aws_region": {"type": "string", "description": "AWS region for Bedrock (default: us-east-1)."},
                "max_completion_tokens": {"type": "integer"},
                "temperature": {"type": "number"},
                "reasoning_effort": {"type": "string", "enum": ["low", "medium", "high", "xhigh"], "description": "Reasoning effort for codex/reasoning models."},
                "system_prompt": {"type": "string"},
                "conversation_max_turns": {"type": "integer"},
                "voice": {"type": "string"},
                "default_models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Default model list for multi_chat when models param is omitted."
                },
                "multi_chat_timeout": {
                    "type": "integer",
                    "description": "Per-model timeout in seconds for multi_chat (default 15)."
                },
                "google_api_key": {"type": "string", "description": "Google Vertex AI API key for Gemini models."},
                "google_project": {"type": "string", "description": "Google Cloud project ID or number."},
                "google_region": {"type": "string", "description": "Vertex AI region (default: global)."},
                "do_api_key": {"type": "string", "description": "DigitalOcean model access key for Serverless Inference."},
                "puter_api_key": {"type": "string", "description": "Puter auth token for OpenAI-compatible AI endpoint."},
                "azure_endpoints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "endpoint": {"type": "string"},
                            "api_key": {"type": "string"},
                        },
                        "required": ["endpoint", "api_key"],
                    },
                    "description": "Additional Azure AI endpoints for multi-region support. Each entry is {endpoint, api_key}.",
                },
            },
        },
    },
    {
        "name": "reset",
        "description": "Clear conversation history for the current session.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "clear_cache",
        "description": "Clear the in-memory response cache.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "status",
        "description": "Show the current status and availability of models.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "create_session",
        "description": "Create a new named chat session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the new session."}
            },
            "required": ["name"]
        },
    },
    {
        "name": "switch_session",
        "description": "Switch to a different chat session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the session to switch to."}
            },
            "required": ["name"]
        },
    },
    {
        "name": "delete_session",
        "description": "Delete a chat session and its history.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name of the session to delete."}
            },
            "required": ["name"]
        },
    },
    {
        "name": "list_sessions",
        "description": "List all available chat sessions.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "models",
        "description": "List and test available models.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "test": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "scan",
        "description": "Scan all models and show availability matrix with deployed/available status and latency.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


async def handle_request(client, req):
    method = req.get("method")
    params = req.get("params", {})
    req_id = req.get("id")
    progress_token = params.get("_meta", {}).get("progressToken")

    if method == "initialize":
        await _write_response({
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": True}},
                "serverInfo": {"name": "cloud-chat-assistant", "version": "1.2.0-async"},
            },
        })
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        await _write_response({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        asyncio.create_task(_run_tool(client, req_id, params.get("name"), params.get("arguments", {}), progress_token))


async def _run_tool(client, req_id, tool_name, args, progress_token):
    global CURRENT_SESSION, _model_status
    if tool_name == "chat":
        message = args.get("message", "")
        if not message:
            await _write_response(_result(req_id, "Error: 'message' is required."))
            return

        response, usage, latency = await chat(client, message, progress_token)
        await _write_response(_result(req_id, response))

    elif tool_name == "multi_chat":
        message = args.get("message", "")
        models = args.get("models")  # None = use configured defaults
        if not message:
            await _write_response(_result(req_id, "Error: 'message' is required."))
            return

        combined_response = await multi_chat(client, message, models, progress_token)
        await _write_response(_result(req_id, combined_response))

    elif tool_name == "configure":
        res = _handle_configure(args)
        await _write_response(_result(req_id, res))

    elif tool_name == "reset":
        clear_session(CURRENT_SESSION)
        await _write_response(_result(req_id, f"History for session '{CURRENT_SESSION}' has been cleared."))

    elif tool_name == "clear_cache":
        global _cache
        count = len(_cache)
        _cache.clear()
        await _write_response(_result(req_id, f"In-memory response cache cleared ({count} items removed)."))

    elif tool_name == "list_sessions":
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.execute("SELECT name FROM sessions ORDER BY name ASC")
                sessions = [r[0] for r in cursor.fetchall()]
                res = "**[Available Sessions]**\n" + "\n".join([f"* {s} {'(current)' if s == CURRENT_SESSION else ''}" for s in sessions])
                await _write_response(_result(req_id, res))
        except Exception as e:
            await _write_response(_result(req_id, f"Error listing sessions: {e}"))

    elif tool_name == "create_session":
        name = args.get("name", "").strip()
        if not name:
            await _write_response(_result(req_id, "Error: 'name' is required."))
            return
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("INSERT INTO sessions (name) VALUES (?)", (name,))
                await _write_response(_result(req_id, f"Session '{name}' created."))
        except sqlite3.IntegrityError:
            await _write_response(_result(req_id, f"Error: Session '{name}' already exists."))
        except Exception as e:
            await _write_response(_result(req_id, f"Error creating session: {e}"))

    elif tool_name == "switch_session":
        name = args.get("name", "").strip()
        if not name:
            await _write_response(_result(req_id, "Error: 'name' is required."))
            return
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.execute("SELECT 1 FROM sessions WHERE name = ?", (name,))
                if cursor.fetchone():
                    CURRENT_SESSION = name
                    await _write_response(_result(req_id, f"Switched to session '{name}'."))
                else:
                    await _write_response(_result(req_id, f"Error: Session '{name}' does not exist."))
        except Exception as e:
            await _write_response(_result(req_id, f"Error switching session: {e}"))

    elif tool_name == "delete_session":
        name = args.get("name", "").strip()
        if not name:
            await _write_response(_result(req_id, "Error: 'name' is required."))
            return
        if name == "default":
            await _write_response(_result(req_id, "Error: Cannot delete the 'default' session."))
            return
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("DELETE FROM sessions WHERE name = ?", (name,))
                if CURRENT_SESSION == name:
                    CURRENT_SESSION = "default"
                await _write_response(_result(req_id, f"Session '{name}' and its history deleted."))
        except Exception as e:
            await _write_response(_result(req_id, f"Error deleting session: {e}"))

    elif tool_name == "status":
        if not _model_status:
            await _write_response(_result(req_id, "All models are currently in standby (no recent calls)."))
        else:
            lines = ["**[Model Status]**"]
            for m, s in _model_status.items():
                lines.append(f"* {m}: {s}")
            await _write_response(_result(req_id, "\n".join(lines)))

    elif tool_name == "models":
        res = await _handle_models(client, args, progress_token)
        await _write_response(_result(req_id, res))

    elif tool_name == "scan":
        res = await _handle_scan(client, progress_token)
        await _write_response(_result(req_id, res))

    else:
        await _write_response({
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
        })


def _result(req_id, text):
    return {
        "jsonrpc": "2.0", "id": req_id,
        "result": {"content": [{"type": "text", "text": text}]},
    }


def _handle_configure(args):
    settable = {
        "api_key", "endpoint", "deployment", "model", "model_type",
        "max_completion_tokens", "temperature", "system_prompt",
        "conversation_max_turns", "voice", "default_models", "multi_chat_timeout", "reasoning_effort",
        "google_api_key", "google_project", "google_region",
        "aws_access_key", "aws_secret_key", "aws_region",
        "do_api_key", "puter_api_key", "azure_endpoints",
    }
    updated = []
    for k, v in args.items():
        if k not in settable: continue
        if k in ("api_key", "google_api_key", "aws_access_key", "aws_secret_key", "do_api_key", "puter_api_key"): CONFIG[k] = str(v)
        elif k == "endpoint": CONFIG[k] = str(v).rstrip("/")
        elif k in ("deployment", "model", "model_type", "system_prompt", "voice", "google_project", "google_region", "aws_region", "reasoning_effort"): CONFIG[k] = str(v)
        elif k == "max_completion_tokens": CONFIG[k] = max(1, min(int(v), 128000))
        elif k == "temperature": CONFIG[k] = max(0.0, min(float(v), 2.0))
        elif k == "conversation_max_turns": CONFIG[k] = max(1, min(int(v), 500))
        elif k == "default_models": CONFIG[k] = list(v) if isinstance(v, list) else [str(v)]
        elif k == "multi_chat_timeout": CONFIG[k] = max(1, min(int(v), 120))
        elif k == "azure_endpoints":
            # Validate and normalize endpoint entries
            eps = []
            for entry in (v if isinstance(v, list) else []):
                if isinstance(entry, dict) and entry.get("endpoint") and entry.get("api_key"):
                    eps.append({"endpoint": str(entry["endpoint"]).rstrip("/"), "api_key": str(entry["api_key"])})
            CONFIG[k] = eps
        else: CONFIG[k] = v
        if k == "azure_endpoints":
            updated.append(f"azure_endpoints=[{len(CONFIG[k])} endpoint(s)]")
        elif k not in ("api_key", "google_api_key", "aws_access_key", "aws_secret_key", "do_api_key", "puter_api_key"):
            updated.append(f"{k}={CONFIG[k]}")
        else:
            updated.append(f"{k}=***{str(v)[-4:]}")

    if updated:
        save_config()
        return "Updated: " + ", ".join(updated)

    lines = ["[Azure AI]"]
    lines.append(f"  endpoint:    {CONFIG.get('endpoint', '')}")
    lines.append(f"  api_key:     ***{CONFIG.get('api_key', '')[-4:]}" if CONFIG.get("api_key") else "  api_key:     (not set)")
    extra_eps = CONFIG.get("azure_endpoints", [])
    if extra_eps:
        lines.append(f"  + {len(extra_eps)} additional endpoint(s):")
        for ep in extra_eps:
            rname = _endpoint_resource_name(ep.get("endpoint", ""))
            lines.append(f"    - {rname}: ***{ep.get('api_key', '')[-4:]}")
    lines.append(f"  deployment:  {CONFIG.get('deployment', '')}")
    lines.append(f"  model:       {CONFIG.get('model', '')}")
    lines.append(f"  model_type:  {CONFIG.get('model_type', '')}")
    lines.append("")
    lines.append("[Generation]")
    lines.append(f"  max_tokens:  {CONFIG.get('max_completion_tokens', '')}")
    lines.append(f"  temp:        {CONFIG.get('temperature', '')}")
    lines.append("")
    lines.append("[Conversation]")
    lines.append(f"  turns:       {len(_conversation_history) // 2} / {CONFIG.get('conversation_max_turns', '')}")
    lines.append("")
    lines.append("[Multi-Chat]")
    dm = CONFIG.get("default_models", DEFAULTS["default_models"])
    lines.append(f"  defaults:    {', '.join(dm)}")
    lines.append(f"  timeout:     {CONFIG.get('multi_chat_timeout', DEFAULTS['multi_chat_timeout'])}s")
    lines.append("")
    lines.append("[Google Vertex AI]")
    gkey = CONFIG.get("google_api_key", "")
    lines.append(f"  api_key:     ***{gkey[-4:]}" if gkey else "  api_key:     (not set)")
    lines.append(f"  project:     {CONFIG.get('google_project', '') or '(not set)'}")
    lines.append(f"  region:      {CONFIG.get('google_region', 'global')}")
    gurl = _google_base_url()
    lines.append(f"  endpoint:    {gurl or '(set google_project to enable)'}")
    lines.append("")
    lines.append("[AWS Bedrock]")
    aws_key = CONFIG.get("aws_access_key", "")
    aws_secret = CONFIG.get("aws_secret_key", "")
    lines.append(f"  access_key:  ***{aws_key[-4:]}" if aws_key else "  access_key:  (not set)")
    lines.append(f"  secret_key:  ***{aws_secret[-4:]}" if aws_secret else "  secret_key:  (not set)")
    lines.append(f"  region:      {CONFIG.get('aws_region', 'us-east-1')}")
    lines.append(f"  models:      {', '.join(list(BEDROCK_MODELS.keys())[:5])}...")
    lines.append("")
    lines.append("[DigitalOcean]")
    do_key = CONFIG.get("do_api_key", "")
    lines.append(f"  api_key:     ***{do_key[-4:]}" if do_key else "  api_key:     (not set)")
    lines.append(f"  endpoint:    {DO_ENDPOINT}")
    lines.append(f"  models:      {', '.join(DO_MODELS[:5])}...")
    lines.append("")
    lines.append("[Puter]")
    puter_key = CONFIG.get("puter_api_key", "")
    lines.append(f"  api_key:     ***{puter_key[-4:]}" if puter_key else "  api_key:     (not set)")
    lines.append(f"  endpoint:    {PUTER_ENDPOINT}")
    lines.append(f"  models:      {', '.join(PUTER_MODELS[:5])}...")
    return "\n".join(lines)


async def _handle_models(client, args, progress_token):
    do_test = args.get("test", False)
    api_key = CONFIG.get("api_key", "")
    endpoint = CONFIG.get("endpoint", "")
    if not api_key or not endpoint: return "Error: api_key and endpoint required."

    current_deployment = CONFIG.get("deployment", "")
    current_model = CONFIG.get("model", "")
    has_aws = CONFIG.get("aws_access_key") and CONFIG.get("aws_secret_key")
    has_do = bool(CONFIG.get("do_api_key"))

    if do_test:
        # Parallel test all models for maximum speed
        all_tests = []
        model_info = []  # (name, type, section)

        for m in AZURE_DEPLOYED:
            ep_info = _deployment_map.get(m)
            if ep_info:
                all_tests.append(_test_model(client, m, "deployed", ep_info["endpoint"], ep_info["api_key"]))
            else:
                all_tests.append(_test_model(client, m, "deployed"))
            model_info.append((m, "deployed", "Azure Deployed"))
        for m in AZURE_SERVERLESS:
            all_tests.append(_test_model(client, m, "serverless"))
            model_info.append((m, "serverless", "Azure Serverless"))
        for m in GOOGLE_MODELS:
            all_tests.append(_test_model(client, m, "google"))
            model_info.append((m, "google", "Google Vertex AI"))
        if has_aws:
            for m in BEDROCK_MODELS.keys():
                all_tests.append(_test_model(client, m, "bedrock"))
                model_info.append((m, "bedrock", "AWS Bedrock"))
        if has_do:
            for m in DO_MODELS:
                all_tests.append(_test_model(client, m, "digitalocean"))
                model_info.append((m, "digitalocean", "DigitalOcean"))

        # Run all tests in parallel
        results = await asyncio.gather(*all_tests, return_exceptions=True)

        # Group results by section
        sections = {"Azure Deployed": [], "Azure Serverless": [], "Google Vertex AI": [], "AWS Bedrock": [], "DigitalOcean": []}
        for (name, mtype, section), result in zip(model_info, results):
            if isinstance(result, Exception):
                status = "can deploy" if section == "Azure Deployed" else "unavailable"
                sections[section].append(f"  {name}: {status}")
            else:
                text, _, latency = result
                if text.startswith("Error"):
                    # For deployed models, 404 means not deployed yet
                    if section == "Azure Deployed" and "404" in text:
                        status = "can deploy"
                    else:
                        status = "unavailable"
                else:
                    status = f"✓ ({latency:.0f}ms)"
                sections[section].append(f"  {name}: {status}")

        lines = [f"Endpoint: {endpoint}\n"]
        for section in ["Azure Deployed", "Azure Serverless", "Google Vertex AI", "AWS Bedrock", "DigitalOcean"]:
            lines.append(f"[{section}]")
            if section == "AWS Bedrock" and not has_aws:
                lines.append("  (set aws_access_key and aws_secret_key to enable)")
            elif section == "DigitalOcean" and not has_do:
                lines.append("  (set do_api_key to enable)")
            else:
                lines.extend(sections[section])
            lines.append("")
        return "\n".join(lines).strip()

    # Non-test mode: just list models
    lines = [f"Endpoint: {endpoint}\n", "[Azure Deployed]"]
    for m in AZURE_DEPLOYED:
        marker = " (current)" if m == current_deployment else ""
        lines.append(f"  {m}{marker}")
    lines.append("\n[Azure Serverless]")
    for m in AZURE_SERVERLESS:
        marker = " (current)" if m == current_model else ""
        lines.append(f"  {m}{marker}")
    lines.append("\n[Google Vertex AI]")
    for m in GOOGLE_MODELS:
        lines.append(f"  {m}")
    lines.append("\n[AWS Bedrock]")
    if has_aws:
        for m in BEDROCK_MODELS.keys():
            lines.append(f"  {m}")
    else:
        lines.append("  (set aws_access_key and aws_secret_key to enable)")
    lines.append("\n[DigitalOcean]")
    if has_do:
        for m in DO_MODELS:
            lines.append(f"  {m}")
    else:
        lines.append("  (set do_api_key to enable)")
    return "\n".join(lines)


async def _test_model(client, name, mtype, endpoint_override=None, api_key_override=None):
    try:
        if mtype == "bedrock":
            messages = [{"role": "user", "content": "hi"}]
            return await call_bedrock(client, messages, None, name)
        elif mtype == "digitalocean":
            url = f"{DO_ENDPOINT}/chat/completions"
            body = {"messages": [{"role": "user", "content": "hi"}], "model": name, "max_completion_tokens": 256}
            headers = {"Authorization": f"Bearer {CONFIG.get('do_api_key', '')}", "Content-Type": "application/json"}
        elif mtype == "puter":
            url = f"{PUTER_ENDPOINT}/chat/completions"
            body = {"messages": [{"role": "user", "content": "hi"}], "model": name, "max_completion_tokens": 256}
            headers = {"Authorization": f"Bearer {CONFIG.get('puter_api_key', '')}", "Content-Type": "application/json"}
        elif mtype == "google":
            google_ep = _google_base_url()
            url = f"{google_ep}/chat/completions"
            body = {"messages": [{"role": "user", "content": "hi"}], "model": f"google/{name}", "max_tokens": 10}
            headers = {"x-goog-api-key": CONFIG.get("google_api_key", ""), "Content-Type": "application/json"}
        elif mtype == "codex":
            ep = endpoint_override or CONFIG['endpoint']
            key = api_key_override or CONFIG['api_key']
            url = f"{ep}/openai/v1/responses"
            body = {"model": name, "input": "Say hi in one word.", "reasoning": {"effort": "low"}, "max_output_tokens": 20}
            headers = {"api-key": key, "Content-Type": "application/json"}
        elif mtype == "deployed":
            ep = endpoint_override or CONFIG['endpoint']
            key = api_key_override or CONFIG['api_key']
            url = f"{ep}/openai/deployments/{name}/chat/completions?api-version=2024-12-01-preview"
            body = {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 10}
            headers = {"api-key": key, "Content-Type": "application/json"}
        else:
            ep = endpoint_override or CONFIG['endpoint']
            key = api_key_override or CONFIG['api_key']
            url = f"{ep}/models/chat/completions?api-version=2024-05-01-preview"
            body = {"messages": [{"role": "user", "content": "hi"}], "model": name, "max_tokens": 10}
            headers = {"api-key": key, "Content-Type": "application/json"}

        t0 = time.perf_counter()
        timeout = 30.0 if mtype == "codex" else 15.0
        resp = await client.post(url, json=body, headers=headers, timeout=timeout)
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200: return f"Error {resp.status_code}", {}, 0
        data = resp.json()
        if mtype == "codex":
            # Responses API output format
            text = ""
            for item in data.get("output", []):
                if item.get("type") == "message":
                    for block in item.get("content", []):
                        if block.get("type") == "output_text":
                            text += block.get("text", "")
            return text or "OK", data.get("usage", {}), latency
        return data["choices"][0]["message"]["content"], data.get("usage", {}), latency
    except Exception as e:
        return f"Error: {e}", {}, 0


async def _handle_scan(client, progress_token):
    """Scan all models across all configured Azure endpoints and other providers.

    Uses CLI tools (az, aws, gcloud) for dynamic discovery when available,
    falls back to hardcoded lists otherwise. Supports multi-region Azure scanning.
    """
    global _deployment_map

    azure_eps = _get_all_azure_endpoints()
    if not azure_eps:
        return "Error: api_key and endpoint required. Use configure tool to set them."

    has_aws = CONFIG.get("aws_access_key") and CONFIG.get("aws_secret_key")
    has_google = CONFIG.get("google_api_key")
    has_do = bool(CONFIG.get("do_api_key"))

    lines = ["## Model Availability Scan\n"]
    cli_info = []

    # Check CLI availability
    has_az = _cli_available("az")
    has_aws_cli = _cli_available("aws")
    has_gcloud = _cli_available("gcloud")
    cli_info.append(f"CLIs: az={'yes' if has_az else 'no'}, aws={'yes' if has_aws_cli else 'no'}, gcloud={'yes' if has_gcloud else 'no'}")

    if progress_token:
        await _send_progress(progress_token, 0.05, f"Discovering models across {len(azure_eps)} Azure endpoint(s)...")

    # ── Multi-region Azure discovery ──
    # Per-endpoint: {endpoint -> {deployable: [...], deployed: [...]}}
    ep_discovery = {}
    all_deployable = set()
    all_deployed = {}  # model_name -> ep_info
    bedrock_models = None

    # Build CLI tasks across all Azure endpoints + Bedrock
    cli_tasks = []
    if has_az:
        for ep_info in azure_eps:
            ep = ep_info["endpoint"]
            rname = _endpoint_resource_name(ep)
            cli_tasks.append((f"deployable:{rname}", _az_list_deployable_models(ep), ep_info))
            cli_tasks.append((f"deployed:{rname}", _az_list_deployed(ep), ep_info))
    if has_aws_cli and has_aws:
        cli_tasks.append(("bedrock", _aws_list_bedrock_models(), None))

    if cli_tasks:
        cli_results = await asyncio.gather(*[t[1] for t in cli_tasks], return_exceptions=True)
        for (task_name, _, ep_info), result in zip(cli_tasks, cli_results):
            if isinstance(result, Exception) or result is None:
                continue
            if task_name.startswith("deployable:"):
                rname = task_name.split(":", 1)[1]
                ep_discovery.setdefault(rname, {})["deployable"] = result
                all_deployable.update(result)
            elif task_name.startswith("deployed:"):
                rname = task_name.split(":", 1)[1]
                ep_discovery.setdefault(rname, {})["deployed"] = result
                for m in result:
                    all_deployed[m] = ep_info
            elif task_name == "bedrock":
                bedrock_models = result

    # Populate deployment map for routing
    _deployment_map = dict(all_deployed)

    # Summary info
    total_deployed = len(all_deployed)
    if has_az and ep_discovery:
        ep_summaries = []
        for rname, info in ep_discovery.items():
            n_dep = len(info.get("deployed", []))
            n_avail = len(info.get("deployable", []))
            ep_summaries.append(f"{rname}: {n_avail} deployable, {n_dep} deployed")
        cli_info.append(f"Azure endpoints: {' | '.join(ep_summaries)}")
    elif has_az:
        cli_info.append("Azure: CLI available but no models discovered")
    else:
        cli_info.append("Azure: using hardcoded model list (az CLI not available)")

    if bedrock_models:
        cli_info.append(f"Bedrock: {len(bedrock_models)} models from CLI")
    else:
        cli_info.append("Bedrock: using hardcoded model list")

    if progress_token:
        await _send_progress(progress_token, 0.1, "Testing models...")

    # ── Build test tasks ──
    all_tests = []
    model_info = []  # (name, type, section, is_deployed, resource_name)

    # Azure deployed models: only test actually-deployed models against their endpoint
    # For deployable-but-not-deployed, mark as "can deploy" without testing
    azure_deployed_models = all_deployable if all_deployable else set(AZURE_DEPLOYED)

    # Codex models use the Responses API, not chat completions
    _codex_patterns = ("codex",)

    for m in sorted(azure_deployed_models):
        if m in all_deployed:
            ep_info = all_deployed[m]
            rname = _endpoint_resource_name(ep_info["endpoint"])
            mtype = "codex" if any(p in m.lower() for p in _codex_patterns) else "deployed"
            all_tests.append(_test_model(client, m, mtype, ep_info["endpoint"], ep_info["api_key"]))
            model_info.append((m, mtype, "Azure OpenAI", True, rname))
        else:
            # Not deployed — skip testing, will mark as "can deploy"
            all_tests.append(asyncio.coroutine(lambda: ("not_deployed", {}, 0))() if False else None)
            model_info.append((m, "deployed", "Azure OpenAI", False, None))

    # Azure serverless — test against primary endpoint
    primary_ep = azure_eps[0]
    for m in AZURE_SERVERLESS:
        all_tests.append(_test_model(client, m, "serverless", primary_ep["endpoint"], primary_ep["api_key"]))
        model_info.append((m, "serverless", "Azure Serverless", True, None))

    if has_google:
        for m in GOOGLE_MODELS:
            all_tests.append(_test_model(client, m, "google"))
            model_info.append((m, "google", "Google Gemini", True, None))
    if has_aws:
        for m in BEDROCK_MODELS.keys():
            all_tests.append(_test_model(client, m, "bedrock"))
            model_info.append((m, "bedrock", "AWS Bedrock", True, None))
    if has_do:
        for m in DO_MODELS:
            all_tests.append(_test_model(client, m, "digitalocean"))
            model_info.append((m, "digitalocean", "DigitalOcean", True, None))

    # Filter out None entries (non-deployed models that we skip testing)
    test_indices = [i for i, t in enumerate(all_tests) if t is not None]
    actual_tests = [all_tests[i] for i in test_indices]

    if progress_token:
        await _send_progress(progress_token, 0.15, f"Testing {len(actual_tests)} models...")

    # Run all tests in parallel
    actual_results = await asyncio.gather(*actual_tests, return_exceptions=True)

    # Map results back to full index
    results = [None] * len(all_tests)
    for idx, result in zip(test_indices, actual_results):
        results[idx] = result

    if progress_token:
        await _send_progress(progress_token, 0.9, "Formatting results...")

    # ── Build table output ──
    sections = {"Azure OpenAI": [], "Azure Serverless": [], "Google Gemini": [], "AWS Bedrock": [], "DigitalOcean": []}
    counts = {"pass": 0, "fail": 0, "deploy": 0, "deployed": 0}

    for i, (name, mtype, section, is_deployed, rname) in enumerate(model_info):
        result = results[i]

        if result is None:
            # Non-deployed model, skipped testing
            status = "can deploy"
            counts["deploy"] += 1
        elif isinstance(result, Exception):
            if section == "Azure OpenAI" and not is_deployed:
                status = "can deploy"
                counts["deploy"] += 1
            else:
                status = "error"
                counts["fail"] += 1
        else:
            text, _, latency = result
            if text.startswith("Error"):
                if section == "Azure OpenAI" and ("404" in text or "not found" in text.lower()):
                    status = "can deploy"
                    counts["deploy"] += 1
                else:
                    status = "unavailable"
                    counts["fail"] += 1
            else:
                status = f"{latency:.0f}ms"
                counts["pass"] += 1
                if section == "Azure OpenAI" and is_deployed:
                    counts["deployed"] += 1

        # Annotate with resource name for deployed models on non-primary endpoints
        display_name = name
        if rname and len(azure_eps) > 1:
            display_name = f"{name} ({rname})"

        sections[section].append(f"| {display_name:<50} | {status:<15} |")

    # Output CLI info
    lines.append("_" + " | ".join(cli_info) + "_\n")

    for section in ["Azure OpenAI", "Azure Serverless", "Google Gemini", "AWS Bedrock", "DigitalOcean"]:
        if not sections[section]:
            if section == "Google Gemini" and not has_google:
                lines.append(f"### {section}\n_(set google_api_key to enable)_\n")
            elif section == "AWS Bedrock" and not has_aws:
                lines.append(f"### {section}\n_(set aws_access_key and aws_secret_key to enable)_\n")
            elif section == "DigitalOcean" and not has_do:
                lines.append(f"### {section}\n_(set do_api_key to enable)_\n")
            continue
        lines.append(f"### {section}")
        lines.append("| Model | Status |")
        lines.append("|-------|--------|")
        lines.extend(sections[section])
        lines.append("")

    lines.append(f"**Summary:** {counts['pass']} working ({counts['deployed']} deployed), {counts['fail']} unavailable, {counts['deploy']} can deploy")

    if progress_token:
        await _send_progress(progress_token, 1.0, "Scan complete")

    return "\n".join(lines)


# ── MCP transport ───────────────────────────────────────────────────────────

async def _send_progress(token, progress, message=""):
    """Send MCP progress notification. Progress is 0.0-1.0 float, sent as 0-100 integer."""
    await _write_response({
        "jsonrpc": "2.0", "method": "notifications/progress",
        "params": {"progressToken": token, "progress": int(progress * 100), "total": 100, "message": message},
    })


async def _write_response(resp):
    async with _stdout_lock:
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


async def main():
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    # Enhanced Connection Pooling: tune limits for high-concurrency multi-agent calls
    limits = httpx.Limits(
        max_connections=100,          # Allow up to 100 concurrent connections
        max_keepalive_connections=20, # Keep up to 20 connections alive for reuse
        keepalive_expiry=30.0         # Connections expire after 30s of inactivity
    )
    timeout = httpx.Timeout(60.0, connect=10.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Pre-warm TLS connection to Azure in background (don't block MCP handshake)
        async def _warmup():
            api_key = CONFIG.get("api_key", "")
            endpoint = CONFIG.get("endpoint", "")
            if api_key and endpoint:
                try:
                    await client.post(
                        f"{endpoint}/openai/deployments/{CONFIG.get('deployment', '')}/chat/completions?api-version=2024-12-01-preview",
                        json={"messages": [{"role": "user", "content": "warmup"}], "max_completion_tokens": 1},
                        headers={"api-key": api_key},
                        timeout=10.0,
                    )
                except Exception:
                    pass
        asyncio.create_task(_warmup())

        while True:
            line = await reader.readline()
            if not line: break
            try:
                req = json.loads(line.decode().strip())
                await handle_request(client, req)
            except Exception:
                continue

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
