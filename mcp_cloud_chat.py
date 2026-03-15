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
import hmac
import json
import os
import shutil
import subprocess
import sys
import time
import sqlite3
from datetime import datetime, timezone
from urllib.parse import quote, urlparse
import httpx

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
    "model_type": "deployed",       # "deployed" (OpenAI endpoint) or "serverless" (unified inference)
    "max_completion_tokens": 2048,
    "temperature": 1.0,
    "system_prompt": "You are a helpful chat assistant. Keep responses concise and conversational.",
    "conversation_max_turns": 50,    # max history turns before auto-trimming
    "voice": "",                     # default TTS voice (empty = use speech config)
    "default_models": ["gpt-5.3-chat", "Meta-Llama-3.1-405B-Instruct", "Phi-4"],  # models for multi_chat when none specified
    "multi_chat_timeout": 15,        # per-model timeout in seconds for multi_chat
    "google_api_key": "",
    "google_project": "",
    "google_region": "global",
    # AWS Bedrock
    "aws_access_key": "",
    "aws_secret_key": "",
    "aws_region": "us-east-1",
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

# AWS Bedrock model IDs (cross-region inference profiles)
BEDROCK_MODELS = {
    # Anthropic Claude 4.x
    "claude-opus-4.5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-opus-4.6": "us.anthropic.claude-opus-4-6-v1",
    "claude-sonnet-4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-sonnet-4.5": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4.6": "us.anthropic.claude-sonnet-4-6",
    "claude-haiku-4.5": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    # Amazon Nova
    "nova-pro": "us.amazon.nova-pro-v1:0",
    "nova-lite": "us.amazon.nova-lite-v1:0",
    "nova-2-lite": "us.amazon.nova-2-lite-v1:0",
    # Meta Llama 4
    "llama4-maverick-17b": "us.meta.llama4-maverick-17b-instruct-v1:0",
    "llama4-scout-17b": "us.meta.llama4-scout-17b-instruct-v1:0",
    # Writer
    "palmyra-x4": "us.writer.palmyra-x4-v1:0",
    "palmyra-x5": "us.writer.palmyra-x5-v1:0",
}

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

async def _az_list_deployable_models():
    """Use Azure CLI to list deployable OpenAI models."""
    if not _cli_available("az"):
        return None
    # Extract resource name from endpoint
    endpoint = CONFIG.get("endpoint", "")
    if not endpoint:
        return None
    # Endpoint format: https://<resource-name>.openai.azure.com or similar
    try:
        from urllib.parse import urlparse
        host = urlparse(endpoint).hostname
        resource_name = host.split(".")[0] if host else None
    except Exception:
        return None
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

async def _az_list_deployed():
    """Use Azure CLI to list currently deployed models."""
    if not _cli_available("az"):
        return None
    endpoint = CONFIG.get("endpoint", "")
    if not endpoint:
        return None
    try:
        from urllib.parse import urlparse
        host = urlparse(endpoint).hostname
        resource_name = host.split(".")[0] if host else None
    except Exception:
        return None
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
    for k in ("api_key", "endpoint", "deployment", "model", "model_type", "google_api_key", "google_project", "google_region", "aws_access_key", "aws_secret_key", "aws_region"):
        disk[k] = CONFIG.get(k, "")
    with open(CONFIG_PATH, "w") as f:
        json.dump(disk, f, indent=4)


CONFIG = load_config()


def _google_base_url():
    """Build Vertex AI OpenAI-compatible base URL from project + region config."""
    project = CONFIG.get("google_project", "")
    region = CONFIG.get("google_region", "global")
    if not project:
        return ""
    return f"https://aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{region}/endpoints/openapi"


# ── AWS SigV4 Signing ───────────────────────────────────────────────────────

def _aws_sign(method, url, headers, payload, region, service="bedrock"):
    """Sign an AWS request using Signature Version 4."""
    access_key = CONFIG.get("aws_access_key", "")
    secret_key = CONFIG.get("aws_secret_key", "")
    if not access_key or not secret_key:
        return headers

    parsed = urlparse(url)
    host = parsed.netloc
    uri = parsed.path or "/"

    t = datetime.now(timezone.utc)
    amz_date = t.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = t.strftime("%Y%m%d")

    canonical_uri = quote(uri, safe="/-_.~")
    headers_to_sign = {"host": host, "x-amz-date": amz_date, "content-type": "application/json"}
    signed_headers = ";".join(sorted(headers_to_sign.keys()))
    canonical_headers = "".join(f"{k}:{v}\n" for k, v in sorted(headers_to_sign.items()))
    payload_hash = hashlib.sha256(payload.encode() if isinstance(payload, str) else payload).hexdigest()

    canonical_request = f"{method}\n{canonical_uri}\n\n{canonical_headers}\n{signed_headers}\n{payload_hash}"

    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = f"{algorithm}\n{amz_date}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode()).hexdigest()}"

    def sign(key, msg):
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    k_date = sign(f"AWS4{secret_key}".encode(), date_stamp)
    k_region = sign(k_date, region)
    k_service = sign(k_region, service)
    k_signing = sign(k_service, "aws4_request")
    signature = hmac.new(k_signing, string_to_sign.encode(), hashlib.sha256).hexdigest()

    authorization = f"{algorithm} Credential={access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"

    return {**headers, "host": host, "x-amz-date": amz_date, "authorization": authorization, "content-type": "application/json"}


def _get_bedrock_model_id(model_name):
    """Convert friendly model name to Bedrock model ID."""
    if model_name in BEDROCK_MODELS:
        return BEDROCK_MODELS[model_name]
    if "." in model_name or ":" in model_name:
        return model_name
    return None


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

    # Route to Bedrock if model type is bedrock (use deployment for model name)
    if model_type == "bedrock":
        return await call_bedrock(client, messages, progress_token, deployment)

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
        if not api_key:
            return "Error: No API key configured. Use configure tool to set api_key.", {}, 0
        if not endpoint:
            return "Error: No endpoint configured.", {}, 0
        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-12-01-preview"
        body = {
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {
            "api-key": api_key,
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
        if "gemini" in ml:
            return "google"
        if model_name in BEDROCK_MODELS or "claude" in ml or "anthropic" in ml or "nova" in ml or "llama4" in ml:
            return "bedrock"
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
            "VOICE FLOW: To read responses aloud, pass the output to multi_speak with these voice assignments: "
            "gpt-5.3-chat→en-US-DavisNeural, Meta-Llama-3.1-405B-Instruct→en-US-AndrewNeural, "
            "DeepSeek-R1→en-US-BrianNeural, Phi-4→en-US-JennyNeural. "
            "Claude (the caller) uses en-US-AvaNeural."
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
                "model_type": {"type": "string", "enum": ["deployed", "serverless", "bedrock", "google"]},
                "aws_access_key": {"type": "string", "description": "AWS Access Key ID for Bedrock."},
                "aws_secret_key": {"type": "string", "description": "AWS Secret Access Key for Bedrock."},
                "aws_region": {"type": "string", "description": "AWS region for Bedrock (default: us-east-1)."},
                "max_completion_tokens": {"type": "integer"},
                "temperature": {"type": "number"},
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
                "serverInfo": {"name": "azure-chat-assistant", "version": "1.2.0-async"},
            },
        })
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        await _write_response({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        asyncio.create_task(_run_tool(client, req_id, params.get("name"), params.get("arguments", {}), progress_token))


async def _run_tool(client, req_id, tool_name, args, progress_token):
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
        global CURRENT_SESSION
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
        global _model_status
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
        "conversation_max_turns", "voice", "default_models", "multi_chat_timeout",
        "google_api_key", "google_project", "google_region",
        "aws_access_key", "aws_secret_key", "aws_region",
    }
    updated = []
    for k, v in args.items():
        if k not in settable: continue
        if k in ("api_key", "google_api_key", "aws_access_key", "aws_secret_key"): CONFIG[k] = str(v)
        elif k == "endpoint": CONFIG[k] = str(v).rstrip("/")
        elif k in ("deployment", "model", "model_type", "system_prompt", "voice", "google_project", "google_region", "aws_region"): CONFIG[k] = str(v)
        elif k == "max_completion_tokens": CONFIG[k] = max(1, min(int(v), 128000))
        elif k == "temperature": CONFIG[k] = max(0.0, min(float(v), 2.0))
        elif k == "conversation_max_turns": CONFIG[k] = max(1, min(int(v), 500))
        elif k == "default_models": CONFIG[k] = list(v) if isinstance(v, list) else [str(v)]
        elif k == "multi_chat_timeout": CONFIG[k] = max(1, min(int(v), 120))
        else: CONFIG[k] = v
        updated.append(f"{k}={CONFIG[k]}" if k not in ("api_key", "google_api_key", "aws_access_key", "aws_secret_key") else f"{k}=***{str(v)[-4:]}")

    if updated:
        save_config()
        return "Updated: " + ", ".join(updated)

    lines = ["[Azure AI]"]
    lines.append(f"  endpoint:    {CONFIG.get('endpoint', '')}")
    lines.append(f"  api_key:     ***{CONFIG.get('api_key', '')[-4:]}" if CONFIG.get("api_key") else "  api_key:     (not set)")
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
    return "\n".join(lines)


async def _handle_models(client, args, progress_token):
    do_test = args.get("test", False)
    api_key = CONFIG.get("api_key", "")
    endpoint = CONFIG.get("endpoint", "")
    if not api_key or not endpoint: return "Error: api_key and endpoint required."

    current_deployment = CONFIG.get("deployment", "")
    current_model = CONFIG.get("model", "")
    has_aws = CONFIG.get("aws_access_key") and CONFIG.get("aws_secret_key")

    if do_test:
        # Parallel test all models for maximum speed
        all_tests = []
        model_info = []  # (name, type, section)

        for m in AZURE_DEPLOYED:
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

        # Run all tests in parallel
        results = await asyncio.gather(*all_tests, return_exceptions=True)

        # Group results by section
        sections = {"Azure Deployed": [], "Azure Serverless": [], "Google Vertex AI": [], "AWS Bedrock": []}
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
        for section in ["Azure Deployed", "Azure Serverless", "Google Vertex AI", "AWS Bedrock"]:
            lines.append(f"[{section}]")
            if section == "AWS Bedrock" and not has_aws:
                lines.append("  (set aws_access_key and aws_secret_key to enable)")
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
    return "\n".join(lines)


async def _test_model(client, name, mtype):
    try:
        if mtype == "bedrock":
            messages = [{"role": "user", "content": "hi"}]
            return await call_bedrock(client, messages, None, name)
        elif mtype == "google":
            google_ep = _google_base_url()
            url = f"{google_ep}/chat/completions"
            body = {"messages": [{"role": "user", "content": "hi"}], "model": f"google/{name}", "max_tokens": 10}
            headers = {"x-goog-api-key": CONFIG.get("google_api_key", ""), "Content-Type": "application/json"}
        elif mtype == "deployed":
            url = f"{CONFIG['endpoint']}/openai/deployments/{name}/chat/completions?api-version=2024-12-01-preview"
            body = {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 10}
            headers = {"api-key": CONFIG["api_key"], "Content-Type": "application/json"}
        else:
            url = f"{CONFIG['endpoint']}/models/chat/completions?api-version=2024-05-01-preview"
            body = {"messages": [{"role": "user", "content": "hi"}], "model": name, "max_tokens": 10}
            headers = {"api-key": CONFIG["api_key"], "Content-Type": "application/json"}

        t0 = time.perf_counter()
        resp = await client.post(url, json=body, headers=headers, timeout=15.0)
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200: return f"Error {resp.status_code}", {}, 0
        data = resp.json()
        return data["choices"][0]["message"]["content"], data.get("usage", {}), latency
    except Exception as e:
        return f"Error: {e}", {}, 0


async def _handle_scan(client, progress_token):
    """Scan all models and return a formatted availability matrix.

    Uses CLI tools (az, aws, gcloud) for dynamic discovery when available,
    falls back to hardcoded lists otherwise.
    """
    api_key = CONFIG.get("api_key", "")
    endpoint = CONFIG.get("endpoint", "")
    if not api_key or not endpoint:
        return "Error: api_key and endpoint required. Use configure tool to set them."

    has_aws = CONFIG.get("aws_access_key") and CONFIG.get("aws_secret_key")
    has_google = CONFIG.get("google_api_key")

    lines = ["## Model Availability Scan\n"]
    cli_info = []

    # Check CLI availability
    has_az = _cli_available("az")
    has_aws_cli = _cli_available("aws")
    has_gcloud = _cli_available("gcloud")
    cli_info.append(f"CLIs: az={'yes' if has_az else 'no'}, aws={'yes' if has_aws_cli else 'no'}, gcloud={'yes' if has_gcloud else 'no'}")

    if progress_token:
        await _send_progress(progress_token, 0.05, "Discovering models via CLI...")

    # Dynamic model discovery
    az_deployable = None
    az_deployed = None
    bedrock_models = None

    # Run CLI queries in parallel
    cli_tasks = []
    if has_az:
        cli_tasks.append(("az_deployable", _az_list_deployable_models()))
        cli_tasks.append(("az_deployed", _az_list_deployed()))
    if has_aws_cli and has_aws:
        cli_tasks.append(("bedrock", _aws_list_bedrock_models()))

    if cli_tasks:
        cli_results = await asyncio.gather(*[t[1] for t in cli_tasks], return_exceptions=True)
        for (name, _), result in zip(cli_tasks, cli_results):
            if isinstance(result, Exception):
                continue
            if name == "az_deployable":
                az_deployable = result
            elif name == "az_deployed":
                az_deployed = result
            elif name == "bedrock":
                bedrock_models = result

    # Build model lists with dynamic discovery fallback
    azure_deployed_models = az_deployable if az_deployable else AZURE_DEPLOYED
    already_deployed = set(az_deployed) if az_deployed else set()

    if az_deployable:
        cli_info.append(f"Azure: {len(az_deployable)} deployable models from CLI, {len(already_deployed)} deployed")
    else:
        cli_info.append("Azure: using hardcoded model list (az CLI not available or auth failed)")

    if bedrock_models:
        cli_info.append(f"Bedrock: {len(bedrock_models)} models from CLI")
    else:
        cli_info.append("Bedrock: using hardcoded model list")

    if progress_token:
        await _send_progress(progress_token, 0.1, f"Testing models...")

    # Build test tasks
    all_tests = []
    model_info = []  # (name, type, section, is_deployed)

    for m in azure_deployed_models:
        all_tests.append(_test_model(client, m, "deployed"))
        model_info.append((m, "deployed", "Azure OpenAI", m in already_deployed))
    for m in AZURE_SERVERLESS:
        all_tests.append(_test_model(client, m, "serverless"))
        model_info.append((m, "serverless", "Azure Serverless", True))
    if has_google:
        for m in GOOGLE_MODELS:
            all_tests.append(_test_model(client, m, "google"))
            model_info.append((m, "google", "Google Gemini", True))
    if has_aws:
        # Use CLI-discovered Bedrock models or fall back to hardcoded
        bedrock_to_test = list(BEDROCK_MODELS.keys())  # Always use friendly names
        for m in bedrock_to_test:
            all_tests.append(_test_model(client, m, "bedrock"))
            model_info.append((m, "bedrock", "AWS Bedrock", True))

    if progress_token:
        await _send_progress(progress_token, 0.15, f"Testing {len(all_tests)} models...")

    # Run all tests in parallel
    results = await asyncio.gather(*all_tests, return_exceptions=True)

    if progress_token:
        await _send_progress(progress_token, 0.9, "Formatting results...")

    # Build table output
    sections = {"Azure OpenAI": [], "Azure Serverless": [], "Google Gemini": [], "AWS Bedrock": []}
    counts = {"pass": 0, "fail": 0, "deploy": 0, "deployed": 0}

    for (name, mtype, section, is_deployed), result in zip(model_info, results):
        if isinstance(result, Exception):
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
        sections[section].append(f"| {name:<40} | {status:<15} |")

    # Output CLI info
    lines.append("_" + " | ".join(cli_info) + "_\n")

    for section in ["Azure OpenAI", "Azure Serverless", "Google Gemini", "AWS Bedrock"]:
        if not sections[section]:
            if section == "Google Gemini" and not has_google:
                lines.append(f"### {section}\n_(set google_api_key to enable)_\n")
            elif section == "AWS Bedrock" and not has_aws:
                lines.append(f"### {section}\n_(set aws_access_key and aws_secret_key to enable)_\n")
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
