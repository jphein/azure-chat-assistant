#!/usr/bin/env python3
"""
Azure Voice Assistant MCP Server — standalone voice assistant powered by Azure AI models.

Uses Azure AI Foundry for LLM (GPT-5.3, grok-3, Llama, etc.) and delegates to the
azure-speech MCP server for TTS/STT when used alongside it, or works as a pure
chat tool on its own.

Refactored for asyncio and httpx for maximum efficiency and concurrency.

Tools:
  - chat:      Send a message to the LLM, get a response (with conversation history)
  - configure: View/change settings dynamically (model, API key, region, etc.)
  - reset:     Clear conversation history
  - models:    List available models and test connectivity
"""

import asyncio
import hashlib
import json
import os
import sys
import time
import sqlite3
import httpx

# ── Config ──────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.expanduser("~/.config/azure-voice-assistant/config.json")
DB_PATH = os.path.expanduser("~/.config/azure-voice-assistant/sessions.db")

DEFAULTS = {
    "api_key": "",
    "endpoint": "https://claud-assistant-resource.services.ai.azure.com",
    "deployment": "gpt-5.3-chat",
    "model": "gpt-5.3-chat-2026-03-03",
    "model_type": "deployed",       # "deployed" (OpenAI endpoint) or "serverless" (unified inference)
    "max_completion_tokens": 2048,
    "temperature": 1.0,
    "system_prompt": "You are a helpful voice assistant. Keep responses concise and conversational.",
    "conversation_max_turns": 50,    # max history turns before auto-trimming
    "voice": "",                     # default TTS voice (empty = use speech config)
    "default_models": ["gpt-5.3-chat", "Meta-Llama-3.1-405B-Instruct", "Phi-4"],  # models for multi_chat when none specified
    "multi_chat_timeout": 15,        # per-model timeout in seconds for multi_chat
}

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
    for k in ("api_key", "endpoint", "deployment", "model", "model_type"):
        disk[k] = CONFIG[k]
    with open(CONFIG_PATH, "w") as f:
        json.dump(disk, f, indent=4)


CONFIG = load_config()

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

    if not api_key:
        return "Error: No API key configured. Use configure tool to set api_key.", {}, 0
    if not endpoint:
        return "Error: No endpoint configured.", {}, 0

    if model_type == "deployed":
        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-10-21"
        body = {
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    else:
        url = f"{endpoint}/models/chat/completions?api-version=2024-05-01-preview"
        body = {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    if temperature != 1.0:
        body["temperature"] = temperature

    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
    }

    if progress_token:
        await _send_progress(progress_token, 0.1, f"[{model}] Thinking...")

    t0 = time.perf_counter()
    
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
                    _model_status[model] = "Filter Triggered"
                    state["error"] = f"**[{model}]** declined to answer (Content Filter). Try rephrasing your request."
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

    for m in models:
        m_type = "deployed" if "gpt" in m.lower() else "serverless"

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
                "model_type": {"type": "string", "enum": ["deployed", "serverless"]},
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
            },
        },
    },
    {
        "name": "reset",
        "description": "Clear conversation history for the current session.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_sessions",
        "description": "List all available chat sessions.",
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
        "name": "models",
        "description": "List and test available models.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "test": {"type": "boolean", "default": False},
            },
        },
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
                "serverInfo": {"name": "azure-voice-assistant", "version": "1.2.0-async"},
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

    elif tool_name == "clear_cache":
        global _cache
        count = len(_cache)
        _cache = {}
        await _write_response(_result(req_id, f"Cache cleared ({count} items removed)."))

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
    }
    updated = []
    for k, v in args.items():
        if k not in settable: continue
        if k == "api_key": CONFIG[k] = str(v)
        elif k == "endpoint": CONFIG[k] = str(v).rstrip("/")
        elif k in ("deployment", "model", "model_type", "system_prompt", "voice"): CONFIG[k] = str(v)
        elif k == "max_completion_tokens": CONFIG[k] = max(1, min(int(v), 128000))
        elif k == "temperature": CONFIG[k] = max(0.0, min(float(v), 2.0))
        elif k == "conversation_max_turns": CONFIG[k] = max(1, min(int(v), 500))
        elif k == "default_models": CONFIG[k] = list(v) if isinstance(v, list) else [str(v)]
        elif k == "multi_chat_timeout": CONFIG[k] = max(1, min(int(v), 120))
        else: CONFIG[k] = v
        updated.append(f"{k}={CONFIG[k]}" if k != "api_key" else f"{k}=***{str(v)[-4:]}")

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
    return "\n".join(lines)


async def _handle_models(client, args, progress_token):
    do_test = args.get("test", False)
    api_key = CONFIG.get("api_key", "")
    endpoint = CONFIG.get("endpoint", "")
    if not api_key or not endpoint: return "Error: api_key and endpoint required."

    lines = [f"Endpoint: {endpoint}\n", "[Deployed]"]
    deployment = CONFIG.get("deployment", "")
    if deployment:
        if do_test:
            text, _, latency = await _test_model(client, deployment, "deployed")
            status = f"OK ({latency:.0f}ms)" if not text.startswith("Error") else text[:60]
            lines.append(f"  {deployment}: {status}")
        else:
            lines.append(f"  {deployment} (current)")

    lines.append("\n[Serverless]")
    models = ["grok-3", "Meta-Llama-3.1-405B-Instruct", "DeepSeek-R1", "Phi-4"]
    for m in models:
        if do_test:
            text, _, latency = await _test_model(client, m, "serverless")
            status = f"OK ({latency:.0f}ms)" if not text.startswith("Error") else "unavailable"
            lines.append(f"  {m}: {status}")
        else:
            marker = " (current)" if m == CONFIG.get("model") else ""
            lines.append(f"  {m}{marker}")
    return "\n".join(lines)


async def _test_model(client, name, mtype):
    try:
        if mtype == "deployed":
            url = f"{CONFIG['endpoint']}/openai/deployments/{name}/chat/completions?api-version=2024-10-21"
            body = {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 10}
        else:
            url = f"{CONFIG['endpoint']}/models/chat/completions?api-version=2024-05-01-preview"
            body = {"messages": [{"role": "user", "content": "hi"}], "model": name, "max_tokens": 10}
        
        t0 = time.perf_counter()
        resp = await client.post(url, json=body, headers={"api-key": CONFIG["api_key"]}, timeout=15.0)
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200: return f"Error {resp.status_code}", {}, 0
        data = resp.json()
        return data["choices"][0]["message"]["content"], data.get("usage", {}), latency
    except Exception as e:
        return f"Error: {e}", {}, 0


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
