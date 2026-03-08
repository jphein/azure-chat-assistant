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
import json
import os
import sys
import time
import httpx

# ── Config ──────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.expanduser("~/.config/azure-voice-assistant/config.json")

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
}

CONFIG = {}
_conversation_history = []          # list of {"role": ..., "content": ...}
_stdout_lock = asyncio.Lock()

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
        await _send_progress(progress_token, 0.1, "Thinking...")

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
                if resp.status_code != 200:
                    body_text = await resp.aread()
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
                            # Fire and forget progress updates to avoid blocking the consumer loop
                            asyncio.create_task(_send_progress(progress_token, 0.5, f"...{preview}"))
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

async def chat(client, user_message, progress_token=None, model_override=None, model_type_override=None):
    """Send a message, get a response, maintain history."""
    global _conversation_history

    messages = []
    sys_prompt = CONFIG.get("system_prompt", "")
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.extend(_conversation_history)
    messages.append({"role": "user", "content": user_message})

    response, usage, latency = await call_llm(client, messages, progress_token, model_override, model_type_override)

    if not response.startswith("Error"):
        # For simplicity in multi-agent, we'll only append history in the main chat tool
        # or we could let the single chat handle history and multi_chat just return results.
        if not model_override:
            _conversation_history.append({"role": "user", "content": user_message})
            _conversation_history.append({"role": "assistant", "content": response})
            max_turns = CONFIG.get("conversation_max_turns", 50)
            while len(_conversation_history) > max_turns * 2:
                _conversation_history.pop(0)
                _conversation_history.pop(0)

    return response, usage, latency

async def multi_chat(client, user_message, models, progress_token=None):
    """Dispatch message to multiple models concurrently."""
    tasks = []
    
    for m in models:
        # Determine type based on name heuristically for this test
        m_type = "deployed" if "gpt" in m.lower() else "serverless"
        
        # We wrap the call so we know which model it came from
        async def _call_model(model_name, model_type):
            resp, usage, lat = await chat(client, user_message, progress_token, model_name, model_type)
            return model_name, resp, usage, lat
            
        tasks.append(asyncio.create_task(_call_model(m, m_type)))
        
    results = await asyncio.gather(*tasks)
    
    final_output = ""
    for name, resp, usage, lat in results:
        # Strip system messages or verbose tags from the output for a cleaner look
        final_output += f"**[{name}]**\n{resp}\n\n"
        
    # Append the combined request/response to history once
    global _conversation_history
    _conversation_history.append({"role": "user", "content": user_message})
    _conversation_history.append({"role": "assistant", "content": final_output.strip()})
    
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
        "description": "Send a message to multiple models concurrently and get a combined response.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The message to send."},
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of model names to query (e.g., ['gpt-5.3-chat', 'grok-3'])."
                }
            },
            "required": ["message", "models"],
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
            },
        },
    },
    {
        "name": "reset",
        "description": "Clear conversation history and start fresh.",
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
                "serverInfo": {"name": "azure-voice-assistant", "version": "1.1.0-async"},
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
        models = args.get("models", ["gpt-5.3-chat", "grok-3"])
        if not message:
            await _write_response(_result(req_id, "Error: 'message' is required."))
            return

        combined_response = await multi_chat(client, message, models, progress_token)
        await _write_response(_result(req_id, combined_response))

    elif tool_name == "configure":
        res = _handle_configure(args)
        await _write_response(_result(req_id, res))

    elif tool_name == "reset":
        global _conversation_history
        count = len(_conversation_history) // 2
        _conversation_history = []
        await _write_response(_result(req_id, f"Conversation cleared ({count} turns removed)."))

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
        "conversation_max_turns", "voice",
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
    await _write_response({
        "jsonrpc": "2.0", "method": "notifications/progress",
        "params": {"progressToken": token, "progress": progress, "total": 1.0, "message": message},
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

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
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
