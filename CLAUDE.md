# Azure Chat Assistant — MCP Server

## What This Is
A lean MCP server that bridges local AI CLI tools to Azure AI Foundry models.
Provides `chat`, `multi_chat`, `configure`, `reset`, and `models` tools over JSON-RPC 2.0 stdio.

## Architecture
- **Runtime**: Python 3, asyncio + httpx (no SDK dependencies beyond httpx)
- **Streaming**: SSE with producer-consumer queue pattern for real-time token delivery
- **Concurrency**: `multi_chat` uses `asyncio.wait` to query multiple models in parallel with per-model timeout
- **Config**: `~/.config/azure-voice-assistant/config.json` — persists API key, endpoint, model settings
- **Protocol**: MCP v2024-11-05 over stdio

## Azure AI Foundry Endpoint Types
- **Deployed** (OpenAI-compat): `/openai/deployments/{name}/chat/completions` — used for GPT models
- **Serverless** (unified inference): `/models/chat/completions` — used for grok-3, Llama, DeepSeek, Phi

## Available Models (Azure Sponsorship)
| Model | Type | Notes |
|-------|------|-------|
| gpt-5.3-chat | deployed | Primary. Temperature must be 1.0 (rejects other values) |
| grok-3 | serverless | Free tier: 15 req/day. Often unavailable |
| Meta-Llama-3.1-405B-Instruct | serverless | Free tier: 15 req/day |
| DeepSeek-R1 | serverless | Outputs `<think>` tags despite instructions. Often unavailable |
| Phi-4 | serverless | Free tier: 15 req/day. Can be slow (cold start 30-70s) |

## Multi-Chat Features (v1.2.0)
- **Default models**: Configurable via `configure(default_models=[...])`. Defaults: gpt-5.3-chat, Llama 405B, Phi-4
- **Per-model timeout**: Configurable via `configure(multi_chat_timeout=15)`. Slow models get skipped cleanly
- **Per-model latency**: Each response shows `(1234ms)` timing
- **Wall time summary**: Footer shows total time + responded/timed-out counts
- **No history pollution**: multi_chat does not save to session DB (prevents echo between models)
- **Progress tracking**: Coordinated progress bar — 0% start, increments as each model completes

## 4-Way Voice Call Flow
This server pairs with [azure-speech](../speech-to-cli/) MCP server for voice I/O.
Optimized 2-call flow: `multi_chat` (parallel LLM) → `multi_speak` (parallel TTS).

### Step 1: multi_chat
```
multi_chat(message="What do you think about X?")
→ Queries all default_models in parallel
→ Returns combined responses with latency
```

### Step 2: multi_speak
```
multi_speak(segments=[
  {text: "Claude's intro", voice: "en-US-AvaNeural"},
  {text: "GPT response", voice: "en-US-DavisNeural"},
  {text: "Llama response", voice: "en-US-AndrewNeural"},
  {text: "Phi response", voice: "en-US-JennyNeural"}
])
→ All TTS fires in parallel, plays back-to-back
```

### Voice Assignments
| Model | Voice |
|-------|-------|
| Claude (host) | en-US-AvaNeural |
| GPT-5.3 | en-US-DavisNeural |
| Llama 405B | en-US-AndrewNeural |
| DeepSeek R1 | en-US-BrianNeural |
| Phi-4 | en-US-JennyNeural |

## Key Files
- `mcp_voice_assistant.py` — The MCP server (v1.2.0, async)
- `test_connection.py` — Connection test script for deployed/serverless models
- `WISDOM_SCROLL.md` — Refactoring history and council recommendations
- `RESEARCH_AND_PLAN.md` — Performance optimization roadmap

## Known Issues
- GPT-5.3 rejects `temperature` != 1.0 — the code only sends temperature when it differs from 1.0
- DeepSeek R1 includes `<think>` tags regardless of system prompt
- Serverless free-tier models have 15 req/day limit
- Grok-3 and DeepSeek-R1 frequently unavailable due to Azure sponsorship limits
- multi_chat does not save per-model history (by design — prevents cross-model echo)
