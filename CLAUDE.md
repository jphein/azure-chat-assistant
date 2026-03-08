# Azure Chat Assistant — MCP Server

## What This Is
A lean MCP server that bridges local AI CLI tools to Azure AI Foundry models.
Provides `chat`, `multi_chat`, `configure`, `reset`, and `models` tools over JSON-RPC 2.0 stdio.

## Architecture
- **Runtime**: Python 3, asyncio + httpx (no SDK dependencies beyond httpx)
- **Streaming**: SSE with producer-consumer queue pattern for real-time token delivery
- **Concurrency**: `multi_chat` uses `asyncio.gather` to query multiple models in parallel
- **Config**: `~/.config/azure-voice-assistant/config.json` — persists API key, endpoint, model settings
- **Protocol**: MCP v2024-11-05 over stdio

## Azure AI Foundry Endpoint Types
- **Deployed** (OpenAI-compat): `/openai/deployments/{name}/chat/completions` — used for GPT models
- **Serverless** (unified inference): `/models/chat/completions` — used for grok-3, Llama, DeepSeek, Phi

## Available Models (Azure Sponsorship)
| Model | Type | Notes |
|-------|------|-------|
| gpt-5.3-chat | deployed | Primary. Temperature must be 1.0 (rejects other values) |
| grok-3 | serverless | Free tier: 15 req/day |
| Meta-Llama-3.1-405B-Instruct | serverless | Free tier: 15 req/day |
| DeepSeek-R1 | serverless | Outputs `<think>` tags despite instructions. Free tier |
| Phi-4 | serverless | Free tier: 15 req/day |

## Voice Conversation Setup
This server pairs with [azure-speech](../speech-to-cli/) MCP server for voice I/O.
Optimized 2-call flow: `multi_chat` (parallel LLM) → `multi_speak` (parallel TTS).

### Voice Assignments
| Model | Voice |
|-------|-------|
| GPT-5.3 | en-US-DavisNeural |
| Llama 405B | en-US-AndrewNeural |
| DeepSeek R1 | en-US-BrianNeural |
| Claude (host) | en-US-AvaNeural |

## Key Files
- `mcp_voice_assistant.py` — The MCP server (530 lines, async)
- `test_connection.py` — Connection test script for deployed/serverless models
- `WISDOM_SCROLL.md` — Refactoring history and council recommendations
- `RESEARCH_AND_PLAN.md` — Performance optimization roadmap

## Known Issues
- GPT-5.3 rejects `temperature` != 1.0 — the code only sends temperature when it differs from 1.0
- DeepSeek R1 includes `<think>` tags regardless of system prompt
- Serverless free-tier models have 15 req/day limit
- `multi_chat` doesn't save per-model history, only the combined result
