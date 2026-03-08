# azure-chat-assistant

MCP server for Azure AI Foundry — talk to GPT-5.3, grok-3, Llama, DeepSeek, and Phi from any AI CLI agent.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)

## What it does

Exposes Azure AI models as MCP tools so AI CLI agents (Claude Code, Gemini CLI, Copilot CLI) can query them programmatically. Supports streaming, conversation history, and parallel multi-model queries.

| Tool | Description |
|------|-------------|
| **chat** | Send a message, get a streaming response with conversation history |
| **multi_chat** | Query multiple models concurrently, get combined results |
| **configure** | View/change settings (model, API key, endpoint, temperature, etc.) |
| **reset** | Clear conversation history |
| **models** | List available models and test connectivity |

## Quick start

### Prerequisites

- Python 3.8+
- An Azure AI Foundry resource with API key
- `httpx` (`pip install httpx`)

### Configure

Create `~/.config/azure-voice-assistant/config.json`:

```json
{
    "api_key": "your-azure-ai-key",
    "endpoint": "https://your-resource.services.ai.azure.com",
    "deployment": "gpt-5.3-chat",
    "model_type": "deployed"
}
```

### Register with your CLI agent

**Claude Code** — add to your project or global `.claude.json`:

```json
{
  "mcpServers": {
    "azure-chat-assistant": {
      "command": "python3",
      "args": ["/path/to/azure-chat-assistant/mcp_voice_assistant.py"]
    }
  }
}
```

**Gemini CLI** — add to `~/.gemini/settings.json` under `mcpServers`.

**Copilot CLI** — add to `~/.copilot/mcp.json` under `mcpServers`.

### Multi-model conversations

Use `multi_chat` to query multiple models at once:

```
multi_chat(message="What is consciousness?", models=["gpt-5.3-chat", "grok-3", "Meta-Llama-3.1-405B-Instruct"])
```

Each model's response is labeled with `**[model-name]**` headers.

### Voice integration

Pair with the [azure-speech](https://github.com/jphein/speech-to-cli) MCP server for full voice conversations. The optimized flow uses just 2 MCP calls:

1. `multi_chat` — queries all models in parallel
2. `multi_speak` — synthesizes all responses in parallel, plays sequentially

## Architecture

- **Async core**: `asyncio` + `httpx` with persistent connection pooling
- **Streaming**: SSE with producer-consumer queue for real-time token delivery
- **Two endpoint types**: "deployed" (OpenAI-compat) for GPT, "serverless" (unified) for everything else
- **MCP v2024-11-05**: JSON-RPC 2.0 over stdio

## License

GPLv3 — see [LICENSE](LICENSE).
