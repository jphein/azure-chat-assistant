# cloud-chat-assistant

Multi-cloud MCP server — talk to models on Azure AI Foundry, AWS Bedrock, and Google Vertex AI from any AI CLI agent.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey)

## What it does

Exposes cloud AI models as MCP tools so AI CLI agents (Claude Code, Gemini CLI, Copilot CLI) can query them programmatically. Supports streaming, conversation history, parallel multi-model queries, and dynamic model discovery via CLIs.

| Tool | Description |
|------|-------------|
| **chat** | Send a message, get a streaming response with conversation history |
| **multi_chat** | Query multiple models concurrently, get combined results |
| **scan** | Test all models across all providers, show availability matrix |
| **configure** | View/change settings (model, provider, credentials, etc.) |
| **models** | List available models and test connectivity |
| **reset** | Clear conversation history |

## Supported Providers

| Provider | Model Types | Auth |
|----------|-------------|------|
| **Azure AI Foundry** | GPT-5.x, o1/o3/o4, Llama, DeepSeek, Phi, Grok, Mistral, Claude | API key |
| **AWS Bedrock** | Claude 4.x, Nova, Llama 4, Writer Palmyra | Access key + secret |
| **Google Vertex AI** | Gemini 2.5/3.x | API key or gcloud auth |

## Quick start

### Prerequisites

- Python 3.8+
- At least one cloud provider configured

### Install

```bash
git clone https://github.com/jphein/cloud-chat-assistant.git
cd cloud-chat-assistant
python3 -m venv venv
./venv/bin/pip install httpx
```

### Configure

The server auto-creates `~/.config/cloud-chat-assistant/` on first run.

**Environment variables** (recommended):
```bash
# Azure AI Foundry
export AZURE_AI_API_KEY="your-azure-key"
export AZURE_AI_ENDPOINT="https://your-resource.services.ai.azure.com"

# AWS Bedrock
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# Google Vertex AI
export GOOGLE_API_KEY="your-vertex-ai-key"
export GOOGLE_PROJECT="your-gcp-project-id"
export GOOGLE_REGION="global"
```

**Or config file** (`~/.config/cloud-chat-assistant/config.json`):
```json
{
    "api_key": "your-azure-key",
    "endpoint": "https://your-resource.services.ai.azure.com",
    "deployment": "gpt-5.3-chat",
    "model_type": "deployed",
    "aws_access_key": "your-access-key",
    "aws_secret_key": "your-secret-key",
    "aws_region": "us-east-1",
    "google_api_key": "your-vertex-ai-key",
    "google_project": "your-gcp-project-id"
}
```

### Register with your CLI agent

**Claude Code** — add to `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "cloud-chat": {
      "command": "python3",
      "args": ["/path/to/cloud-chat-assistant/mcp_cloud_chat.py"]
    }
  }
}
```

**Gemini CLI** — add to `~/.gemini/settings.json` under `mcpServers`.

**Copilot CLI** — add to `~/.copilot/mcp.json` under `mcpServers`.

## Usage Examples

### Switch providers
```
configure(model_type="bedrock", deployment="claude-opus-4.6")
configure(model_type="deployed", deployment="gpt-5.3-chat")
configure(model_type="serverless", deployment="Meta-Llama-3.1-405B-Instruct")
```

### Multi-model queries
```
multi_chat(message="Explain quantum entanglement", models=["gpt-5.3-chat", "claude-opus-4.6", "gemini-3.1-pro-preview"])
```

### Scan all providers
```
scan()
```
Returns a matrix showing which models are working, unavailable, or deployable.

## CLI Integration (Optional)

Install cloud CLIs for dynamic model discovery:

```bash
# Azure — list deployable models
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az login

# AWS — list Bedrock models
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install
aws configure

# Google — auth tokens for Vertex AI
sudo apt install google-cloud-cli
gcloud auth login
```

See [CLI_SETUP.md](CLI_SETUP.md) for detailed instructions.

## Ecosystem

This project is part of a four-project voice AI system:

| Project | Role |
|---------|------|
| [speech-to-cli](https://github.com/jphein/speech-to-cli) | Audio engine — STT, TTS, VAD, recorder |
| **cloud-chat-assistant** (this) | Multi-cloud LLM provider |
| [gnome-speaks](https://github.com/jphein/gnome-speaks) | GNOME Shell extension — desktop voice UI |
| [the-oracle](https://github.com/jphein/the-oracle) | Web frontend — proxies both MCP servers |

### Voice Integration

Pair with [speech-to-cli](https://github.com/jphein/speech-to-cli) for voice conversations:

1. `multi_chat` — queries all models in parallel
2. `multi_speak` — synthesizes all responses, plays sequentially

### GNOME Speaks Integration

[gnome-speaks](https://github.com/jphein/gnome-speaks) can call cloud-chat-assistant directly for AI conversation mode, and its preferences panel can configure this project's settings (`~/.config/cloud-chat-assistant/config.json`) — including provider credentials, generation parameters, and model selection — from a unified GNOME settings UI.

## Architecture

- **Async**: `asyncio` + `httpx` with connection pooling
- **Streaming**: SSE with producer-consumer queue
- **Protocol**: MCP v2024-11-05 over stdio, JSON-RPC 2.0
- **Config**: Auto-migrates from old `azure-chat-assistant` location

## License

GPLv3 — see [LICENSE](LICENSE).
