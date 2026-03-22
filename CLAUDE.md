<!-- claude-md-version: c3113ea | updated: 2026-03-22 -->
# Cloud Chat Assistant — MCP Server

Multi-cloud MCP server for Claude Code — connects to Azure AI Foundry, AWS Bedrock, Google Vertex AI, and DigitalOcean.

## File Map
- `mcp_cloud_chat.py` — The entire server (~1700 lines, async)
- `setup.sh` — Creates venv, installs deps, prints MCP registration snippet
- `CLI_SETUP.md` — Installation guide for az, aws, gcloud CLIs
- `test_connection.py` — Standalone connection test (uses urllib, not async)

## Do NOT Read
- `venv/` — Python virtualenv, never modify
- `WISDOM_SCROLL.md`, `FORWARD_GUIDE.md`, `RESEARCH_AND_PLAN.md` — Reference docs only

## Architecture
- **Runtime**: Python 3, asyncio + httpx (only external dep)
- **Config**: `~/.config/cloud-chat-assistant/config.json`
- **Sessions**: SQLite at `~/.config/cloud-chat-assistant/sessions.db`
- **Protocol**: MCP v2024-11-05 over stdio, JSON-RPC 2.0

## Providers
| Provider | Type | Models |
|----------|------|--------|
| Azure AI Foundry | deployed | GPT-5.x, o1, o3, o4 (require deployment) |
| Azure AI Foundry | serverless | Llama, DeepSeek, Phi, Grok, Cohere, Mistral |
| AWS Bedrock | bedrock | Claude 4.x, Nova, Llama 4, Writer |
| Google Vertex | google | Gemini 2.5/3.x |
| DigitalOcean | digitalocean | Claude, GPT, Llama, Mistral, DeepSeek, Qwen, etc. |
| Puter | puter | Claude 4.x, GPT-5.x, o3/o4, DeepSeek, Grok, Gemini, Mistral |

## MCP Tools (12)
`chat`, `multi_chat`, `configure`, `reset`, `clear_cache`, `status`, `models`, `scan`, `create_session`, `switch_session`, `delete_session`, `list_sessions`

## CLI Integration
The `scan` tool uses CLIs for dynamic model discovery:

| CLI | Purpose |
|-----|---------|
| `az` | List deployable/deployed Azure models |
| `aws` | List Bedrock foundation models |
| `gcloud` | Auth tokens for Vertex AI |

See `CLI_SETUP.md` for installation.

## Env Vars
`AZURE_AI_API_KEY`, `AZURE_AI_ENDPOINT`, `GOOGLE_API_KEY`, `GOOGLE_PROJECT`, `GOOGLE_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `DIGITALOCEAN_API_KEY`, `PUTER_API_KEY`

## How to Run
```bash
./venv/bin/python3 mcp_cloud_chat.py   # starts MCP stdio server
```

## Voice Integration
Pairs with `../speech-to-cli/` MCP server. Flow: `multi_chat` -> `multi_speak`.
