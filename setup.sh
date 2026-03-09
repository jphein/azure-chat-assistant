#!/bin/bash
# Setup script for azure-chat-assistant MCP server
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$HOME/.config/azure-chat-assistant"

echo "Setting up azure-chat-assistant..."

# Create venv and install dependencies
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
fi

echo "Installing dependencies..."
"$SCRIPT_DIR/venv/bin/pip" install -q httpx

# Create config directory
mkdir -p "$CONFIG_DIR"

# Create default config if it doesn't exist
if [ ! -f "$CONFIG_DIR/config.json" ]; then
    echo "Creating default config at $CONFIG_DIR/config.json"
    cat > "$CONFIG_DIR/config.json" << 'EOF'
{
    "api_key": "",
    "endpoint": "https://claud-assistant-resource.services.ai.azure.com",
    "deployment": "gpt-5.3-chat",
    "model": "gpt-5.3-chat-2026-03-03",
    "model_type": "deployed",
    "max_completion_tokens": 2048,
    "temperature": 1.0,
    "system_prompt": "You are a helpful chat assistant. Keep responses concise and conversational.",
    "conversation_max_turns": 50,
    "default_models": ["gpt-5.3-chat", "Meta-Llama-3.1-405B-Instruct", "Phi-4"],
    "multi_chat_timeout": 15
}
EOF
    echo ""
    echo ">>> IMPORTANT: Edit $CONFIG_DIR/config.json and add your Azure AI API key <<<"
    echo ""
else
    echo "Config already exists at $CONFIG_DIR/config.json"
fi

# Print MCP registration snippet
PYTHON_PATH="$SCRIPT_DIR/venv/bin/python3"
SERVER_PATH="$SCRIPT_DIR/mcp_chat_assistant.py"

echo ""
echo "Setup complete! Add this to your ~/.claude.json mcpServers:"
echo ""
echo "    \"azure-chat-assistant\": {"
echo "      \"type\": \"stdio\","
echo "      \"command\": \"$PYTHON_PATH\","
echo "      \"args\": [\"$SERVER_PATH\"],"
echo "      \"env\": {}"
echo "    }"
echo ""
