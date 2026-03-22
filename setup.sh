#!/bin/bash
# Setup script for cloud-chat-assistant MCP server
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$HOME/.config/cloud-chat-assistant"

echo "Setting up cloud-chat-assistant..."

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
    "endpoint": "",
    "deployment": "",
    "model": "",
    "model_type": "deployed",
    "google_api_key": "",
    "google_project": "",
    "google_region": "global"
}
EOF
    echo ""
    echo ">>> IMPORTANT: Configure via MCP configure() tool or set env vars: <<<"
    echo ">>>   AZURE_AI_API_KEY, AZURE_AI_ENDPOINT, GOOGLE_API_KEY, GOOGLE_PROJECT <<<"
    echo ""
else
    echo "Config already exists at $CONFIG_DIR/config.json"
fi

# Print MCP registration snippet
PYTHON_PATH="$SCRIPT_DIR/venv/bin/python3"
SERVER_PATH="$SCRIPT_DIR/mcp_cloud_chat.py"

echo ""
echo "Setup complete! Add this to your ~/.claude.json mcpServers:"
echo ""
echo "    \"cloud-chat-assistant\": {"
echo "      \"type\": \"stdio\","
echo "      \"command\": \"$PYTHON_PATH\","
echo "      \"args\": [\"$SERVER_PATH\"],"
echo "      \"env\": {}"
echo "    }"
echo ""
