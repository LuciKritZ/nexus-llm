#!/usr/bin/env bash

set -e

echo "This script will install nexus-llm as a background macOS service (LaunchAgent)."
read -p "Do you want to proceed? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This service installation is only supported on macOS."
    exit 1
fi

PLIST_PATH="$HOME/Library/LaunchAgents/com.lucikritz.nexus-llm.plist"
PROJECT_DIR="$(pwd)"
UV_PATH="$(which uv)"

# Create the logs directory
mkdir -p "$PROJECT_DIR/.logs"

if [ -z "$UV_PATH" ]; then
    echo "Error: uv command not found in PATH."
    exit 1
fi

# Create the plist
cat <<EOF > "$PLIST_PATH"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lucikritz.nexus-llm</string>
    <key>ProgramArguments</key>
    <array>
        <string>$UV_PATH</string>
        <string>run</string>
        <string>python</string>
        <string>-m</string>
        <string>nexus_llm</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/.logs/nexus-llm.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/.logs/nexus-llm.err.log</string>
</dict>
</plist>
EOF

echo "Created LaunchAgent at $PLIST_PATH"

# Setup aliases
SHELL_RC="$HOME/.zshrc"
if [[ "$SHELL" == *"bash"* ]]; then
    SHELL_RC="$HOME/.bashrc"
fi

if ! grep -q "nexus-llm-start" "$SHELL_RC"; then
    echo "" >> "$SHELL_RC"
    echo "# nexus-llm aliases" >> "$SHELL_RC"
    echo "alias nexus-llm-start=\"launchctl bootstrap gui/\\\$(id -u) $PLIST_PATH\"" >> "$SHELL_RC"
    echo "alias nexus-llm-stop=\"launchctl bootout gui/\\\$(id -u) $PLIST_PATH\"" >> "$SHELL_RC"
    echo "alias nexus-llm-log=\"tail -f $PROJECT_DIR/.logs/nexus-llm.log\"" >> "$SHELL_RC"
    echo "Aliases added to $SHELL_RC. Run 'source $SHELL_RC' to load them."
else
    echo "Aliases already exist in $SHELL_RC"
fi

echo "Installation complete! Run 'source $SHELL_RC' then 'nexus-llm-start' to launch the service."
