#!/bin/bash
# Install a macOS LaunchAgent that runs the daily lineup refresh at 10:30am local.
# Requires this repo path and a working .venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.lineupintelligence.dailyrefresh"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
PYTHON="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/data/logs"
mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON — create the venv first:"
  echo "  cd $ROOT && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>WorkingDirectory</key>
  <string>${ROOT}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${PYTHON}</string>
    <string>${ROOT}/scripts/daily_refresh.py</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONPATH</key>
    <string>${ROOT}/backend</string>
  </dict>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>10</integer>
    <key>Minute</key>
    <integer>30</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>${LOG_DIR}/daily_refresh.out.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/daily_refresh.err.log</string>
  <key>RunAtLoad</key>
  <false/>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl enable "gui/$(id -u)/${LABEL}"

echo "Installed ${PLIST}"
echo "Runs daily at 10:30am local. Logs: ${LOG_DIR}/daily_refresh.*.log"
echo "Unload later with:"
echo "  launchctl bootout gui/\$(id -u)/${LABEL}"
