#!/usr/bin/env bash
# Restart the WaferPulse service and tail the logs
echo "Restarting WaferPulse service..."
sudo systemctl restart waferpulse
echo "Service restarted. Streaming logs (Ctrl+C to exit):"
sudo journalctl -u waferpulse -f
