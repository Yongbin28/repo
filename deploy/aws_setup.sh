#!/usr/bin/env bash
# =============================================================================
# WaferPulse — Automated Cloud EC2 Setup Script (Ubuntu 22.04 / 24.04 LTS)
# =============================================================================
# Run ON the EC2 instance from the unzipped waferpulse_app folder:
#     cd ~/waferpulse_app && bash deploy/aws_setup.sh
#
# What it does:
#   1. Installs Python 3.11 and system build dependencies
#   2. Sets up isolated Python virtual environment (.venv)
#   3. Installs all required ML & Dashboard dependencies from requirements.txt
#   4. Registers a systemd service (auto-start on boot, auto-restart on crash)
#   5. Launches the WaferPulse server on port 8501
# =============================================================================
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "== WaferPulse Cloud EC2 Setup =="
echo "   Application Directory: $APP_DIR"

# ── 1. System packages ───────────────────────────────────────────────────────
echo "--> Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y software-properties-common curl unzip build-essential

# Ensure Python 3.11 is installed
if ! command -v python3.11 >/dev/null 2>&1; then
    echo "--> Installing Python 3.11..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa || true
    sudo apt-get update -y
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

# ── 2. Python virtual environment ────────────────────────────────────────────
cd "$APP_DIR"
if [ ! -d ".venv" ]; then
    echo "--> Creating Python 3.11 virtual environment..."
    python3.11 -m venv .venv
fi

source .venv/bin/activate
echo "--> Upgrading pip and installing wheels..."
pip install --upgrade pip wheel setuptools

echo "--> Installing WaferPulse dependencies (this takes 2-3 minutes)..."
pip install -r requirements.txt

# ── 3. Configure systemd service ─────────────────────────────────────────────
SVC=/etc/systemd/system/waferpulse.service
echo "--> Configuring 24/7 systemd daemon ($SVC)..."

sudo tee $SVC >/dev/null <<EOF
[Unit]
Description=WaferPulse Semiconductor AI Quality Risk Detection Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
Environment="PYTHONUNBUFFERED=1"
Environment="STREAMLIT_SERVER_PORT=8501"
Environment="STREAMLIT_SERVER_HEADLESS=true"
Environment="STREAMLIT_SERVER_ENABLE_CORS=false"
Environment="STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false"
ExecStart=$APP_DIR/.venv/bin/python -m streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=10
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

# ── 4. Enable and start service ──────────────────────────────────────────────
sudo systemctl daemon-reload
sudo systemctl enable waferpulse
sudo systemctl restart waferpulse

PUBLIC_IP=$(curl -s http://checkip.amazonaws.com || echo "<YOUR-EC2-PUBLIC-IP>")

echo ""
echo "========================================================================="
echo "🎉 WaferPulse is LIVE and running 24/7 on AWS!"
echo "   Public URL: http://${PUBLIC_IP}:8501"
echo ""
echo "   Default Sign-In Accounts:"
echo "     - Administrator:     admin / admin123"
echo "     - Process Engineer:  engineer / engineer123"
echo "     - Quality Auditor:   auditor / auditor123"
echo ""
echo "   Useful Commands:"
echo "     - Check status:  sudo systemctl status waferpulse"
echo "     - Live logs:     sudo journalctl -u waferpulse -f"
echo "     - Restart:       sudo systemctl restart waferpulse"
echo "========================================================================="
