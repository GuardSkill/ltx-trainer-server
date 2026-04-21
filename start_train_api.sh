#!/usr/bin/env bash
# Start LTX Trainer API server (port 8777) and FRP tunnel
set -e

SERVER_DIR="/root/lisiyuan/LTX-2/packages/ltx-trainer-server"
TRAINER_DIR="/root/lisiyuan/LTX-2/packages/ltx-trainer"
FRP_DIR="/usr/local/frp_0.55.1_linux_amd64"
FRP_CONFIG="$FRP_DIR/frpc_LTX2_3_train_api.toml"

cd "$TRAINER_DIR"   # uv must run from ltx-trainer (has pyproject.toml + .venv)

echo "[$(date)] Starting LTX Trainer API..."

# Start FRP tunnel in background
if pgrep -f "frpc.*train_api" > /dev/null 2>&1; then
    echo "[$(date)] FRP tunnel already running"
else
    nohup "$FRP_DIR/frpc" -c "$FRP_CONFIG" >> /var/log/frpc_train_api.log 2>&1 &
    echo "[$(date)] FRP tunnel started (pid=$!)"
fi

# Start API server (script lives in ltx-trainer-server, but uv env is from ltx-trainer)
exec uv run python "$SERVER_DIR/ltx_api_server.py"
