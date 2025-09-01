#!/bin/bash

set -e

echo "=== Starting Validator ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if dependencies are installed
if [ ! -d ".venv" ]; then
    log_error "Virtual environment not found. Run: bash scripts/install_dependencies.sh"
    exit 1
fi

if ! command -v pm2 &> /dev/null; then
    log_error "PM2 not found. Run: bash scripts/install_dependencies.sh"
    exit 1
fi

# Check if validator.env exists
if [ ! -f "validator.env" ]; then
    log_error "validator.env not found. Copy from validator.env.example and configure it."
    exit 1
fi

# Load environment variables
log_info "Loading configuration..."
set -a
source validator.env
set +a


REQUIRED_ENV_VARS=(
  "NETUID"
  "SUBTENSOR_NETWORK"
  "WALLET_NAME"
  "WALLET_HOTKEY"
)

# Verify required environment variables
log_info "Verifying configuration..."
MISSING_VARS=0
for VAR in "${REQUIRED_ENV_VARS[@]}"; do
  if [ -z "${!VAR}" ]; then
    log_error "Missing required environment variable: $VAR"
    MISSING_VARS=1
  else
    log_info "$VAR = ${!VAR}"
  fi
done

if [ "$MISSING_VARS" = 1 ]; then
  log_error "Please configure missing variables in validator.env"
  exit 1
fi

PROCESS_NAME="sn$NETUID-$SUBTENSOR_NETWORK-validator-$WALLET_NAME-$WALLET_HOTKEY"

# Stop existing process if running
if pm2 list | grep -q "$PROCESS_NAME"; then
  log_warn "Process '$PROCESS_NAME' is already running. Stopping it..."
  pm2 delete $PROCESS_NAME
fi

# Activate virtual environment
log_info "Activating Python environment..."
source .venv/bin/activate

# Setup WANDB if API key is provided
if [ -n "$WANDB_API_KEY" ]; then
  log_info "Setting up WANDB logging..."
  wandb login $WANDB_API_KEY --relogin
fi

log_info "Starting validator process..."

# Initialize the base command
CMD="pm2 start neurons/validator.py --name $PROCESS_NAME --"

# Add mandatory arguments
CMD+=" --netuid $NETUID"
CMD+=" --subtensor.network $SUBTENSOR_NETWORK"
CMD+=" --wallet.name $WALLET_NAME"
CMD+=" --wallet.hotkey $WALLET_HOTKEY"
CMD+=" --logging.trace"

# Conditionally add optional arguments
[ -n "$SUBTENSOR_CHAIN_ENDPOINT" ] && CMD+=" --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT"
[ -n "$WANDB_PROJECT" ] && CMD+=" --wandb.project $WANDB_PROJECT"
[ -n "$WANDB_ENTITY" ] && CMD+=" --wandb.entity $WANDB_ENTITY"
[ -n "$AXON_PORT" ] && CMD+=" --axon.port $AXON_PORT"

# Execute the constructed command
log_info "Executing command: $CMD"
eval "$CMD"

# Start auto-updater process
AUTO_UPDATE_PROCESS_NAME="auto_update_monitor"

if ! pm2 list | grep -q "$AUTO_UPDATE_PROCESS_NAME"; then
  log_info "Starting auto-updater process..."
  chmod +x scripts/auto_update.sh
  pm2 start scripts/auto_update.sh --name $AUTO_UPDATE_PROCESS_NAME
else
  log_info "Auto-updater process is already running"
fi

# Show status
log_info "Process status:"
pm2 list

echo ""
log_info "=== Validator started successfully ==="
echo ""
log_info "Monitoring commands:"
echo "  pm2 monit"
echo "  pm2 logs $PROCESS_NAME"
echo "  pm2 logs $AUTO_UPDATE_PROCESS_NAME"
echo ""
log_info "Stop commands:"
echo "  pm2 stop $PROCESS_NAME"
echo "  pm2 stop $AUTO_UPDATE_PROCESS_NAME"
