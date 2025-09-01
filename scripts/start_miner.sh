#!/bin/bash

set -e

echo "=== Starting Miner ==="

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

# Get env file (default: miner.env)
ENV_FILE=${1:-miner.env}

# Check if miner.env exists
if [ ! -f "$ENV_FILE" ]; then
    log_error "$ENV_FILE not found. Copy from miner.env.example and configure it."
    exit 1
fi

# Load environment variables
log_info "Loading configuration from $ENV_FILE..."
set -a
source "$ENV_FILE"
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
  log_error "Please configure missing variables in $ENV_FILE"
  exit 1
fi

PROCESS_NAME="sn$NETUID-$SUBTENSOR_NETWORK-miner-$WALLET_NAME-$WALLET_HOTKEY"

# Stop existing process if running
if pm2 list | grep -q "$PROCESS_NAME"; then
  log_warn "Process '$PROCESS_NAME' is already running. Stopping it..."
  pm2 delete $PROCESS_NAME
fi

# Activate virtual environment
log_info "Activating Python environment..."
source .venv/bin/activate

log_info "Starting miner process..."

CMD="pm2 start neurons/miner.py --name $PROCESS_NAME --"

# Add mandatory arguments
CMD+=" --netuid $NETUID"
CMD+=" --subtensor.network $SUBTENSOR_NETWORK"
CMD+=" --wallet.name $WALLET_NAME"
CMD+=" --wallet.hotkey $WALLET_HOTKEY"
CMD+=" --logging.info"

# Conditionally add optional arguments
[ -n "$AXON_PORT" ] && CMD+=" --axon.port $AXON_PORT"
[ -n "$SUBTENSOR_CHAIN_ENDPOINT" ] && CMD+=" --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT"
[ -n "$BLACKLIST_FORCE_VALIDATOR_PERMIT" ] && CMD+=" --blacklist.force_validator_permit $BLACKLIST_FORCE_VALIDATOR_PERMIT"
[ -n "$BLACKLIST_VALIDATOR_MIN_STAKE" ] && CMD+=" --blacklist.validator_min_stake $BLACKLIST_VALIDATOR_MIN_STAKE"

# Execute the constructed command
log_info "Executing command: $CMD"
eval "$CMD"

# Show status
log_info "Process status:"
pm2 list

echo ""
log_info "=== Miner started successfully ==="
echo ""
log_info "Monitoring commands:"
echo "  pm2 monit"
echo "  pm2 logs $PROCESS_NAME"
echo ""
log_info "Stop commands:"
echo "  pm2 stop $PROCESS_NAME"
