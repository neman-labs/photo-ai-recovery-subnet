#!/bin/bash

set -e

echo "=== Testing Validator Installation ==="

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

test_command() {
    local cmd=$1
    local description=$2
    
    if command -v $cmd &> /dev/null; then
        log_info "✓ $description"
        return 0
    else
        log_error "✗ $description"
        return 1
    fi
}

test_python_import() {
    local module=$1
    local description=$2
    
    if python3 -c "import $module" &> /dev/null; then
        log_info "✓ $description"
        return 0
    else
        log_error "✗ $description"
        return 1
    fi
}

FAILED_TESTS=0

# Test system dependencies
log_info "Testing system dependencies..."
test_command python3 "Python3 installed" || ((FAILED_TESTS++))
test_command pip3 "Pip3 installed" || ((FAILED_TESTS++))
test_command git "Git installed" || ((FAILED_TESTS++))
test_command curl "Curl installed" || ((FAILED_TESTS++))
test_command node "Node.js installed" || ((FAILED_TESTS++))
test_command npm "NPM installed" || ((FAILED_TESTS++))
test_command pm2 "PM2 installed" || ((FAILED_TESTS++))
test_command btcli "Bittensor CLI installed" || ((FAILED_TESTS++))

# Test btcli functionality
log_info "Testing btcli functionality..."
if btcli --help &> /dev/null; then
    log_info "✓ btcli responds to commands"
    btcli --version
else
    log_error "✗ btcli not responding"
    ((FAILED_TESTS++))
fi

# Test virtual environment
log_info "Testing Python virtual environment..."
if [ -d ".venv" ]; then
    log_info "✓ Virtual environment exists"
    
    # Activate and test imports
    source .venv/bin/activate
    
    test_python_import bittensor "Bittensor library import" || ((FAILED_TESTS++))
    test_python_import torch "PyTorch import" || ((FAILED_TESTS++))
    test_python_import numpy "NumPy import" || ((FAILED_TESTS++))
    test_python_import pillow "Pillow import" || ((FAILED_TESTS++))
    test_python_import imagerecovery "Project package import" || ((FAILED_TESTS++))
    
    # Test specific versions
    log_info "Checking package versions..."
    python3 -c "import bittensor; print(f'Bittensor: {bittensor.__version__}')"
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
    
    deactivate
else
    log_error "✗ Virtual environment not found"
    ((FAILED_TESTS++))
fi

# Test configuration files
log_info "Testing configuration files..."

# Check validator config
if [ -f "validator.env.example" ]; then
    log_info "✓ validator.env.example exists"
else
    log_error "✗ validator.env.example missing"
    ((FAILED_TESTS++))
fi

if [ -f "validator.env" ]; then
    log_info "✓ validator.env exists"
    
    # Test if required variables are set
    source validator.env
    
    if [ -n "$NETUID" ]; then
        log_info "✓ NETUID configured ($NETUID)"
    else
        log_warn "⚠ NETUID not configured"
    fi
    
    if [ -n "$WALLET_NAME" ]; then
        log_info "✓ WALLET_NAME configured ($WALLET_NAME)"
    else
        log_warn "⚠ WALLET_NAME not configured"
    fi
    
    if [ -n "$WALLET_HOTKEY" ]; then
        log_info "✓ WALLET_HOTKEY configured ($WALLET_HOTKEY)"
    else
        log_warn "⚠ WALLET_HOTKEY not configured"
    fi
    
else
    log_warn "⚠ validator.env not found (copy from validator.env.example)"
fi

# Check miner config
if [ -f "miner.env.example" ]; then
    log_info "✓ miner.env.example exists"
else
    log_error "✗ miner.env.example missing"
    ((FAILED_TESTS++))
fi

if [ -f "miner.env" ]; then
    log_info "✓ miner.env exists"
    
    # Test miner-specific variables
    source miner.env
    
    if [ -n "$AXON_PORT" ]; then
        log_info "✓ AXON_PORT configured ($AXON_PORT)"
    else
        log_warn "⚠ AXON_PORT not configured"
    fi
    
    if [ -n "$BLACKLIST_FORCE_VALIDATOR_PERMIT" ]; then
        log_info "✓ BLACKLIST_FORCE_VALIDATOR_PERMIT configured ($BLACKLIST_FORCE_VALIDATOR_PERMIT)"
    else
        log_warn "⚠ BLACKLIST_FORCE_VALIDATOR_PERMIT not configured"
    fi
    
else
    log_warn "⚠ miner.env not found (copy from miner.env.example)"
fi

# Test script permissions
log_info "Testing script permissions..."
if [ -x "scripts/install_dependencies.sh" ]; then
    log_info "✓ install_dependencies.sh executable"
else
    log_warn "⚠ install_dependencies.sh not executable"
fi

if [ -x "scripts/start_validator.sh" ]; then
    log_info "✓ start_validator.sh executable"
else
    log_warn "⚠ start_validator.sh not executable"
fi

if [ -x "scripts/start_miner.sh" ]; then
    log_info "✓ start_miner.sh executable"
else
    log_warn "⚠ start_miner.sh not executable"
fi

# Summary
echo ""
if [ $FAILED_TESTS -eq 0 ]; then
    log_info "=== All tests passed! Installation appears successful ==="
    echo ""
    log_info "Ready to start validator!"
    echo ""
    log_info "Next steps:"
    echo "  1. Configure wallet if not done: see WALLET_SETUP.md"
    echo "  2. Configure environment files:"
    echo "     - For validator: cp validator.env.example validator.env"
    echo "     - For miner: cp miner.env.example miner.env"
    echo "  3. Start nodes:"
    echo "     - Validator: bash scripts/start_validator.sh"
    echo "     - Miner: bash scripts/start_miner.sh"
else
    log_error "=== $FAILED_TESTS test(s) failed ==="
    echo ""
    log_error "Please fix the issues above before starting the validator"
    echo ""
    log_info "Common solutions:"
    echo "  - Run: bash scripts/install_dependencies.sh"
    echo "  - Copy config files:"
    echo "    cp validator.env.example validator.env"
    echo "    cp miner.env.example miner.env"
    echo "  - Make executable: chmod +x scripts/*.sh"
fi

exit $FAILED_TESTS