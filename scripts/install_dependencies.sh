#!/bin/bash

set -e

echo "=== Installing Validator Dependencies ==="

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

check_command() {
    if ! command -v $1 &> /dev/null; then
        log_warn "$1 not found"
        return 1
    else
        log_info "$1 already installed"
        return 0
    fi
}

# Update package lists
log_info "Updating package lists..."
apt update

# Install system dependencies
log_info "Installing system dependencies..."
apt install -y python3 python3-pip python3-venv python3-full pipx curl git build-essential

# Install OpenGL dependencies for OpenCV
log_info "Installing OpenGL dependencies for OpenCV..."
apt install -y libgl1 libgl1-mesa-dri libglib2.0-0

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
log_info "Using Python $PYTHON_VERSION"

# Install Node.js (required for PM2)
if ! check_command node; then
    log_info "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
    apt-get install -y nodejs
fi

# Install PM2 globally
if ! check_command pm2; then
    log_info "Installing PM2..."
    npm install pm2 -g
    pm2 update
fi

# Install btcli globally using pipx
log_info "Installing Bittensor CLI globally with pipx..."
pipx install --force "bittensor[cli]"

# Create symlink for btcli in pipx bin directory
ln -sf /root/.local/share/pipx/venvs/bittensor/bin/btcli /root/.local/bin/btcli

# Add pipx bin to PATH
export PATH="/root/.local/bin:$PATH"

# Add to bashrc for persistent PATH (avoid duplicates)
if ! grep -q "/root/.local/bin" ~/.bashrc; then
    echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
fi

# Verify btcli installation
if btcli --help &> /dev/null; then
    log_info "btcli installed successfully"
    btcli --version
else
    log_error "btcli installation failed"
    exit 1
fi

# Create Python virtual environment for project
log_info "Setting up Python virtual environment for project..."

if [ ! -d ".venv" ]; then
    log_info "Creating virtual environment..."
    python3 -m venv .venv
else
    log_info "Virtual environment already exists"
fi

# Activate virtual environment and install project dependencies
source .venv/bin/activate

log_info "Installing project Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt --no-cache

# Install project in editable mode
log_info "Installing project in editable mode..."
pip install -e .

# Test key imports
log_info "Testing Python package imports..."
python3 -c "import bittensor; print(f'Bittensor library: {bittensor.__version__}')"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import imagerecovery; print('Project package imported successfully')"

deactivate

log_info "=== Dependencies installation completed ==="
echo ""
log_info "Installed components:"
echo "  - System dependencies (Python3, Node.js, build tools)"
echo "  - PM2 process manager"
echo "  - Bittensor CLI (global)"
echo "  - Project Python environment (.venv)"
echo ""
log_info "Next steps:"
echo "  1. Register wallet: btcli wallet create"
echo "  2. Configure: .env file for your role"
echo "  3. Start validator: bash scripts/start_validator.sh"