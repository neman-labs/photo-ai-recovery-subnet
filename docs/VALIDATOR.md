## Overview

**What you need:**
- Ubuntu server (20.04+ recommended)

**Time required:** ~15-30 minutes

## Step 1: Download Repository

```bash
git clone https://github.com/neman-labs/photo-ai-recovery-subnet.git
cd photo-ai-recovery-subnet
```

## Step 2: Install Dependencies

Run the automated installation script:

```bash
sudo bash scripts/install_dependencies.sh
```

This script installs:
- System dependencies (Python, Node.js, PM2, build tools)
- Bittensor CLI (btcli) globally
- Python virtual environment with all required packages

**Note:** Installation may take 10-15 minutes depending on internet speed.

## Step 3: Setup Bittensor Wallet

Bittensor CLI is already installed globally. You may either create a new wallet or regenerate an existing one.

### Create a new wallet
```bash
btcli wallet new --wallet.name your_wallet_name
```

### Regenerate an existing wallet
```bash
btcli wallet regen_coldkey --wallet.name your_wallet_name
```

### Register your wallet in the subnet
```bash
btcli subnet register --wallet.name your_wallet_name --wallet.hotkey default --subtensor.network finney --subnet.netuid 00
```

### Verify registration
```bash
btcli subnet list --subtensor.network finney --subnet.netuid 00
```

## Step 4: Configure Environment

Copy and edit the configuration file:

```bash
cp validator.env.example validator.env
nano validator.env  # or use any text editor
```

**Required settings:**
```bash
# Wallet configuration (use your wallet names)
WALLET_NAME=validator
WALLET_HOTKEY=default

# Network settings
NETUID=                                     # Current subnet ID
SUBTENSOR_NETWORK=finney                     # 'finney' for mainnet, 'test' for testnet
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443

# Optional: W&B logging
WANDB_API_KEY=                              # Leave empty to disable
```

**For testnet, use:**
```bash
SUBTENSOR_NETWORK=test
SUBTENSOR_CHAIN_ENDPOINT=wss://test.finney.opentensor.ai:443
```

## Step 5: Start Validator

```bash
bash scripts/start_validator.sh
```

### Success Indicators

You should see:
```
[INFO] ✓ btcli installed successfully
[INFO] ✓ Virtual environment exists  
[INFO] ✓ Bittensor library import
[INFO] NETUID = 
[INFO] WALLET_NAME = validator
[INFO] Starting validator process...
[INFO] Starting auto-updater process...
[INFO] === Validator started successfully ===
```

## Monitoring

### Check Status
```bash
pm2 list
```

### View Logs
```bash
# Validator logs
pm2 logs sn-finney-validator-default

# Auto-updater logs  
pm2 logs auto_update_monitor

# Interactive monitoring
pm2 monit
```

### Stop Validator
```bash
pm2 stop sn-finney-validator-default
pm2 stop auto_update_monitor
```

## Troubleshooting

### Test Installation
```bash
bash scripts/test_installation.sh
```

### Common Issues

**1. "btcli command not found"**
- **Solution 1:** Open a new terminal session (PATH updated in new sessions)
- **Solution 2:** Reload PATH in current session:
```bash
source ~/.bashrc
export PATH="/root/.local/bin:$PATH"
```

**2. "Virtual environment not found"**
```bash
bash scripts/install_dependencies.sh
```

**3. "Missing environment variable"**
- Check `validator.env` file exists and contains required values

**4. Registration failed**
- Ensure sufficient TAO balance for registration costs
- Check network connectivity to Bittensor chain

### Get Help

- **Logs:** Always check PM2 logs first: `pm2 logs`
- **Discord:** [Bittensor Discord](https://discord.com/channels/xxxxx)

## Network Information

### Current Networks
- **Mainnet:** NETUID=, network=finney
- **Testnet:** NETUID=, network=test (check current testnet subnet ID)

