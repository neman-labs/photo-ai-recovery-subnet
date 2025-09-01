# Miner Guide

## Overview

**What you need:**
- Ubuntu server (20.04+ recommended)
- Basic command line knowledge

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
cp miner.env.example miner.env
nano miner.env  # or use any text editor
```

**Required settings:**
```bash
# Wallet configuration (use your wallet names)
WALLET_NAME=miner
WALLET_HOTKEY=default

# Network settings
NETUID=                                    # Current subnet ID
SUBTENSOR_NETWORK=finney                     # 'finney' for mainnet, 'test' for testnet
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443

# Miner settings
AXON_PORT=8091                              # Port for miner communication
BLACKLIST_FORCE_VALIDATOR_PERMIT=True       # Only allow permitted validators
BLACKLIST_VALIDATOR_MIN_STAKE=1000          # Minimum validator stake required
```

**For testnet, use:**
```bash
SUBTENSOR_NETWORK=test
SUBTENSOR_CHAIN_ENDPOINT=wss://test.finney.opentensor.ai:443
```

## Step 5: Start Miner

```bash
bash scripts/start_miner.sh
```

### Success Indicators

You should see:
```
[INFO] ✓ Virtual environment exists  
[INFO] ✓ Loading configuration from miner.env
[INFO] NETUID = 
[INFO] WALLET_NAME = miner
[INFO] Starting miner process...
[INFO] === Miner started successfully ===
```

## Monitoring

### Check Status
```bash
pm2 list
```

### View Logs
```bash
# Miner logs
pm2 logs finney-miner-default

# Interactive monitoring
pm2 monit
```

### Stop Miner
```bash
pm2 stop finney-miner-default
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
- Check `miner.env` file exists and contains required values


### Get Help

- **Logs:** Always check PM2 logs first: `pm2 logs`
- **Discord:** [Bittensor Discord](https://discord.com/channels/xxxxx)

## Network Information

### Current Networks
- **Mainnet:** NETUID=, network=finney
- **Testnet:** NETUID=, network=test (check current testnet subnet ID)

### Requirements
- **Mainnet:** Registered wallet with TAO for transactions
- **Testnet:** Free testnet TAO from faucet

## Miner Development

### Baseline Implementation
The repository includes a baseline miner implementation that:
- Receives image restoration tasks from validators
- Uses basic algorithms for upscaling, denoising, and inpainting
- Returns processed images for evaluation

### Improving Your Miner
To earn higher rewards:
1. **Implement better algorithms** - Use state-of-the-art ML models
2. **Optimize performance** - Faster processing = more tasks completed
3. **Handle edge cases** - Robust error handling for various image types
4. **Resource management** - Efficient memory and GPU usage

### Key Files
- `neurons/miner.py` - Main miner entry point
- `imagerecovery/base/miner.py` - Base miner class
- `imagerecovery/services/` - Image processing services

---

**Success!** Your miner should now be running and receiving tasks from validators. Better algorithms and implementations will earn higher rewards over time.
