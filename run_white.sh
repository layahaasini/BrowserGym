#!/bin/bash
# Start BrowserGym White Agent
# This script activates the Python 3.11/3.12 environment and starts the white agent

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate BrowserGym environment
source .gym/bin/activate

# Load environment variables from .env
set -a
source .env
set +a

# Start white agent
python agents/white_agent.py --a2a-server --port 8001 --card-url $WHITE_AGENT_URL

