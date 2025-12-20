#!/bin/bash
# Start BrowserGym Green Agent
# This script activates the Python 3.11/3.12 environment and starts the green agent

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate BrowserGym environment
source .gym/bin/activate

# Load environment variables from .env
set -a
source .env
set +a

# Start green agent
python agents/green_agent.py --a2a-server --port 8000 --card-url $GREEN_AGENT_URL

