# BrowserGym: Green and White Agent Evaluation System

A system for evaluating web automation agents on BrowserGym benchmarks. The green agent evaluates white agents on standardized web tasks, measures performance, and generates reports.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Running Agents](#running-agents)
5. [Testing](#testing)
6. [Benchmarks](#benchmarks)
7. [Configuration](#configuration)
8. [A2A Server Mode](#a2a-server-mode)
9. [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# Install
make install

# Run white agent on task
cd agents
python green_agent.py --task miniwob.click-test --max_steps 50

# Run tests
make test-green
```

## Architecture

### Green Agent (Evaluator)

**Location**: `agents/green_agent.py`

**Purpose**: Evaluate white agents on benchmark tasks

**Functions**:
- Loads white agents and initializes benchmark environments
- Executes white agents on tasks
- Collects metrics (success rate, steps taken, rewards)
- Generates evaluation reports
- Saves results to `agents/results/`

**Configuration**:
- `--task`: Specific task to run (e.g., `miniwob.click-test`)
- `--max_steps`: Maximum steps per task (default: 50)
- `--model_name`: LLM model to use (default: `gpt-4o-mini`)

**Key Features**:
- A2A protocol compliant
- Supports all BrowserGym benchmarks
- Produces detailed evaluation artifacts
- Can run as standalone or as A2A server

### White Agent (Competitor)

**Location**: `agents/white_agent.py`

**Purpose**: Perform web automation tasks

**Required Interface**:
```python
from browsergym.experiments import Agent
from browsergym.core.action.highlevel import HighLevelActionSet

class WhiteAgent(Agent):
    def __init__(self):
        super().__init__()
        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "nav", "tab", "infeas"],
            strict=False,
            multiaction=False
        )
    
    def obs_preprocessor(self, obs):
        # Process observation from environment
        return {
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            # ... other observations
        }
    
    def get_action(self, obs):
        # Decide action based on observation
        # Return (action_string, metadata_dict)
        return 'click("5")', {}
```

**Components**:

1. **Action Set** - Available actions:
   - `bid`: Click elements by browser ID (e.g., `click("5")`)
   - `nav`: Navigate URLs (`goto(url)`, `go_back()`, `go_forward()`)
   - `tab`: Manage tabs (`new_tab()`, `tab_close()`, `tab_focus(index)`)
   - `chat`: Send messages (`send_msg_to_user(message)`)
   - `infeas`: Report infeasible tasks (`report_infeasible()`)

2. **Observation Types**:
   - `axtree_object`: Accessibility tree (page structure)
   - `dom_object`: HTML DOM
   - `screenshot`: Page screenshot (optional)
   - `goal_object`: Task description
   - `last_action`: Previous action taken
   - `last_action_error`: Error from previous action
   - `open_pages_urls`: All open tabs
   - `open_pages_titles`: Tab titles
   - `active_page_index`: Current tab index

3. **Configuration**:
   - `model_name`: OpenAI model (default: `gpt-4o-mini`)
   - `chat_mode`: Enable chat interactions (default: `False`)
   - `use_html`: Use HTML DOM (default: `False`)
   - `use_axtree`: Use accessibility tree (default: `True`)
   - `use_screenshot`: Use screenshots (default: `False`)

## Installation

### Basic Installation

```bash
make install
```

This installs:
- BrowserGym core
- Python dependencies
- Playwright with Chromium browser

### Manual Installation

```bash
python3 -m venv .gym
source .gym/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### Benchmark-Specific Setup

```bash
# MiniWoB
make install-bm-miniwob

# WebArena (requires Docker)
make install-bm-webarena

# VisualWebArena (requires Docker)
make install-bm-visualwebarena

# WorkArena (requires ServiceNow instance)
make install-bm-workarena
```

## Running Agents

### Running White Agent on Tasks

#### Single Task
```bash
cd agents
python green_agent.py --task miniwob.click-test --max_steps 50
```

#### Multiple Tasks
```bash
cd agents
python green_agent.py --task miniwob.click-test --max_steps 50
python green_agent.py --task miniwob.click-dialog --max_steps 50
python green_agent.py --task miniwob.choose-date --max_steps 50
```

#### Custom Configuration
```bash
cd agents
python green_agent.py \
  --task webarena.shopping.0 \
  --max_steps 100 \
  --model_name gpt-4o
```

### Running Green Agent Evaluation

The green agent evaluates white agents by running them on tasks and measuring performance.

#### Evaluate on Single Task
```bash
cd agents
python green_agent.py --task miniwob.click-test --max_steps 50
```

Results saved to `agents/results/` directory with:
- Task name
- Success/failure
- Reward (0 or 1)
- Number of steps taken
- Action history
- Screenshots (if enabled)

#### Evaluate on Benchmark Suite
```bash
cd agents

# MiniWoB suite
for task in miniwob.click-test miniwob.click-dialog miniwob.choose-date; do
    python green_agent.py --task $task --max_steps 50
done

# WebArena suite
for task in webarena.shopping.0 webarena.reddit.1 webarena.gitlab.2; do
    python green_agent.py --task $task --max_steps 100
done
```

#### Using Makefile
```bash
# Run with specific tasks
make demo TASKS="miniwob.click-test miniwob.click-dialog"
```

### Available Tasks

**MiniWoB** (simple web interactions):
- `miniwob.click-test`
- `miniwob.click-dialog`
- `miniwob.choose-date`
- `miniwob.enter-text`
- `miniwob.login-user`
- And 100+ more tasks

**WebArena** (complex web tasks):
- `webarena.shopping.0` - E-commerce tasks
- `webarena.reddit.1` - Social media tasks
- `webarena.gitlab.2` - Code repository tasks
- `webarena.wikipedia.3` - Information retrieval

**VisualWebArena** (visually-grounded tasks):
- `visualwebarena.398`
- `visualwebarena.1`
- Requires screenshot processing

**AssistantBench** (real-world tasks):
- `assistantbench.find-gym`
- `assistantbench.compare-hotels`
- `assistantbench.book-restaurant`

## Testing

### Run All Green Agent Tests
```bash
make test-green
```

### Run Specific Test Categories

```bash
# Initialization tests (fast)
pytest tests/green_evaluator/test_green_evaluator.py::TestInitialization -v

# Task evaluation tests
pytest tests/green_evaluator/test_green_evaluator.py::TestTaskEvaluation -v

# Benchmark comparison tests
pytest tests/green_evaluator/test_green_evaluator.py::TestBenchmarkComparison -v

# A2A server tests
pytest tests/green_evaluator/test_green_evaluator.py::TestA2AServer -v
```

### Run Fast Tests Only
```bash
pytest tests/green_evaluator/ -v -m "not slow"
```

### Test Benchmark-Specific Functionality
```bash
# Test only MiniWoB
pytest tests/miniwob/ -v

# Test only AssistantBench
pytest tests/assistantbench/ -v

# Test only installed benchmarks
make test-benchmarks
```

## Reproducing Benchmark Results

### MiniWoB Benchmark
```bash
cd agents
python green_agent.py --task miniwob.click-test
python green_agent.py --task miniwob.click-dialog
python green_agent.py --task miniwob.choose-date
```

Expected results:
- Success rate: 80-95% on simple tasks
- Average steps: 2-5 for click tasks

### WebArena Benchmark

Requires Docker services running.

**Setup**:
```bash
make install-bm-webarena
```

**Run**:
```bash
cd agents
python green_agent.py --task webarena.shopping.0 --max_steps 100
python green_agent.py --task webarena.reddit.1 --max_steps 100
```

Expected results:
- Success rate: 40-60% on complex tasks
- Average steps: 10-30 for shopping tasks

### VisualWebArena Benchmark

Requires Docker services and screenshot processing.

**Setup**:
```bash
make install-bm-visualwebarena
```

**Run**:
```bash
cd agents
python green_agent.py --task visualwebarena.398 --max_steps 100
```

### AssistantBench
```bash
cd agents
python green_agent.py --task assistantbench.find-gym
python green_agent.py --task assistantbench.compare-hotels
```

Expected results:
- Success rate: 50-70%
- Average steps: 15-25

## Configuration

### Environment Variables

Copy `sample.env` to `.env`:
```bash
cp sample.env .env
```

**Required Variables**:

```bash
# Basic Configuration
HOSTNAME=your-hostname-or-ip
OPENAI_API_KEY=sk-your-openai-api-key

# GitHub (for Docker publishing)
GITHUB_USERNAME=yourusername
GITHUB_TOKEN=ghp_your_token

# MiniWoB
MINIWOB_URL=http://${HOSTNAME}:8080

# WebArena
WA_SHOPPING=http://${HOSTNAME}:7770
WA_SHOPPING_ADMIN=http://${HOSTNAME}:7780/admin
WA_REDDIT=http://${HOSTNAME}:9999
WA_GITLAB=http://${HOSTNAME}:8023
WA_MAP=http://${HOSTNAME}:4444
WA_WIKIPEDIA=http://${HOSTNAME}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing
WA_HOMEPAGE=http://${HOSTNAME}:4399

# VisualWebArena
VWA_CLASSIFIEDS=http://${HOSTNAME}:9980
VWA_CLASSIFIEDS_RESET_TOKEN=4b61655535e7ed388f0d40a93600254c
VWA_SHOPPING=http://${HOSTNAME}:7770
VWA_REDDIT=http://${HOSTNAME}:9999
VWA_WIKIPEDIA=http://${HOSTNAME}:8888
VWA_HOMEPAGE=http://${HOSTNAME}:80
DATASET=visualwebarena

# WorkArena (ServiceNow)
SNOW_INSTANCE_URL=https://your-instance.service-now.com
SNOW_INSTANCE_UNAME=admin
SNOW_INSTANCE_PWD=your-password
```

### Docker Setup

**Setup WebArena Services**:
```bash
make install-bm-webarena
```

**Setup VisualWebArena Services**:
```bash
make install-bm-visualwebarena
```

**Check Services Running**:
```bash
docker ps
```

## A2A Server Mode

Both agents can run as A2A (Agent-to-Agent) protocol servers for integration with AgentBeats platform.

### Start Green Agent Server

```bash
cd agents
python green_agent.py --a2a-server --port 8000 --card-url http://YOUR_HOST:8000
```

Endpoints:
- `/.well-known/agent-card.json` - Agent card
- `/sendMessage` - Send assessment request
- `/sendMessageStream` - Stream assessment updates
- `/getTask` - Get task status
- `/cancelTask` - Cancel running task
- `/health` - Health check

### Start White Agent Server

```bash
cd agents
python white_agent.py --a2a-server --port 8001 --card-url http://YOUR_HOST:8001
```

Endpoints:
- `/.well-known/agent-card.json` - Agent card
- `/sendMessage` - Send task request
- `/getTask` - Get task status
- `/health` - Health check

### With Cloudflare Tunnel

For secure HTTPS access without opening firewall ports:

**Terminal 1 - Green Tunnel**:
```bash
cloudflared tunnel --url http://localhost:8000
# Note the URL: https://abc-123-def.trycloudflare.com
```

**Terminal 2 - White Tunnel**:
```bash
cloudflared tunnel --url http://localhost:8001
# Note the URL: https://ghi-456-jkl.trycloudflare.com
```

**Terminal 3 - Green Agent**:
```bash
cd agents
python green_agent.py --a2a-server --port 8000 \
  --card-url https://abc-123-def.trycloudflare.com
```

**Terminal 4 - White Agent**:
```bash
cd agents
python white_agent.py --a2a-server --port 8001 \
  --card-url https://ghi-456-jkl.trycloudflare.com
```

### Register on AgentBeats

1. Go to https://v2.agentbeats.org
2. Register green agent with URL from tunnel
3. Register white agent with URL from tunnel
4. Create assessment between them

### Docker Images

Build and publish agent images:

```bash
make containerize-agents
```

This creates:
- `ghcr.io/yourusername/browsergym-green-evaluator:v1.0`
- `ghcr.io/yourusername/browsergym-white-agent:v1.0`

## Results Format

Results are saved in JSON format to `agents/results/` directory:

```json
{
  "task_name": "miniwob.click-test",
  "success": true,
  "reward": 1.0,
  "steps": 3,
  "max_steps": 50,
  "model": "gpt-4o-mini",
  "details": {
    "cum_reward": 1.0,
    "n_steps": 3,
    "action_history": ["click('5')", "click('submit')"],
    "screenshots": ["step_0.png", "step_1.png"]
  }
}
```

## Troubleshooting

### Installation Issues

**Missing dependencies**:
```bash
pip install -r requirements.txt
playwright install chromium
```

**Import errors**:
```bash
pip install -e .
```

### Runtime Issues

**Agent fails to start**:
```bash
# Check Python version (3.10+ required)
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Task execution timeout**:
```bash
# Increase max_steps
cd agents
python green_agent.py --task miniwob.click-test --max_steps 100
```

**OpenAI API errors**:
```bash
# Check API key is set
echo $OPENAI_API_KEY

# Or set directly
export OPENAI_API_KEY=sk-your-key
```

### Benchmark Issues

**Benchmark not found**:
```bash
# Install benchmark
make install-bm-miniwob
```

**Docker services not running**:
```bash
# Start WebArena services
cd classifieds_workspace/classifieds_docker_compose
docker-compose up -d

# Check services
docker ps
```

**Port conflicts**:
```bash
# Check what's using port
sudo lsof -i :8000

# Kill process
sudo kill -9 PID
```

### A2A Server Issues

**Agent card not loading**:
```bash
# Test locally
curl http://localhost:8000/.well-known/agent-card.json

# Test through tunnel
curl https://your-tunnel-url/.well-known/agent-card.json
```

**Firewall blocking**:
```bash
# Open ports
sudo ufw allow 8000/tcp
sudo ufw allow 8001/tcp
```

**Cloudflare tunnel not connecting**:
```bash
# Restart tunnel
pkill cloudflared
cloudflared tunnel --url http://localhost:8000
```

## Resources

- BrowserGym: https://github.com/ServiceNow/BrowserGym
- AgentBeats: https://agentbeats.dev
- A2A Protocol: https://a2aprotocol.org
- OpenAI API: https://platform.openai.com

## License

See LICENSE file.
