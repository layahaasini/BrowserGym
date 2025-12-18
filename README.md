# BrowserGym: Green and White Agent System

A system for evaluating web agents. The **green agent** evaluates **white agents** on BrowserGym benchmarks. The green agent loads white agents, runs them through standardized web tasks, measures performance, and generates reports.

---

## Architecture

### Green Agent (Evaluator)
**Purpose**: Evaluate white agents on benchmark tasks

**Location**: `green_evaluator.py`

**Functions**:
- Loads white agents from Python files
- Initializes benchmark environments (MiniWoB, WebArena, etc.)
- Executes white agents on tasks
- Collects metrics (success rate, steps taken, rewards)
- Generates evaluation reports
- Saves results to `green_evaluation_results/`

**Key Methods**:
- `load_agent(agent_path)` - Load white agent from file
- `evaluate_agent_on_task(agent, task_name, max_steps)` - Run single task
- `evaluate_agent_on_benchmark_suite(agent, task_list)` - Run multiple tasks

**Configuration**:
- `--agent_path`: Path to white agent file
- `--task`: Specific task to run
- `--max_steps`: Maximum steps per task (default: 50)
- `--results_dir`: Where to save results

### White Agent (Agent Under Test)
**Purpose**: Perform web automation tasks

**Locations**: 
- `demo_agent/agent.py` - Demo implementation
- `white_agent/white_agent.py` - Reference implementation

**Required Interface**:
```python
from browsergym.experiments.agent import Agent
from browsergym.core.action.highlevel import HighLevelActionSet

class YourWhiteAgent(Agent):
    def __init__(self):
        super().__init__()
        # Define what actions your agent can take
        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "nav", "tab"],
            strict=False,
            multiaction=False
        )
    
    def obs_preprocessor(self, obs):
        # Process observation from environment
        # Extract relevant information from obs dict
        return processed_obs
    
    def get_action(self, obs):
        # Decide what action to take based on observation
        # Return (action_string, info_dict)
        return 'click("5")', {}
```

**White Agent Components**:

1. **Action Set** - What the agent can do:
   - `bid`: Click elements by browsergym ID
   - `nav`: Navigate URLs (goto, go_back, go_forward)
   - `tab`: Manage tabs (new_tab, tab_close, tab_focus)
   - `chat`: Send messages (send_msg_to_user)
   - `infeas`: Report infeasible tasks

2. **Observation Preprocessor** - What the agent sees:
   - `axtree_object`: Accessibility tree (page structure)
   - `dom_object`: HTML DOM
   - `screenshot`: Page screenshot (optional)
   - `goal_object`: Task description
   - `last_action`: Previous action taken
   - `last_action_error`: Error from previous action
   - `open_pages_urls`: All open tabs

3. **Action Function** - How the agent decides:
   - Receives processed observation
   - Returns action string (e.g., `'click("42")'`, `'goto("https://example.com")'`)
   - Optionally returns metadata dict

**White Agent Configuration Options**:
- Model selection (GPT-4, GPT-3.5, etc.)
- Input modalities (HTML, accessibility tree, screenshots)
- Reasoning mode (chain-of-thought, direct)
- Temperature for LLM calls
- Maximum context length

**Example White Agents**:

*Simple white agent*:
```python
class SimpleAgent(Agent):
    def __init__(self):
        super().__init__()
        self.action_set = HighLevelActionSet(subsets=["bid", "chat"], strict=False)
        self.step = 0
    
    def obs_preprocessor(self, obs):
        return obs
    
    def get_action(self, obs):
        self.step += 1
        if self.step == 1:
            return 'click("5")', {}
        return 'send_msg_to_user("Done")', {}
```

*LLM-based white agent* (see `white_agent/white_agent.py`):
- Uses OpenAI API for decision-making
- Processes accessibility tree
- Implements chain-of-thought reasoning
- Maintains action history

---

## Benchmarks

The green agent evaluates white agents on:

- [**MiniWoB**](https://miniwob.farama.org/) - Simple web tasks (click buttons, fill forms)
- [**WebArena**](https://webarena.dev/) - Realistic websites (shopping, GitLab, Wikipedia)
- [**VisualWebArena**](https://jykoh.com/vwa) - Visual understanding required
- [**WorkArena**](https://github.com/ServiceNow/WorkArena) - Enterprise workflows (ServiceNow)
- [**AssistantBench**](https://github.com/oriyor/assistantbench) - Assistant tasks
- [**WebLINX**](https://github.com/McGill-NLP/weblinx) - Real-world interaction data

---

## Setup

### 1. Compute Instance
- **AWS EC2** (recommended for WebArena): Easier Docker setup
- **GCP Compute Engine**: Works for all benchmarks
- **Local machine**: MiniWoB and AssistantBench only

SSH access:
```bash
ssh -i <private_key> <user>@<external_ip>
```

### 2. Environment Variables

Create `.env` file in project root:
```bash
# Required for white agents using OpenAI
OPENAI_API_KEY="sk-proj-..."

# Required for all benchmarks
HOSTNAME=<EXTERNAL_IP>  # or localhost for local setup
```

### 3. Clone and Install

```bash
git clone <repository-url>
cd BrowserGym

# Install make if needed
sudo apt update && sudo apt install make -y  # Ubuntu
brew install make  # macOS

# Install Python dependencies
make install
```

This creates a virtual environment at `.gym/` and installs BrowserGym packages.

### 4. Install Benchmarks

#### MiniWoB++ (Required)
```bash
make install-bm-miniwob
```

Adds to `.env`:
```bash
MINIWOB_URL="file:///<PWD>/miniwob-plusplus/miniwob/html/miniwob/"
```

**What it does**: Clones MiniWoB++ repository with HTML task files

#### WebArena (Optional)

**Step 1**: Configure `.env`:
```bash
HOSTNAME=<EXTERNAL_IP>  # Your server's external IP
DOCKER_USERNAME=<DOCKERHUB_USERNAME>
DOCKER_PASSWORD=<DOCKERHUB_PASSWORD>
```

**Step 2**: Install Docker images

**Option A: AWS EC2 (Recommended)**

Use the pre-built AMI image that includes all WebArena Docker containers:

1. Go to EC2 console >> AMIs
2. Find WebArena AMI (see [WebArena Docker setup](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md#individual-website))
3. Launch instance from AMI
4. Run:
```bash
make install-bm-webarena
```

This configures the existing containers with your hostname.

**Option B: Non-AWS (GCP, Local, etc.)**

Download and load Docker images manually (images are 50-100GB total):

```bash
# Download Docker image tar files (large downloads, can resume if interrupted)
make install-bm-webarena-image-tars

# This downloads:
# - shopping_final_0712.tar (~5GB)
# - shopping_admin_final_0719.tar (~5GB)
# - postmill-populated-exposed-withimg.tar (~3GB)
# - gitlab-populated-final-port8023.tar (~15GB)
# - wikipedia_en_all_maxi_2022-05.zim (~95GB)

# Configure and start containers
make install-bm-webarena
```

**What gets installed**:
- Shopping website (port 7770)
- Shopping admin (port 7780)
- Reddit clone (port 9999)
- GitLab (port 8023)
- Wikipedia (port 8888)

Adds to `.env`:
```bash
WA_SHOPPING="http://${HOSTNAME}:7770"
WA_SHOPPING_ADMIN="http://${HOSTNAME}:7780/admin"
WA_REDDIT="http://${HOSTNAME}:9999"
WA_GITLAB="http://${HOSTNAME}:8023"
WA_MAP="http://${HOSTNAME}:4444"
WA_WIKIPEDIA="http://${HOSTNAME}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
WA_HOMEPAGE="http://${HOSTNAME}:4399"
```

**Note**: The Makefile uses these `WA_*` variables directly when configuring Docker containers.

**Troubleshooting**:
- If downloads fail, re-run `make install-bm-webarena-image-tars` (resumes from where it stopped)
- Ensure 200GB+ free disk space
- Check containers running: `docker ps`
- Test access: `curl http://localhost:7770`

#### VisualWebArena (Optional)
**Important**: Source installation required.
```bash
pip install -e browsergym/visualwebarena
make install-bm-visualwebarena
```

Adds to `.env`:
```bash
VWA_CLASSIFIEDS="http://${HOSTNAME}:9980"
VWA_CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
```

Uses Docker Compose to run classifieds website on port 9980.

#### WorkArena (Optional)
**Requirement**: Access is now managed via Hugging Face.
1. Gain access to [ServiceNow/WorkArena-Instances](https://huggingface.co/datasets/ServiceNow/WorkArena-Instances).
2. Create a Hugging Face Access Token (Read permission).

Add to `.env`:
```bash
HUGGING_FACE_HUB_TOKEN="hf_..."
# Remove SNOW_INSTANCE_* variables if present
```

#### AssistantBench (Optional)
No setup required - uses live websites.

#### WebLINX (Optional)

**Important**: WebLINX requires source code modification.

1. Clone WebLINX:
```bash
git clone https://github.com/McGill-NLP/weblinx
cd weblinx/modeling
```

2. Install WebLINX dependencies:
```bash
pip install -e .
```

3. **Modify BrowserGym source**:

Edit `browsergym/core/env.py` (in your `.gym` virtual environment):
```python
# Add WebLINX import
from browsergym.weblinx import weblinx_env  # Add this line

# Register WebLINX tasks
gym.register(
    id='weblinx',
    entry_point='browsergym.weblinx:WebLINXEnv',
)
```

4. Add to `.env`:
```bash
WEBLINX_PROJECT_DIR=${PWD}/weblinx/modeling
WEBLINX_DATA_DIR=${PWD}/weblinx/modeling/wl_data
```

5. Download WebLINX data:
```bash
cd weblinx/modeling
python scripts/download_data.py
```

**Note**: WebLINX integration is experimental and may require additional BrowserGym updates.

### 5. Verify Environment Setup

Your `.env` should now contain variables with dynamic references:
```bash
HOSTNAME=<your-ip>
MINIWOB_URL="file:///${PWD}/miniwob-plusplus/miniwob/html/miniwob/"
WA_SHOPPING="http://${HOSTNAME}:7770"
```

The Makefile automatically sources `.env` before running demos and tests, so you don't need to manually run `source .env`.

To manually verify environment variables:
```bash
source .env
echo $MINIWOB_URL  # Should show expanded path
echo $WA_SHOPPING  # Should show expanded URL
```

---

## Using the System

### Evaluate White Agent with Green Agent

The Makefile automatically sources `.env` before running, so environment variables are available.

**Single task evaluation**:
```bash
python green_evaluator.py --agent_path demo_agent/agent.py --task miniwob.click-dialog --max_steps 20
```

**Benchmark suite evaluation**:
```bash
python green_evaluator.py --agent_path white_agent/white_agent.py
```

**Using make commands** (automatically sources .env):
```bash
# WorkArena
make green-workarena TASK=workarena.servicenow.order-standard-laptop GREEN_AGENT_PATH=demo_agent/agent.py

# WebArena
make green-webarena TASK=webarena.4 GREEN_AGENT_PATH=white_agent/white_agent.py
```

**Evaluation flow**:
1. Green agent loads white agent Python file
2. Creates benchmark environment for specified task
3. Calls white agent's `get_action()` repeatedly
4. Tracks success, steps, rewards
5. Saves results to `green_evaluation_results/`

**Output example**:
```
Task: miniwob.click-dialog
Success: True
Steps: 3
Reward: 1.0
Exp dir: green_evaluation_results/eval_miniwob.click-dialog_1234567890/
```

**Results include**:
- `benchmark_evaluation_<timestamp>.json` - Full results
- `green_evaluator.log` - Execution logs
- `eval_<task>_<timestamp>/` - Per-task directories with traces

### Run White Agent Directly (No Evaluation)

Test white agents without metrics collection. The Makefile automatically sources `.env`:

```bash
# Single task (simplest way to test)
make demo TASKS="miniwob.click-test"

# WebArena (Verified working!)
make demo TASKS="webarena.0"

# WorkArena (Requires setup)
# Note: Fails if browsergym-workarena is not installed or HUGGING_FACE_HUB_TOKEN is missing.
make demo TASKS="workarena.servicenow.order-standard-laptop"

# VisualWebArena (Requires setup)
# Note: Fails if scikit-image build fails (Python 3.14 issue) or package not installed.
make demo TASKS="visualwebarena.classifieds.0"

# All task types
make demo TASKS="miniwob.click-test webarena.0"
```

This runs the demo white agent (`demo_agent/agent.py`) directly on tasks without green agent evaluation.

### Create Custom White Agent

1. Create Python file:
```python
# my_agent.py
from browsergym.experiments.agent import Agent
from browsergym.core.action.highlevel import HighLevelActionSet
import openai

class MyWhiteAgent(Agent):
    def __init__(self):
        super().__init__()
        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "nav", "tab"],
            strict=False,
            multiaction=False
        )
        self.client = openai.OpenAI()
    
    def obs_preprocessor(self, obs):
        # Extract what your agent needs from observation
        return {
            "goal": obs["goal_object"],
            "axtree": obs["axtree_object"],
            "url": obs["open_pages_urls"][obs["active_page_index"]]
        }
    
    def get_action(self, obs):
        # Your agent's logic here
        # Use LLM, rules, or any approach
        
        prompt = f"Task: {obs['goal']}\nPage: {obs['url']}\nWhat action?"
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        action = response.choices[0].message.content
        return action, {}
```

2. Evaluate with green agent:
```bash
python green_evaluator.py --agent_path my_agent.py --task miniwob.click-test
```

---

## Test the System

Verify green agent and benchmarks work correctly. All test commands automatically source `.env`:

```bash
# Test green agent functionality (29 tests)
make test-green

# Test core BrowserGym (benchmark environments)
make test-core

# Test installed benchmarks only (skips WebArena/VisualWebArena if not installed)
make test-benchmarks

# Test everything (includes WebArena/VisualWebArena - will fail if not installed)
make test-all
```

**Recommended**: Use `make test-benchmarks` to test only installed benchmarks.

**Test categories**:
- **Initialization**: Green agent starts correctly
- **Agent loading**: Can load white agents from files
- **Task evaluation**: MiniWoB, WebArena, WorkArena tasks work
- **Backend preparation**: Benchmark backends initialize properly
- **Error handling**: Invalid tasks, broken agents handled gracefully
- **Metrics**: Success rate, steps, rewards calculated correctly
- **Reproducibility**: Same agent + task = same results
- **Integration**: Full evaluation pipelines work end-to-end

**Run specific tests**:
```bash
# Just initialization tests (fast)
pytest tests/green_evaluator/test_green_evaluator.py::TestInitialization -v

# Skip slow tests (that run actual agents)
pytest tests/green_evaluator/ -m "not slow" -v

# Only slow tests (actual agent runs)
pytest tests/green_evaluator/ -m "slow" -v
```

---

## A2A Server Mode

Run green agent as server for remote white agent evaluation:

### 1. Install
```bash
brew install wget  # macOS
sudo apt install wget -y  # Ubuntu

make install-agentbeats
```

### 2. Start Server
```bash
python green_evaluator.py --a2a-server --host 0.0.0.0 --port 8000 --card-url http://<EXTERNAL_IP>:8000
```

### 3. Register
```bash
make register-agent
make register-battle
```

The green agent exposes endpoints for remote white agent evaluation via A2A protocol.

---

## File Structure

```
BrowserGym/
├── green_evaluator.py              # Green agent (evaluator)
├── green_evaluator_card.json       # Green agent A2A card
│
├── demo_agent/
│   └── agent.py                    # Demo white agent
│
├── white_agent/
│   └── white_agent.py              # Reference white agent (LLM-based)
│
├── green_evaluation_results/       # Evaluation outputs
│   ├── benchmark_evaluation_*.json # Results
│   ├── green_evaluator.log         # Logs
│   └── eval_*/                     # Per-task traces
│
├── tests/
│   ├── green_evaluator/            # Green agent tests
│   └── core/                       # BrowserGym core tests
│
├── browsergym/
│   ├── core/                       # BrowserGym core
│   ├── miniwob/                    # MiniWoB benchmark
│   ├── webarena/                   # WebArena benchmark
│   ├── visualwebarena/             # VisualWebArena benchmark
│   └── experiments/                # Experiment framework
│
├── .env                            # Configuration
├── Makefile                        # Commands
└── pytest.ini                      # Test configuration
```

---

## Configuration Reference

### .env File

Variables use `${HOSTNAME}` and `${PWD}` for dynamic expansion when sourced:

```bash
# General
HOSTNAME=<EXTERNAL_IP or localhost>
OPENAI_API_KEY="sk-proj-..."
DOCKER_USERNAME=<username>
DOCKER_PASSWORD=<password>

# MiniWoB (uses ${PWD} for dynamic path)
MINIWOB_URL="file:///${PWD}/miniwob-plusplus/miniwob/html/miniwob/"

# WebArena (uses ${HOSTNAME} for dynamic URLs)
WA_SHOPPING="http://${HOSTNAME}:7770"
WA_SHOPPING_ADMIN="http://${HOSTNAME}:7780/admin"
WA_REDDIT="http://${HOSTNAME}:9999"
WA_GITLAB="http://${HOSTNAME}:8023"
WA_MAP="http://${HOSTNAME}:4444"
WA_WIKIPEDIA="http://${HOSTNAME}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
WA_HOMEPAGE="http://${HOSTNAME}:4399:"
WA_FULL_RESET=""

# VisualWebArena
DATASET=visualwebarena
VWA_CLASSIFIEDS="http://${HOSTNAME}:9980"
VWA_CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
VWA_SHOPPING="http://${HOSTNAME}:7770"
VWA_REDDIT="http://${HOSTNAME}:9999"
VWA_WIKIPEDIA="http://${HOSTNAME}:8888"
VWA_HOMEPAGE="http://${HOSTNAME}:80"
VWA_FULL_RESET=""

# WorkArena
SNOW_INSTANCE_URL="https://instance.service-now.com"
SNOW_INSTANCE_UNAME="admin"
SNOW_INSTANCE_PWD="password"

# WebLINX
WEBLINX_PROJECT_DIR=${PWD}/weblinx/modeling
WEBLINX_DATA_DIR=${PWD}/weblinx/modeling/wl_data

# AgentBeats
AGENT_HOST=${HOSTNAME}
AGENT_PORT=8000
LAUNCHER_HOST=${HOSTNAME}
LAUNCHER_PORT=8080
```

**Note**: The Makefile automatically sources `.env` before running demos, tests, and evaluations, so you don't need to manually run `source .env` each time.

### Green Agent Arguments
```bash
--agent_path <path>         # White agent file
--task <task_name>          # Specific task
--max_steps <int>           # Max steps per task (default: 50)
--results_dir <dir>         # Results directory
--a2a-server                # Run as A2A server
--host <ip>                 # Server host
--port <port>               # Server port
```

### White Agent Configuration
Depends on implementation. Example (`white_agent/white_agent.py`):
- `model_name`: GPT model (gpt-4o-mini, gpt-4, etc.)
- `use_axtree`: Use accessibility tree (default: True)
- `use_html`: Use HTML DOM (default: False)
- `use_screenshot`: Use screenshots (default: False)
- `chat_mode`: Interactive mode (default: False)

---

## Ecosystem

- [AgentLab](https://github.com/ServiceNow/AgentLab): Agent experiment framework
- [WorkArena(++)](https://github.com/ServiceNow/WorkArena): Enterprise task benchmarks
- [WebArena](https://github.com/web-arena-x/webarena): Realistic web tasks
- [VisualWebArena](https://github.com/web-arena-x/visualwebarena): Visual reasoning tasks
- [MiniWoB(++)](https://miniwob.farama.org/): Synthetic web microtasks
- [WebLINX](https://github.com/McGill-NLP/weblinx): Real-world interaction data
- [AssistantBench](https://github.com/oriyor/assistantbench): Assistant evaluation
- [DoomArena](https://github.com/ServiceNow/DoomArena): Security testing

---

## Citation

```bibtex
@article{
    chezelles2025browsergym,
    title={The BrowserGym Ecosystem for Web Agent Research},
    author={Thibault Le Sellier de Chezelles and Maxime Gasse and Alexandre Lacoste and Massimo Caccia and Alexandre Drouin and L{\'e}o Boisvert and Megh Thakkar and Tom Marty and Rim Assouel and Sahar Omidi Shayegan and Lawrence Keunho Jang and Xing Han L{\`u} and Ori Yoran and Dehan Kong and Frank F. Xu and Siva Reddy and Graham Neubig and Quentin Cappart and Russ Salakhutdinov and Nicolas Chapados},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2025},
    url={https://openreview.net/forum?id=5298fKGmv3},
    note={Expert Certification}
}
```
