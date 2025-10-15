# 🚀 Getting Familiar with BrowserGym: Testing a Web Agent

This guide will walk you through testing a web agent in BrowserGym step by step, so you understand how the evaluation process works before building your green agent.

## 📋 Prerequisites

1. **Python 3.10+** installed
2. **Conda** or **pip** for package management
3. **OpenAI API key** (for the demo agent)
4. **Git** (already have BrowserGym cloned)

## 🛠️ Step 1: Environment Setup

### Option A: Using Conda (Recommended)
```bash
# Navigate to BrowserGym directory
cd /Users/layahaasini/Desktop/projects/BrowserGym

# Create and activate demo environment
conda env create -f demo_agent/environment.yml
conda activate demo-agent

# Install Playwright browser
python -m playwright install chromium
```

### Option B: Using pip
```bash
# Install BrowserGym
pip install browsergym

# Install demo dependencies
pip install openai

# Install Playwright browser
python -m playwright install chromium
```

## 🎯 Step 2: Set Up OpenAI API Key

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Or create a .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## 🧪 Step 3: Test Different Types of Tasks

### Test 1: Open-ended Web Navigation
```bash
# Navigate to demo_agent directory
cd demo_agent

# Run open-ended task (interactive)
python run_demo.py \
  --task_name openended \
  --start_url https://www.google.com \
  --visual_effects true \
  --use_screenshot true

# This will:
# - Open a browser window
# - Show you the agent interacting with Google
# - Let you see the evaluation process in real-time
```

### Test 2: MiniWoB Benchmark Task
```bash
# Run a MiniWoB task (automated)
python run_demo.py \
  --task_name miniwob.click-dialog \
  --visual_effects true \
  --use_axtree true

# This will:
# - Run a specific MiniWoB task
# - Show the agent trying to click a dialog
# - Demonstrate benchmark evaluation
```

### Test 3: WebArena Task
```bash
# Run a WebArena task
python run_demo.py \
  --task_name webarena.310 \
  --visual_effects true \
  --use_screenshot true

# This will:
# - Run a realistic web task
# - Show more complex interactions
# - Demonstrate real-world evaluation
```

## 📊 Step 4: Understanding the Evaluation Process

### What Happens During Evaluation:

1. **Environment Reset**: BrowserGym sets up the task environment
2. **Agent Observation**: The agent receives:
   - Current page state (HTML, accessibility tree)
   - Screenshot (if enabled)
   - Chat messages (if in chat mode)
   - Goal/task description

3. **Agent Action**: The agent decides what to do:
   - Click elements
   - Type text
   - Navigate pages
   - Send messages

4. **Environment Step**: BrowserGym executes the action and returns:
   - New observation
   - Reward/score
   - Success/failure status
   - Additional info

5. **Evaluation Metrics**: Track:
   - Success rate
   - Number of steps taken
   - Execution time
   - Specific task completion

### Key Files to Examine:

```bash
# Look at the demo agent implementation
cat demo_agent/agent.py

# Check the experiment runner
cat demo_agent/run_demo.py

# Examine results (after running)
ls -la results/
cat results/*/exp_record.json
```

## 🔍 Step 5: Analyze Results

After running a task, you'll see output like:

```
episode_length: 15
success: True
reward: 1.0
score: 1.0
task_completion: True
execution_time: 45.2
```

### Understanding the Metrics:

- **episode_length**: Number of actions taken
- **success**: Whether the task was completed successfully
- **reward**: Numerical reward from the environment
- **score**: Task-specific scoring
- **task_completion**: Boolean completion status
- **execution_time**: Time taken in seconds

## 🎬 Step 6: Create Your First Simple Test

Let's create a simple test to understand the evaluation process:

```python
# Create: simple_agent_test.py
import gymnasium as gym
import browsergym.core  # register the openended task

# Start an openended environment
env = gym.make(
    "browsergym/openended",
    task_kwargs={"start_url": "https://www.google.com"},
    headless=False,  # Show browser
    wait_for_user_message=True,  # Wait for user input
)

# Run the environment
obs, info = env.reset()
print("Initial observation keys:", obs.keys())
print("Goal:", info.get('goal', 'No specific goal'))

# Simple agent: just take a screenshot and exit
action = "take_screenshot()"
obs, reward, terminated, truncated, info = env.step(action)

print(f"Reward: {reward}")
print(f"Terminated: {terminated}")
print(f"Info: {info}")

env.close()
```

Run this test:
```bash
python simple_agent_test.py
```

## 📈 Step 7: Understanding White Agent Evaluation

Now you understand how BrowserGym evaluates agents. For your green agent, you'll need to:

### 1. **Set up multiple white agents**
```python
# Example white agent configurations
white_agents = [
    {
        "name": "GPT-4 Agent",
        "model": "gpt-4",
        "temperature": 0.1
    },
    {
        "name": "GPT-3.5 Agent", 
        "model": "gpt-3.5-turbo",
        "temperature": 0.3
    }
]
```

### 2. **Run evaluations in parallel**
```python
# Evaluate each white agent on multiple benchmarks
benchmarks = [
    "miniwob.click-dialog",
    "miniwob.choose-list", 
    "webarena.310",
    "webarena.4"
]

# Your green agent will:
# - Create isolated environments for each white agent
# - Run the benchmarks
# - Collect and aggregate results
# - Generate evaluation reports
```

### 3. **Collect and compare results**
```python
# Example evaluation results structure
evaluation_results = {
    "gpt-4-agent": {
        "miniwob.click-dialog": {"success": True, "score": 1.0, "steps": 5},
        "webarena.310": {"success": False, "score": 0.3, "steps": 15}
    },
    "gpt-3.5-agent": {
        "miniwob.click-dialog": {"success": True, "score": 0.8, "steps": 8},
        "webarena.310": {"success": True, "score": 0.9, "steps": 12}
    }
}
```

## 🎯 Next Steps for Your Green Agent

Now that you understand how BrowserGym works:

1. **Study the demo agent code** - Understand how agents interact with BrowserGym
2. **Experiment with different benchmarks** - See how different tasks are evaluated
3. **Analyze the evaluation metrics** - Understand what makes a good evaluation
4. **Start building your green agent** - Use this knowledge to create your evaluation orchestrator

## 🔧 Troubleshooting

### Common Issues:

1. **Playwright installation fails**:
   ```bash
   python -m playwright install-deps
   python -m playwright install chromium
   ```

2. **OpenAI API errors**:
   - Check your API key is set correctly
   - Ensure you have credits in your OpenAI account

3. **Browser won't open**:
   - Try `headless=False` to see the browser
   - Check if you have a display (for headless mode)

4. **Task not found**:
   - Make sure you've installed the specific benchmark package
   - Check available tasks with: `gym.envs.registry.keys()`

## 📚 Additional Resources

- **BrowserGym Documentation**: Check the README.md for more examples
- **Demo Agent Code**: Study `demo_agent/agent.py` for agent implementation
- **Experiment Framework**: Look at `browsergym/experiments/` for evaluation tools
- **Benchmark Details**: Check individual benchmark READMEs in `browsergym/*/README.md`

This hands-on experience will give you the foundation to build your green agent evaluation system!
