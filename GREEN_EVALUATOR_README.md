# Green Evaluator Agent | BrowserGym MiniWoB Wrapper

An evaluator agent that acts as a wrapper for testing other web agents (white/demo agents) on BrowserGym benchmarks, specifically MiniWoB.

## What is the Green Evaluator?

The Green Evaluator is a **benchmark testing framework** that:

1. **Takes white agents as input** (like the demo agent from `demo_agent/agent.py`)
2. **Runs them through MiniWoB benchmark tasks** (click dialog, choose list, etc.)
3. **Evaluates their performance** (success rate, steps taken, rewards)

## Architecture

```
Green Evaluator (this code)
    ↓ (loads and tests)
White/Demo Agent (agent.py)
    ↓ (executes actions in)
BrowserGym Environment (MiniWoB tasks)
    ↓ (returns results to)
Green Evaluator (evaluates performance)
```

## Quick Start

### 1. Set up Environment

```bash
# Navigate to BrowserGym directory
cd /Users/layahaasini/Desktop/projects/BrowserGym

# Set your OpenAI API key (required)
export OPENAI_API_KEY="your-api-key-here"

# Note: our Green Evaluator automatically loads MINIWOB_URL from .env file
# If you need to load it manually: source .env
```

### 2. Test Single Task

```bash
python3 green_evaluator.py --agent_path demo_agent/agent.py --task miniwob.click-dialog
```

### 3. Test Full Benchmark Suite

```bash
python3 green_evaluator.py --agent_path demo_agent/agent.py
```

### 4. Run Example

```bash
python3 green_evaluator.py
```

## What Gets Evaluated

The Green Evaluator tracks these metrics:

- **Success Rate**: How many tasks the agent completed successfully
- **Efficiency**: Average number of steps taken per task
- **Rewards**: Total and average rewards earned
- **Task-by-Task Results**: Detailed results for each individual task
- **Error Handling**: How the agent handles failures

## Available MiniWoB Tasks

The evaluator can test agents on these MiniWoB tasks:

- `miniwob.click-dialog` - Click on dialog buttons
- `miniwob.choose-list` - Select items from lists
- `miniwob.click-checkboxes` - Check/uncheck boxes
- `miniwob.choose-date` - Select dates from calendars
- `miniwob.ascending-numbers` - Arrange numbers in order
- `miniwob.bisect-angle` - Bisect angles
- `miniwob.book-flight` - Book flights
- `miniwob.buy-ticket` - Purchase tickets

## 🧪 Example Usage

```python
from green_evaluator import GreenEvaluator

# Initialize evaluator
evaluator = GreenEvaluator(results_dir="./my_evaluation")

# Load an agent
agent = evaluator.load_agent("demo_agent/agent.py")

# Test single task
result = evaluator.evaluate_agent_on_task(agent, "miniwob.click-dialog")
print(f"Success: {result['success']}, Steps: {result['steps_taken']}")

# Test multiple tasks
results = evaluator.evaluate_agent_on_benchmark_suite(
    agent, 
    ["miniwob.click-dialog", "miniwob.choose-list"]
)
print(f"Success rate: {results['success_rate']:.2%}")
```

## Requirements

- BrowserGym installed and configured
- MiniWoB setup completed (`make setup-miniwob`)
- OpenAI API key for agent evaluation
- Python 3.8+
