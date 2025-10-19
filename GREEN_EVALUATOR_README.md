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

  ## Evaluating the Evaluator

- Our next steps for this project heavily emphasize formulating more rigorous and thorough test cases. For now, since we are still in early stages of building our green agent (and it doesn't have many functionalities to test yet), we have been relying on **compilation** checks, checks for **reproductability**, as well as validating that **metrics** calculated from manually running the MiniWoB task are the same as those calculated by the green agent.

## Requirements

- BrowserGym installed and configured
- MiniWoB setup completed (`make setup-miniwob`)
- OpenAI API key for agent evaluation
- Python 3.8+
