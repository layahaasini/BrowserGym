# BrowserGym Agents: Green Evaluator & White Solver

This repository contains the implementation of our submission for the AgentBeats competition: a **White Agent** (Solver) and a **Green Agent** (Evaluator) built on the BrowserGym ecosystem.

## Project Overview

Our goal is to provide a robust, unified interface for evaluating web agents across all 6 BrowserGym benchmarks (MiniWoB, WorkArena, WebArena, VisualWebArena, AssistantBench, and WebLINX). 

- **Green Agent (Evaluator)**: A universal wrapper that standardizes evaluation. instead of re-implementing BrowserGym's framework, it wraps the native environment to ensure 100% faithfulness to original benchmarks while exposing an AgentBeats-compatible A2A server interface.
- **White Agent (Solver)**: A generalist agent using `gpt-4o-mini` with Chain-of-Thought (CoT) reasoning to solve tasks. It follows a "Receive-Reason-Act" pipeline, analyzing the Accessibility Tree and screenshots to plan actions step-by-step.

## Setup

### 1. Prerequisites
- **OS**: Ubuntu (Recommended), macOS, or Linux.
- **Python**: 3.10+
- **Make**: Ensure `make` is installed (e.g., `sudo apt install make`).

### 2. Installation
Clone the repo and install dependencies:
```bash
git clone <repository-url>
cd BrowserGym
make install
pip install fastapi uvicorn python-dotenv
```

### 3. Environment Config
Create a `.env` file with your credentials:
```bash
# Required
OPENAI_API_KEY="sk-..."

# WorkArena Setup (New)
# Authenticate with Hugging Face for WorkArena instances
# 1. Login via CLI: huggingface-cli login
# 2. OR set token here:
HUGGING_FACE_HUB_TOKEN="hf_..."

# Benchmark Paths (as needed)
MINIWOB_URL="file:///${PWD}/browsergym/miniwob/..."
```

---

## Part 1: White Agent (Solver)
Located in `white_agent/white_agent.py`.

### 1. Run as Standalone Server (AgentBeats Mode)
To run the agent as a persistent server that accepts tasks:
```bash
python white_agent/white_agent.py --a2a-server --port 8000
```
- **Endpoints**: 
    - `GET /status`: Health check.
    - `GET /card`: Returns agent capabilities.

### 2. Run Single Task (Development)
You can test the agent on a specific task using the Green Evaluator locally:
```bash
python green_evaluator.py --agent_path white_agent/white_agent.py --task miniwob.click-test
```

---

## Part 2: Green Agent (Evaluator)
Located in `green_evaluator.py`.

### 1. Run as Evaluation Server (AgentBeats Mode)
To start the Green Agent as an A2A-compliant evaluator service:
```bash
python green_evaluator.py --a2a-server --port 8000
```
This allows it to receive `assessment_request` messages and orchestrate evaluations.

### 2. Run Local Evaluation (CLI)
To manually evaluate a White Agent on a specific task:
```bash
python green_evaluator.py --agent_path white_agent/white_agent.py --task <task_name>
```

**Examples:**
```bash
# MiniWoB
python green_evaluator.py --agent_path white_agent/white_agent.py --task miniwob.click-test

# WorkArena
python green_evaluator.py --agent_path white_agent/white_agent.py --task workarena.servicenow.order-standard-laptop
```

### 3. Run Tests
Verify the installation and core functionality:
```bash
make test-core
```

---

## Submission Details
- **Faithfulness**: The Green Agent uses `browsergym.make_env()` directly, ensuring score parity with native benchmarks.
- **Coverage**: Supports all 6 BrowserGym benchmarks.
- **Architecture**: Both agents implement an A2A Server layer for seamless integration with the AgentBeats platform.

## Citation
If you use this work, please cite the original BrowserGym paper:
```bibtex
@article{chezelles2025browsergym,
    title={The BrowserGym Ecosystem for Web Agent Research},
    author={Thibault Le Sellier de Chezelles et al.},
    journal={Transactions on Machine Learning Research},
    year={2025}
}
```
