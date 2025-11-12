# Green Agent and A2A Functionality - Complete Beginner's Guide

## Table of Contents
1. [What is BrowserGym?](#what-is-browsergym)
2. [What is the Green Evaluator (Green Agent)?](#what-is-the-green-evaluator-green-agent)
3. [What is A2A (Agent-to-Agent)?](#what-is-a2a-agent-to-agent)
4. [Why Do We Need A2A?](#why-do-we-need-a2a)
5. [How Does It All Work Together?](#how-does-it-all-work-together)
6. [Implementation Details](#implementation-details)
7. [How to Code This Yourself](#how-to-code-this-yourself)

---

## What is BrowserGym?

**BrowserGym** is a framework for testing AI agents on web tasks. Think of it like a video game where:
- **The game world** = A web browser (like Chrome)
- **The player** = An AI agent (a program that tries to complete tasks)
- **The tasks** = Things like "click this button", "fill out this form", "book a flight"

### Key Components:

1. **Environment**: A web browser (using Playwright) that the agent can interact with
2. **Observation**: What the agent "sees" (screenshots, HTML, accessibility tree)
3. **Action**: What the agent "does" (click, type, navigate)
4. **Reward**: Feedback on whether the agent did well (success/failure)

### Example Flow:
```
1. Environment shows a web page with a button
2. Agent observes the page (sees the button)
3. Agent decides to click the button → Action: "click('button_id')"
4. Environment executes the click
5. Agent observes the result (new page appears)
6. Environment gives reward (1.0 if task completed, 0.0 otherwise)
```

---

## What is the Green Evaluator (Green Agent)?

The **Green Evaluator** is like a **teacher** or **test administrator**. It doesn't do the tasks itself—instead, it:

1. **Takes other agents** (called "White Agents" or "Demo Agents") as input
2. **Runs them through benchmark tasks** (like MiniWoB challenges)
3. **Evaluates their performance** (did they succeed? how many steps? what reward?)
4. **Generates reports** (success rate, average steps, etc.)

### Why "Green"?
In testing terminology:
- **Green** = Test/Evaluation framework (the thing that tests others)
- **White** = The agent being tested (the thing that does the work)

Think of it like:
- **Green Evaluator** = The exam proctor (watches, evaluates, grades)
- **White Agent** = The student (takes the exam, does the work)

### What the Green Evaluator Does:

```python
# Pseudocode of what Green Evaluator does:
def evaluate_agent(agent, task):
    # 1. Set up the test environment
    environment = create_browsergym_environment(task)
    
    # 2. Run the agent through the task
    observation = environment.reset()
    steps = 0
    success = False
    
    while steps < max_steps:
        # Get action from the agent
        action = agent.get_action(observation)
        
        # Execute action in environment
        observation, reward, done = environment.step(action)
        steps += 1
        
        # Check if task is complete
        if done and reward > 0:
            success = True
            break
    
    # 3. Return evaluation results
    return {
        "success": success,
        "steps": steps,
        "reward": reward
    }
```

---

## What is A2A (Agent-to-Agent)?

**A2A** stands for **Agent-to-Agent** protocol. It's a way for different agents to communicate with each other over HTTP (the same protocol web browsers use).

### The Problem It Solves:

Normally, when you want to test an agent, you would:
1. Import the agent's code directly into your evaluation script
2. Create an instance of the agent
3. Call its methods directly

```python
# Direct mode (old way):
from demo_agent.agent import DemoAgent

agent = DemoAgent(...)
action = agent.get_action(observation)  # Direct function call
```

**But what if:**
- The agent is running on a different computer?
- The agent is written in a different language?
- The agent is running as a separate service?
- You want to test multiple agents without changing code?

**A2A solves this** by allowing agents to communicate over HTTP, like web APIs!

### How A2A Works:

Instead of direct function calls, agents communicate via HTTP requests:

```
Green Evaluator (Client)          White Agent (Server)
      |                                    |
      |  HTTP POST /get_action             |
      |  { "obs": {...} }                  |
      |----------------------------------->|
      |                                    | Process observation
      |                                    | Generate action
      |  HTTP 200 OK                       |
      |  { "action": "click('12')",        |
      |    "agent_info": {...} }           |
      |<-----------------------------------|
      |                                    |
```

### A2A Protocol Endpoints:

A White Agent (server) must implement these endpoints:

1. **`GET /health`** - Health check (is the agent running?)
2. **`POST /get_action`** - Get an action from the agent
   - Input: `{ "obs": {...} }` (observation dictionary)
   - Output: `{ "action": "...", "agent_info": {...} }`
3. **`GET /action_set`** - Get information about the agent's action set
4. **`GET /info`** - Get information about the agent

A Green Evaluator (client) uses these endpoints to communicate with the agent.

---

## Why Do We Need A2A?

### 1. **Separation of Concerns**
- Green Evaluator focuses on evaluation
- White Agent focuses on task completion
- They don't need to know each other's internal details

### 2. **Flexibility**
- Test agents written in different languages (Python, JavaScript, etc.)
- Test agents running on different machines
- Test agents as separate services (microservices architecture)

### 3. **Scalability**
- Run multiple agents in parallel
- Test agents remotely (cloud, different servers)
- Integrate with agent platforms (like AgentBeats)

### 4. **Isolation**
- Agents can't break the evaluator (separate processes)
- Easy to restart agents without affecting evaluator
- Better error handling (if agent crashes, evaluator continues)

### 5. **Platform Integration**
- Integrate with agent hosting platforms
- Test agents deployed as web services
- Enable distributed evaluation systems

---

## How Does It All Work Together?

### Architecture Overview:

```
┌─────────────────────────────────────────────────────────────┐
│                    Green Evaluator                          │
│  (Green Agent - Test Administrator)                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  GreenEvaluator Class                                │  │
│  │  - Sets up test environment                          │  │
│  │  - Runs agent through tasks                          │  │
│  │  - Collects metrics                                  │  │
│  │  - Generates reports                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          │ HTTP (A2A Protocol)              │
│                          ▼                                  │
└─────────────────────────────────────────────────────────────┘
                          │
                          │
┌─────────────────────────────────────────────────────────────┐
│                    White Agent Server                       │
│  (Demo Agent - Task Executor)                              │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Flask HTTP Server                                   │  │
│  │  - /health endpoint                                  │  │
│  │  - /get_action endpoint                              │  │
│  │  - /action_set endpoint                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          │                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  DemoAgent Class                                     │  │
│  │  - obs_preprocessor()                                │  │
│  │  - get_action()                                      │  │
│  │  - Uses OpenAI API                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          │
┌─────────────────────────────────────────────────────────────┐
│              BrowserGym Environment                         │
│  (MiniWoB Tasks - The Test)                                │
│                                                             │
│  - Web browser (Playwright)                                │
│  - Task: "click-dialog", "choose-list", etc.               │
│  - Returns: observations, rewards, done flags              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Complete Flow:

1. **Setup Phase:**
   ```
   Green Evaluator starts
   ↓
   Loads White Agent (via A2A or Direct Import)
   ↓
   Sets up BrowserGym environment for task
   ```

2. **Evaluation Loop:**
   ```
   Environment.reset() → Get initial observation
   ↓
   Green Evaluator sends observation to White Agent (via A2A)
   ↓
   White Agent processes observation → Generates action
   ↓
   White Agent returns action to Green Evaluator
   ↓
   Green Evaluator executes action in environment
   ↓
   Environment returns: new observation, reward, done flag
   ↓
   Repeat until done or max_steps reached
   ```

3. **Results Phase:**
   ```
   Green Evaluator collects metrics:
   - Success: Did agent complete task?
   - Steps: How many steps taken?
   - Reward: What was the total reward?
   ↓
   Generate evaluation report
   ↓
   Save results to file
   ```

---

## Implementation Details

### 1. Green Evaluator (`green_evaluator.py`)

#### Key Classes:

**`GreenEvaluator`**: Main evaluation class
- `load_agent()`: Load agent from Python file (Direct Mode)
- `load_agent_a2a()`: Load agent via A2A protocol (A2A Mode)
- `evaluate_agent_on_task()`: Evaluate agent on single task
- `evaluate_agent_on_benchmark_suite()`: Evaluate agent on multiple tasks

**`AgentAdapter`**: Abstract interface for agents
- `DirectAgentAdapter`: Wraps directly imported agents
- `A2AAgentAdapter`: Wraps A2A client connections

#### How It Works:

```python
class GreenEvaluator:
    def evaluate_agent_on_task(self, agent, task_name, max_steps=50):
        # 1. Set up environment
        env = create_browsergym_environment(task_name)
        
        # 2. Reset environment
        obs, info = env.reset()
        obs_preprocessed = agent.obs_preprocessor(obs)
        
        # 3. Run agent loop
        steps = 0
        while steps < max_steps:
            # Get action from agent (via A2A or direct)
            action, agent_info = agent.get_action(obs_preprocessed)
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            obs_preprocessed = agent.obs_preprocessor(obs)
            
            steps += 1
            
            # Check if task complete
            if done:
                success = True
                break
        
        # 4. Return results
        return {
            "success": success,
            "steps": steps,
            "reward": reward
        }
```

### 2. A2A Client (`green_evaluator_a2a_client.py`)

#### Key Classes:

**`A2AAgentClient`**: HTTP client for communicating with White Agents
- `health_check()`: Check if agent is running
- `get_action(obs)`: Send observation, get action back
- `get_action_set_info()`: Get action set information
- `_prepare_payload(obs)`: Convert observation to JSON (handles images, numpy arrays)

#### How It Works:

```python
class A2AAgentClient:
    def get_action(self, obs):
        # 1. Prepare observation for JSON
        payload = self._prepare_payload(obs)
        # Converts:
        # - numpy arrays → lists
        # - PIL images → base64 strings
        # - nested dicts → JSON-compatible structures
        
        # 2. Send HTTP POST request
        response = self.session.post(
            f"{self.agent_url}/get_action",
            json=payload,
            timeout=self.timeout
        )
        
        # 3. Parse response
        result = response.json()
        action = result["action"]
        agent_info = result["agent_info"]
        
        # 4. Return action
        return action, agent_info
```

### 3. White Agent Server (`example_white_agent_server.py`)

#### Key Components:

**Flask HTTP Server**: Exposes A2A endpoints
- `GET /health`: Health check
- `POST /get_action`: Get action from agent
- `GET /action_set`: Get action set info
- `GET /info`: Get agent information

#### How It Works:

```python
# Create Flask app
app = Flask(__name__)

# Initialize agent
agent = DemoAgent(...)

# Define endpoints
@app.route('/get_action', methods=['POST'])
def get_action():
    # 1. Get observation from request
    data = request.get_json()
    obs = data['obs']
    
    # 2. Preprocess observation
    obs_preprocessed = agent.obs_preprocessor(obs)
    
    # 3. Get action from agent
    action, agent_info = agent.get_action(obs_preprocessed)
    
    # 4. Return action
    return jsonify({
        "action": action,
        "agent_info": agent_info
    })
```

### 4. Agent Adapter Pattern

The **Adapter Pattern** allows the Green Evaluator to work with both:
- **Direct agents** (imported directly)
- **A2A agents** (communicated via HTTP)

Both implement the same `AgentAdapter` interface:

```python
class AgentAdapter(ABC):
    @abstractmethod
    def get_action(self, obs) -> tuple[str, dict]:
        pass
    
    @abstractmethod
    def obs_preprocessor(self, obs) -> Any:
        pass
    
    @property
    @abstractmethod
    def action_set(self) -> AbstractActionSet:
        pass
```

**DirectAgentAdapter**: Wraps directly imported agents
```python
class DirectAgentAdapter(AgentAdapter):
    def __init__(self, agent: Agent):
        self.agent = agent
    
    def get_action(self, obs):
        return self.agent.get_action(obs)  # Direct call
```

**A2AAgentAdapter**: Wraps A2A client
```python
class A2AAgentAdapter(AgentAdapter):
    def __init__(self, client: A2AAgentClient):
        self.client = client
    
    def get_action(self, obs):
        return self.client.get_action(obs)  # HTTP call
```

This allows the Green Evaluator to use the same code for both modes!

---

## How to Code This Yourself

### Step 1: Understand the Basics

#### 1.1. HTTP Communication (Flask + Requests)

**Flask** (server side):
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/get_action', methods=['POST'])
def get_action():
    data = request.get_json()  # Get JSON from request
    obs = data['obs']
    
    # Process observation
    action = process_observation(obs)
    
    # Return JSON response
    return jsonify({"action": action})
```

**Requests** (client side):
```python
import requests

# Send POST request
response = requests.post(
    "http://localhost:5002/get_action",
    json={"obs": observation},
    timeout=30
)

# Get response
result = response.json()
action = result["action"]
```

#### 1.2. JSON Serialization

**Problem**: Observations contain non-JSON types (numpy arrays, PIL images)

**Solution**: Convert to JSON-compatible types:
```python
import numpy as np
import base64
from PIL import Image
import io

def prepare_payload(obs):
    payload = {}
    
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            # Convert numpy array to list
            payload[key] = value.tolist()
        elif isinstance(value, Image.Image):
            # Convert PIL image to base64
            buffer = io.BytesIO()
            value.save(buffer, format="JPEG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            payload[key] = f"data:image/jpeg;base64,{image_base64}"
        elif isinstance(value, dict):
            # Recursively process nested dicts
            payload[key] = prepare_payload(value)
        else:
            # Keep as is (str, int, float, bool, None)
            payload[key] = value
    
    return payload
```

### Step 2: Implement A2A Client

```python
import requests
from typing import Dict, Any, Tuple

class A2AAgentClient:
    def __init__(self, agent_url: str, timeout: int = 30):
        self.agent_url = agent_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        try:
            response = self.session.get(
                f"{self.agent_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def get_action(self, obs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        # Prepare payload
        payload = self._prepare_payload(obs)
        
        # Send request
        response = self.session.post(
            f"{self.agent_url}/get_action",
            json=payload,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        return result["action"], result.get("agent_info", {})
    
    def _prepare_payload(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # Convert observation to JSON-compatible format
        # (implementation from above)
        pass
```

### Step 3: Implement White Agent Server

```python
from flask import Flask, request, jsonify
from demo_agent.agent import DemoAgent

app = Flask(__name__)

# Initialize agent
agent = DemoAgent(
    model_name='gpt-4o-mini',
    chat_mode=False,
    demo_mode='off',
    use_html=False,
    use_axtree=True,
    use_screenshot=False
)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/get_action', methods=['POST'])
def get_action():
    try:
        # Get observation from request
        data = request.get_json()
        obs = data.get('obs', {})
        
        # Preprocess observation
        obs_preprocessed = agent.obs_preprocessor(obs)
        
        # Get action from agent
        action, agent_info = agent.get_action(obs_preprocessed)
        
        # Return action
        return jsonify({
            "action": action or "",
            "agent_info": agent_info if isinstance(agent_info, dict) else {}
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
```

### Step 4: Implement Green Evaluator

```python
from green_evaluator_a2a_client import A2AAgentClient
from browsergym.experiments import EnvArgs, ExpArgs

class GreenEvaluator:
    def __init__(self, results_dir: str = "./green_evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def load_agent_a2a(self, agent_url: str):
        """Load agent via A2A protocol"""
        client = A2AAgentClient(agent_url)
        
        # Health check
        if not client.health_check():
            raise ValueError(f"Agent at {agent_url} failed health check")
        
        return A2AAgentAdapter(client)
    
    def evaluate_agent_on_task(self, agent, task_name: str, max_steps: int = 50):
        """Evaluate agent on single task"""
        # Set up environment
        env_args = EnvArgs(
            task_name=task_name,
            task_seed=None,
            max_steps=max_steps,
            headless=True,
        )
        
        # Create environment
        env = env_args.make_env(
            action_mapping=agent.action_set.to_python_code
        )
        
        # Reset environment
        obs, info = env.reset()
        obs_preprocessed = agent.obs_preprocessor(obs)
        
        # Run agent loop
        steps = 0
        total_reward = 0
        success = False
        
        while steps < max_steps:
            # Get action from agent
            action, agent_info = agent.get_action(obs_preprocessed.copy())
            
            if action is None:
                break
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            obs_preprocessed = agent.obs_preprocessor(obs)
            
            total_reward += reward
            steps += 1
            
            # Check if task complete
            if done:
                success = True
                break
            elif truncated:
                break
        
        # Close environment
        env.close()
        
        # Return results
        return {
            "task_name": task_name,
            "success": success,
            "steps": steps,
            "total_reward": total_reward
        }
```

### Step 5: Put It All Together

**Terminal 1 - Start White Agent Server:**
```bash
python3 example_white_agent_server.py --port 5002
```

**Terminal 2 - Run Green Evaluator:**
```bash
python3 green_evaluator.py --agent_url http://localhost:5002 --task miniwob.click-dialog
```

### Step 6: Test Your Implementation

1. **Test A2A Client:**
```python
from green_evaluator_a2a_client import A2AAgentClient

client = A2AAgentClient("http://localhost:5002")

# Health check
if client.health_check():
    print("Agent is healthy!")
    
    # Get action
    obs = {"goal_object": [{"role": "user", "content": "Click the button"}], ...}
    action, info = client.get_action(obs)
    print(f"Action: {action}")
```

2. **Test White Agent Server:**
```bash
# Start server
python3 example_white_agent_server.py --port 5002

# Test health endpoint
curl http://localhost:5002/health

# Test get_action endpoint
curl -X POST http://localhost:5002/get_action \
  -H "Content-Type: application/json" \
  -d '{"obs": {"goal_object": [{"role": "user", "content": "Click the button"}]}}'
```

3. **Test Green Evaluator:**
```python
from green_evaluator import GreenEvaluator

evaluator = GreenEvaluator()

# Load agent via A2A
agent = evaluator.load_agent_a2a("http://localhost:5002")

# Evaluate on task
result = evaluator.evaluate_agent_on_task(agent, "miniwob.click-dialog")
print(f"Success: {result['success']}")
print(f"Steps: {result['steps']}")
print(f"Reward: {result['total_reward']}")
```

---

## Key Concepts Summary

### 1. **Agent Interface**
All agents must implement:
- `obs_preprocessor(obs)`: Process observation before use
- `get_action(obs)`: Return action given observation
- `action_set`: Define what actions are available

### 2. **A2A Protocol**
Standard HTTP endpoints:
- `GET /health`: Health check
- `POST /get_action`: Get action from agent
- `GET /action_set`: Get action set info
- `GET /info`: Get agent information

### 3. **Adapter Pattern**
Allows Green Evaluator to work with:
- Direct agents (imported directly)
- A2A agents (communicated via HTTP)

Both implement the same `AgentAdapter` interface.

### 4. **JSON Serialization**
Observations must be converted to JSON-compatible formats:
- Numpy arrays → Lists
- PIL images → Base64 strings
- Nested dicts → Recursively processed

### 5. **Evaluation Loop**
Standard evaluation flow:
1. Reset environment → Get initial observation
2. Agent processes observation → Generates action
3. Execute action in environment → Get new observation
4. Repeat until task complete or max steps reached
5. Collect metrics → Generate report

---

## Common Pitfalls and Solutions

### 1. **Image Serialization**
**Problem**: PIL images can't be directly serialized to JSON

**Solution**: Convert to base64 strings
```python
import base64
import io
from PIL import Image

def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()
```

### 2. **Numpy Array Serialization**
**Problem**: Numpy arrays can't be directly serialized to JSON

**Solution**: Convert to lists
```python
import numpy as np

def numpy_to_list(arr: np.ndarray) -> list:
    if arr.ndim == 0:
        return arr.item()
    return arr.tolist()
```

### 3. **Timeout Issues**
**Problem**: Agent takes too long to respond

**Solution**: Set appropriate timeouts
```python
# Client side
client = A2AAgentClient(agent_url, timeout=60)  # 60 seconds

# Server side
response = requests.post(..., timeout=30)  # 30 seconds
```

### 4. **Error Handling**
**Problem**: Agent crashes or returns errors

**Solution**: Implement proper error handling
```python
try:
    action, info = agent.get_action(obs)
except Exception as e:
    logger.error(f"Error getting action: {e}")
    # Handle error (retry, skip, etc.)
```

### 5. **Health Checks**
**Problem**: Agent not responding

**Solution**: Implement health checks
```python
if not client.health_check():
    raise ValueError("Agent is not healthy")
```

---

## Next Steps

1. **Read the code**: Study `green_evaluator.py`, `green_evaluator_a2a_client.py`, and `example_white_agent_server.py`
2. **Run examples**: Try running the examples in the README
3. **Modify agents**: Create your own agents and test them
4. **Extend functionality**: Add new endpoints, metrics, or features
5. **Debug issues**: Use logging and error handling to debug problems

---

## Resources

- **BrowserGym Documentation**: See `README.md` and `docs/`
- **Flask Documentation**: https://flask.palletsprojects.com/
- **Requests Documentation**: https://requests.readthedocs.io/
- **HTTP Protocol**: https://developer.mozilla.org/en-US/docs/Web/HTTP
- **JSON Serialization**: https://docs.python.org/3/library/json.html

---

## Conclusion

The Green Evaluator and A2A functionality provide a flexible, scalable way to test AI agents on web tasks. By separating concerns (evaluation vs. task execution) and using HTTP communication, we can:

- Test agents written in different languages
- Test agents running on different machines
- Test agents as separate services
- Scale evaluation to multiple agents
- Integrate with agent platforms

The key is understanding:
1. **What each component does** (Green Evaluator, White Agent, BrowserGym)
2. **How they communicate** (A2A protocol over HTTP)
3. **How to implement it** (Flask server, HTTP client, JSON serialization)

With this knowledge, you can build your own evaluation systems and test your own agents!

