# Code Walkthrough: Green Evaluator and A2A

This document walks through the actual code execution step by step.

## Scenario: Testing an Agent via A2A

Let's trace what happens when you run:
```bash
python3 green_evaluator.py --agent_url http://localhost:5002 --task miniwob.click-dialog
```

---

## Step 1: Main Function Starts

**File**: `green_evaluator.py`, line 703

```python
def main():
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()
    
    # Initialize the Green Evaluator
    evaluator = GreenEvaluator(results_dir=args.results_dir)
```

**What happens**:
1. Parses command line arguments
2. Creates a `GreenEvaluator` instance
3. Sets up logging and environment variables

---

## Step 2: GreenEvaluator Initialization

**File**: `green_evaluator.py`, line 134

```python
class GreenEvaluator:
    def __init__(self, results_dir: str = "./green_evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Load required environment variables
        self._load_environment_variables()
```

**What happens**:
1. Creates results directory
2. Sets up logging (file + console)
3. Loads environment variables from `.env` file (MINIWOB_URL, OPENAI_API_KEY)

---

## Step 3: Load Agent via A2A

**File**: `green_evaluator.py`, line 796

```python
if args.agent_url:
    # A2A Mode
    evaluator.logger.info("Running in A2A Mode")
    agent = evaluator.load_agent_a2a(args.agent_url)
```

**What happens**: Calls `load_agent_a2a()` method

---

## Step 4: load_agent_a2a() Method

**File**: `green_evaluator.py`, line 316

```python
def load_agent_a2a(self, agent_url: str) -> AgentAdapter:
    self.logger.info(f"Loading agent via A2A from: {agent_url}")
    
    try:
        # Create A2A client
        client = A2AAgentClient(agent_url)
        
        # Perform health check
        if not client.health_check():
            raise ValueError(f"Agent at {agent_url} failed health check")
        
        self.logger.info(f"Successfully connected to agent at {agent_url}")
        return A2AAgentAdapter(client)
```

**What happens**:
1. Creates an `A2AAgentClient` instance
2. Performs health check (calls `GET /health` endpoint)
3. If healthy, returns an `A2AAgentAdapter` wrapping the client

---

## Step 5: A2AAgentClient Initialization

**File**: `green_evaluator_a2a_client.py`, line 27

```python
class A2AAgentClient:
    def __init__(self, agent_url: str, timeout: int = 30):
        self.agent_url = agent_url.rstrip('/')
        self.timeout = timeout
        
        # Set up session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
```

**What happens**:
1. Stores agent URL and timeout
2. Creates a `requests.Session` with retry strategy
3. Sets up HTTP adapters for automatic retries on server errors

---

## Step 6: Health Check

**File**: `green_evaluator_a2a_client.py`, line 51

```python
def health_check(self) -> bool:
    try:
        response = self.session.get(
            f"{self.agent_url}/health",
            timeout=5
        )
        if response.status_code == 200:
            logger.info(f"Health check passed for {self.agent_url}")
            return True
        else:
            logger.warning(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False
```

**What happens**:
1. Sends `GET` request to `http://localhost:5002/health`
2. White Agent Server responds with `{"status": "healthy"}`
3. If status code is 200, returns `True`

**On White Agent Server side** (`example_white_agent_server.py`, line 53):
```python
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "service": "WhiteAgent",
        "agent_type": agent.__class__.__name__
    }), 200
```

---

## Step 7: A2AAgentAdapter Creation

**File**: `green_evaluator.py`, line 96

```python
class A2AAgentAdapter(AgentAdapter):
    def __init__(self, client: A2AAgentClient):
        self.client = client
        # We'll use default action set if we can't get it from agent
        self._action_set = DEFAULT_ACTION_SET
        # Try to get action set info
        self._try_get_action_set_info()
```

**What happens**:
1. Wraps the A2A client
2. Sets default action set
3. Optionally tries to get action set info from agent

---

## Step 8: Evaluate Agent on Task

**File**: `green_evaluator.py`, line 806

```python
if args.task:
    # Evaluate on single task
    evaluator.logger.info(f"Single task evaluation: {args.task}")
    result = evaluator.evaluate_agent_on_task(agent, args.task, args.max_steps)
```

**What happens**: Calls `evaluate_agent_on_task()` method

---

## Step 9: evaluate_agent_on_task() Method - Setup

**File**: `green_evaluator.py`, line 351

```python
def evaluate_agent_on_task(self, agent: AgentAdapter, task_name: str, max_steps: int = 50):
    self.logger.info(f"Evaluating agent on task: {task_name}")
    
    # Set up environment arguments for the benchmark task
    env_args = EnvArgs(
        task_name=task_name,
        task_seed=None,
        max_steps=max_steps,
        headless=True,
        record_video=False,
        wait_for_user_message=False,
    )
    
    # Set up experiment arguments
    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=None,
    )
    
    # Create a unique experiment directory
    timestamp = int(time.time())
    exp_dir = self.results_dir / f"eval_{task_name}_{timestamp}"
    exp_args.exp_dir = str(exp_dir)
    exp_args.exp_name = f"GreenEval_{task_name}_{timestamp}"
```

**What happens**:
1. Creates `EnvArgs` for the task (miniwob.click-dialog)
2. Creates `ExpArgs` for the experiment
3. Creates a unique directory for this evaluation run
4. Sets up experiment directory and name

---

## Step 10: Create BrowserGym Environment

**File**: `green_evaluator.py`, line 395

```python
# Prepare the experiment
exp_args.prepare(str(self.results_dir))

# Create environment
env = env_args.make_env(
    action_mapping=agent.action_set.to_python_code,
    exp_dir=exp_dir,
)
```

**What happens**:
1. Prepares the experiment (creates directories, etc.)
2. Creates a BrowserGym environment for the task
3. Sets up action mapping (converts actions to Python code)

---

## Step 11: Reset Environment

**File**: `green_evaluator.py`, line 406

```python
# Reset environment
obs, env_info = env.reset()
obs_preprocessed = agent.obs_preprocessor(obs)
```

**What happens**:
1. Resets the BrowserGym environment
2. Gets initial observation (screenshot, HTML, accessibility tree, goal, etc.)
3. Preprocesses observation using agent's preprocessor

**Observation structure**:
```python
obs = {
    "chat_messages": [],
    "goal_object": [{"role": "user", "content": "Click the button"}],
    "screenshot": numpy.ndarray,  # Image of the page
    "dom_object": dict,  # DOM structure
    "axtree_object": dict,  # Accessibility tree
    "open_pages_urls": ["http://..."],
    "open_pages_titles": ["Page Title"],
    "active_page_index": 0,
    "last_action": "",
    "last_action_error": "",
    ...
}
```

**Preprocessing** (for A2A, this just returns the observation as-is):
```python
# A2AAgentAdapter.obs_preprocessor()
def obs_preprocessor(self, obs: dict) -> Any:
    # For A2A, we'll do basic preprocessing here
    return obs
```

---

## Step 12: Agent Loop - Get Action

**File**: `green_evaluator.py`, line 410

```python
# Run the agent
while step_count < max_steps:
    try:
        # Get agent's action
        action, agent_info = agent.get_action(obs_preprocessed.copy())
```

**What happens**: Calls `agent.get_action()` which for A2A goes through the adapter

---

## Step 13: A2AAgentAdapter.get_action()

**File**: `green_evaluator.py`, line 119

```python
def get_action(self, obs: Any) -> tuple[str, Dict[str, Any]]:
    # A2A clients expect the preprocessed observation
    return self.client.get_action(obs)
```

**What happens**: Calls the A2A client's `get_action()` method

---

## Step 14: A2AAgentClient.get_action() - Prepare Payload

**File**: `green_evaluator_a2a_client.py`, line 73

```python
def get_action(self, obs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    try:
        # Prepare request payload
        # Convert numpy arrays to lists for JSON serialization
        payload = self._prepare_payload(obs)
```

**What happens**: Calls `_prepare_payload()` to convert observation to JSON-compatible format

---

## Step 15: _prepare_payload() Method

**File**: `green_evaluator_a2a_client.py`, line 142

```python
def _prepare_payload(self, obs: Dict[str, Any]) -> Dict[str, Any]:
    payload = {}
    
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            # Convert numpy arrays to lists
            if value.ndim == 0:
                payload[key] = value.item()
            else:
                payload[key] = value.tolist()
        elif isinstance(value, (Image.Image,)):
            # Convert PIL images to base64 strings
            import base64
            import io
            buffer = io.BytesIO()
            if value.mode in ("RGBA", "LA"):
                value = value.convert("RGB")
            value.save(buffer, format="JPEG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            payload[key] = f"data:image/jpeg;base64,{image_base64}"
        elif isinstance(value, dict):
            # Recursively process nested dictionaries
            payload[key] = self._prepare_payload(value)
        # ... handle lists, etc.
```

**What happens**:
1. Iterates through observation dictionary
2. Converts numpy arrays to lists
3. Converts PIL images to base64 strings
4. Recursively processes nested dictionaries and lists
5. Returns JSON-compatible payload

**Example conversion**:
```python
# Before:
obs = {
    "screenshot": numpy.ndarray(...),  # Can't serialize to JSON
    "goal_object": [{"role": "user", "content": "Click button"}],
}

# After:
payload = {
    "screenshot": [[[255, 255, 255], ...], ...],  # List of lists
    "goal_object": [{"role": "user", "content": "Click button"}],
}
```

---

## Step 16: A2AAgentClient.get_action() - Send HTTP Request

**File**: `green_evaluator_a2a_client.py`, line 94

```python
# Make POST request to /get_action endpoint
response = self.session.post(
    f"{self.agent_url}/get_action",
    json=payload,
    timeout=self.timeout,
    headers={"Content-Type": "application/json"}
)

response.raise_for_status()

# Parse response
result = response.json()

action = result.get("action", "")
agent_info = result.get("agent_info", {})

return action, agent_info
```

**What happens**:
1. Sends `POST` request to `http://localhost:5002/get_action`
2. Includes observation as JSON in request body
3. Waits for response (with timeout)
4. Parses JSON response
5. Extracts action and agent_info
6. Returns action and agent_info

**HTTP Request**:
```
POST http://localhost:5002/get_action
Content-Type: application/json

{
    "obs": {
        "goal_object": [{"role": "user", "content": "Click the button"}],
        "screenshot": "data:image/jpeg;base64,...",
        "axtree_txt": "...",
        ...
    }
}
```

---

## Step 17: White Agent Server - Receive Request

**File**: `example_white_agent_server.py`, line 62

```python
@app.route('/get_action', methods=['POST'])
def get_action():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        obs = data.get('obs', {})
        
        if not obs:
            return jsonify({"error": "No observation provided"}), 400
```

**What happens**:
1. Flask receives `POST` request to `/get_action`
2. Extracts JSON data from request
3. Gets observation from `data['obs']`

---

## Step 18: White Agent Server - Preprocess Observation

**File**: `example_white_agent_server.py`, line 89

```python
# Preprocess observation using agent's preprocessor
logger.debug("Preprocessing observation...")
obs_preprocessed = agent.obs_preprocessor(obs)
```

**What happens**: Calls agent's `obs_preprocessor()` method

**On DemoAgent side** (`demo_agent/agent.py`, line 36):
```python
def obs_preprocessor(self, obs: dict) -> dict:
    return {
        "chat_messages": obs["chat_messages"],
        "screenshot": obs["screenshot"],
        "goal_object": obs["goal_object"],
        "last_action": obs["last_action"],
        "last_action_error": obs["last_action_error"],
        "open_pages_urls": obs["open_pages_urls"],
        "open_pages_titles": obs["open_pages_titles"],
        "active_page_index": obs["active_page_index"],
        "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
        "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
    }
```

**What happens**:
1. Extracts relevant fields from observation
2. Converts accessibility tree to text
3. Converts DOM to pruned HTML text
4. Returns preprocessed observation

---

## Step 19: White Agent Server - Get Action from Agent

**File**: `example_white_agent_server.py`, line 94

```python
# Get action from agent
logger.debug("Getting action from agent...")
action, agent_info = agent.get_action(obs_preprocessed)
```

**What happens**: Calls agent's `get_action()` method

**On DemoAgent side** (`demo_agent/agent.py`, line 84):
```python
def get_action(self, obs: dict) -> tuple[str, dict]:
    # Build prompt for OpenAI
    system_msgs = [...]
    user_msgs = [...]
    
    # Append goal, observation, action space, etc.
    # ...
    
    # Query OpenAI model
    response = self.openai_client.chat.completions.create(
        model=self.model_name,
        messages=[
            {"role": "system", "content": system_msgs},
            {"role": "user", "content": user_msgs},
        ],
    )
    action = response.choices[0].message.content
    
    return action, {}
```

**What happens**:
1. Builds prompt with goal, observation, action space, history
2. Sends prompt to OpenAI API
3. Gets response from OpenAI
4. Extracts action from response
5. Returns action and agent_info

**Example action**:
```
click("12")
```

---

## Step 20: White Agent Server - Return Response

**File**: `example_white_agent_server.py`, line 104

```python
# Convert agent_info to dict if it's not already
if hasattr(agent_info, '__dict__'):
    agent_info = agent_info.__dict__
elif not isinstance(agent_info, dict):
    agent_info = {}

return jsonify({
    "action": action or "",
    "agent_info": agent_info
}), 200
```

**What happens**:
1. Converts agent_info to dictionary
2. Returns JSON response with action and agent_info
3. Flask sends HTTP 200 response

**HTTP Response**:
```
HTTP/1.1 200 OK
Content-Type: application/json

{
    "action": "click(\"12\")",
    "agent_info": {}
}
```

---

## Step 21: Green Evaluator - Receive Action

**File**: `green_evaluator.py`, line 413

```python
# Get agent's action
action, agent_info = agent.get_action(obs_preprocessed.copy())

if action is None:
    self.logger.info("Agent returned None action, ending evaluation")
    break
```

**What happens**:
1. Receives action from A2A client
2. Checks if action is None (if so, break loop)
3. Otherwise, continues to execute action

---

## Step 22: Green Evaluator - Execute Action

**File**: `green_evaluator.py`, line 419

```python
# Execute action in environment
obs, reward, terminated, truncated, env_info = env.step(action)
obs_preprocessed = agent.obs_preprocessor(obs)
```

**What happens**:
1. Executes action in BrowserGym environment
2. Environment performs the action (clicks button, types text, etc.)
3. Gets new observation, reward, and done flags
4. Preprocesses new observation

**Action execution**:
- Action: `click("12")`
- Environment: Finds element with bid="12", clicks it
- Result: New page loads, new observation returned

---

## Step 23: Green Evaluator - Check Task Completion

**File**: `green_evaluator.py`, line 423

```python
total_reward += reward
step_count += 1

self.logger.info(f"Step {step_count}: Action='{action[:50]}...', Reward={reward}")

# Check if task is complete
if terminated:
    success = True
    self.logger.info(f"Task completed successfully in {step_count} steps")
    break
elif truncated:
    self.logger.info(f"Task truncated after {step_count} steps")
    break
```

**What happens**:
1. Updates total reward and step count
2. Logs step information
3. Checks if task is complete (`terminated`) or truncated (`truncated`)
4. If complete, breaks loop
5. Otherwise, continues to next step

---

## Step 24: Repeat Loop

**File**: `green_evaluator.py`, line 410

```python
# Run the agent
while step_count < max_steps:
    # Get action
    action, agent_info = agent.get_action(obs_preprocessed.copy())
    
    # Execute action
    obs, reward, terminated, truncated, env_info = env.step(action)
    obs_preprocessed = agent.obs_preprocessor(obs)
    
    # Check completion
    if terminated or truncated:
        break
```

**What happens**: Repeats steps 12-23 until:
- Task is complete (`terminated = True`)
- Task is truncated (`truncated = True`)
- Max steps reached (`step_count >= max_steps`)
- Agent returns None action

---

## Step 25: Close Environment

**File**: `green_evaluator.py`, line 442

```python
# Close environment
env.close()
```

**What happens**:
1. Closes BrowserGym environment
2. Cleans up browser instances
3. Releases resources

---

## Step 26: Compile Results

**File**: `green_evaluator.py`, line 444

```python
# Compile results
result = {
    "task_name": task_name,
    "success": success,
    "steps_taken": step_count,
    "total_reward": total_reward,
    "max_steps": max_steps,
    "exp_dir": str(exp_dir),
    "timestamp": timestamp
}

self.logger.info(f"Evaluation complete: {result}")
return result
```

**What happens**:
1. Creates result dictionary with metrics
2. Logs evaluation results
3. Returns result dictionary

**Example result**:
```python
{
    "task_name": "miniwob.click-dialog",
    "success": True,
    "steps_taken": 3,
    "total_reward": 1.0,
    "max_steps": 50,
    "exp_dir": "./green_evaluation_results/eval_miniwob.click-dialog_1234567890",
    "timestamp": 1234567890
}
```

---

## Step 27: Print Results

**File**: `green_evaluator.py`, line 810

```python
print(f"\nSingle Task Results:")
print(f"Task: {result['task_name']}")
print(f"Success: {result['success']}")
print(f"Steps: {result['steps_taken']}")
print(f"Reward: {result['total_reward']}")
print(f"\nFull JSON Result:")
print(json.dumps(result, indent=2))
```

**What happens**:
1. Prints evaluation results to console
2. Shows task name, success, steps, reward
3. Shows full JSON result

---

## Summary

The complete flow:

1. **Green Evaluator** starts and loads agent via A2A
2. **A2A Client** performs health check on White Agent Server
3. **Green Evaluator** sets up BrowserGym environment
4. **Environment** resets and returns initial observation
5. **A2A Client** sends observation to White Agent Server via HTTP
6. **White Agent Server** preprocesses observation and gets action from agent
7. **Agent** (DemoAgent) processes observation and calls OpenAI API
8. **White Agent Server** returns action to A2A Client
9. **Green Evaluator** executes action in environment
10. **Environment** returns new observation, reward, and done flags
11. **Repeat** steps 5-10 until task complete or max steps reached
12. **Green Evaluator** compiles results and prints report

This is the complete A2A communication flow!

