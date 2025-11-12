# A2A Quick Start Guide

## What You Need to Know in 5 Minutes

### The Problem
You want to test an AI agent on web tasks, but:
- The agent is running on a different computer
- The agent is a separate service
- You want to test multiple agents easily

### The Solution: A2A (Agent-to-Agent)
Agents communicate via HTTP (like web APIs) instead of direct function calls.

### The Three Components

1. **Green Evaluator** (Test Administrator)
   - Tests other agents
   - Runs benchmark tasks
   - Collects metrics

2. **White Agent Server** (The Agent Being Tested)
   - Exposes HTTP endpoints
   - Processes observations
   - Returns actions

3. **BrowserGym Environment** (The Test)
   - Web browser with tasks
   - Returns observations
   - Executes actions

### How It Works

```
┌──────────────────┐
│ Green Evaluator  │
│  (Tests Agent)   │
└────────┬─────────┘
         │
         │ HTTP POST /get_action
         │ { "obs": {...} }
         │
         ▼
┌──────────────────┐
│ White Agent      │
│  (Does Tasks)    │
│                  │
│  /health         │
│  /get_action ◄───┘
│  /action_set     │
└────────┬─────────┘
         │
         │ Action
         │
         ▼
┌──────────────────┐
│ BrowserGym       │
│  Environment     │
│  (Web Browser)   │
└──────────────────┘
```

### Quick Example

**1. Start White Agent Server:**
```bash
python3 example_white_agent_server.py --port 5002
```

**2. Run Green Evaluator:**
```bash
python3 green_evaluator.py --agent_url http://localhost:5002 --task miniwob.click-dialog
```

**3. That's it!** The Green Evaluator will:
- Connect to the White Agent via A2A
- Run it through the task
- Collect metrics
- Generate a report

### Key Code Snippets

**A2A Client (Green Evaluator side):**
```python
import requests

class A2AAgentClient:
    def get_action(self, obs):
        response = requests.post(
            "http://localhost:5002/get_action",
            json={"obs": obs},
            timeout=30
        )
        result = response.json()
        return result["action"], result["agent_info"]
```

**A2A Server (White Agent side):**
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/get_action', methods=['POST'])
def get_action():
    data = request.get_json()
    obs = data['obs']
    
    # Process observation
    action = agent.get_action(obs)
    
    return jsonify({"action": action})
```

### Common Questions

**Q: Why use A2A instead of direct imports?**
A: A2A allows testing agents that are:
- Running on different machines
- Written in different languages
- Deployed as separate services

**Q: What if the agent is on my local machine?**
A: You can still use A2A! Just use `http://localhost:5002` as the agent URL.

**Q: What if I want to test multiple agents?**
A: Start multiple agent servers on different ports, then test them one by one.

**Q: How do I debug A2A communication?**
A: Use logging and check the HTTP responses. You can also use `curl` to test endpoints manually.

### Next Steps

1. Read the full guide: `GREEN_AGENT_AND_A2A_EXPLAINED.md`
2. Run the examples in the codebase
3. Create your own agent server
4. Test your agent with the Green Evaluator

### Troubleshooting

**Problem: Connection refused**
- Solution: Make sure the White Agent server is running

**Problem: Timeout errors**
- Solution: Increase the timeout value in the A2A client

**Problem: JSON serialization errors**
- Solution: Make sure observations are converted to JSON-compatible formats (see full guide)

**Problem: Agent not responding**
- Solution: Check the `/health` endpoint to verify the agent is running

---

For more details, see `GREEN_AGENT_AND_A2A_EXPLAINED.md`

