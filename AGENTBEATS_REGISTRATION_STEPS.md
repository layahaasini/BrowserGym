# AgentBeats Registration - Step by Step Guide

## Quick Summary: Why Both Tunnel AND A2A?

- **Cloudflare Tunnel** = Makes your server reachable (networking layer)
- **A2A Protocol** = Defines HOW AgentBeats talks to your agent (application layer)

Think of it like:
- Tunnel = Phone line (connectivity)
- A2A = Language/protocol (what to say and how)

Without A2A, AgentBeats could reach your server but wouldn't know how to communicate with it!

---

## Step-by-Step Registration Process

### Step 1: Install Dependencies

Make sure you have the A2A dependencies installed:

```bash
pip install fastapi uvicorn
```

### Step 2: Start Your Agent Server

**Terminal 1** - Start your agent (without tunnel URL first):

```bash
cd /Users/layahaasini/Desktop/projects/BrowserGym
python3 green_evaluator.py --a2a-server --host 127.0.0.1 --port 8000
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Keep this terminal running!**

### Step 3: Start Cloudflare Tunnel

**Terminal 2** - Start the tunnel:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

You'll see output like:
```
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
|  https://abc-123-def-456.trycloudflare.com                                                |
+--------------------------------------------------------------------------------------------+
```

**Copy the URL** (e.g., `https://abc-123-def-456.trycloudflare.com`)

**Keep this terminal running too!**

### Step 4: Restart Agent with Tunnel URL

Go back to **Terminal 1**, stop the server (Ctrl+C), then restart with the tunnel URL:

```bash
python3 green_evaluator.py --a2a-server \
    --host 127.0.0.1 \
    --port 8000 \
    --card-url https://abc-123-def-456.trycloudflare.com
```

(Replace with your actual tunnel URL)

### Step 5: Verify Your Server is Accessible

Test that your agent is reachable through the tunnel:

```bash
# Test health endpoint
curl https://abc-123-def-456.trycloudflare.com/health

# Test agent card
curl https://abc-123-def-456.trycloudflare.com/card

# Test root endpoint
curl https://abc-123-def-456.trycloudflare.com/
```

You should get JSON responses. If you get errors, check:
- Is your agent server running? (Terminal 1)
- Is the tunnel running? (Terminal 2)
- Did you use the correct tunnel URL?

### Step 6: Register on AgentBeats Platform

1. **Go to AgentBeats**: Navigate to [https://agentbeats.org](https://agentbeats.org)

2. **Create Account or Log In**:
   - If you don't have an account, create one
   - If you have an account, log in

3. **Find Agent Registration**:
   - Look for "Register Agent", "Add Agent", or "My Agents" in the navigation
   - This might be in a dashboard or settings page

4. **Fill in Agent Details**:
   - **Agent Name**: "BrowserGym Green Evaluator" (or your preferred name)
   - **Agent URL**: Your tunnel URL (e.g., `https://abc-123-def-456.trycloudflare.com`)
   - **Agent Type**: Select "Green Agent" or "Evaluator"
   - **Description**: "Evaluates web agents on BrowserGym benchmarks (MiniWoB, WebArena, etc.)"
   - **Capabilities**: 
     - MiniWoB benchmark
     - WebArena benchmark
     - Agent evaluation

5. **Submit Registration**:
   - Click "Register" or "Submit"
   - The platform may verify your agent is reachable
   - Wait for confirmation

### Step 7: Verify Registration

After registration:

1. **Check Agent Status**:
   - Your agent should appear in "My Agents" or similar section
   - Status should show as "Online" or "Active"

2. **Test Connection**:
   - AgentBeats may automatically test the connection
   - Or you can manually trigger a test/health check

3. **View Agent Card**:
   - Your agent card should be visible
   - Other users can discover your agent

### Step 8: Create Your First Assessment (Optional)

Once registered, you can:

1. **Create an Assessment**:
   - Select your green evaluator as the evaluator
   - Choose purple agents to evaluate
   - Configure tasks and parameters

2. **Run Assessment**:
   - Start the assessment
   - Monitor progress
   - View results

---

## Troubleshooting

### "Agent not reachable" error

- **Check tunnel is running**: Make sure Terminal 2 (cloudflared) is still active
- **Check agent server**: Make sure Terminal 1 (your agent) is still running
- **Test manually**: Try `curl https://your-tunnel-url/health`
- **Restart tunnel**: If tunnel URL changed, update your agent's `--card-url`

### "Invalid A2A protocol" error

- **Check dependencies**: `pip install fastapi uvicorn`
- **Verify endpoints**: Test `/a2a/message` endpoint
- **Check logs**: Look at your agent server logs for errors

### Tunnel URL keeps changing

- **This is normal**: Cloudflare Tunnel URLs change each time you restart
- **Solution**: Restart your agent with the new URL each time
- **For permanent URL**: Set up a "named tunnel" (more advanced, see Cloudflare docs)

### Agent works locally but not through tunnel

- **Check firewall**: Make sure nothing is blocking the connection
- **Verify tunnel**: Test with `curl` through the tunnel URL
- **Check logs**: Look at both agent and tunnel logs

---

## Important Notes

1. **Keep Both Terminals Running**:
   - Terminal 1: Your agent server
   - Terminal 2: Cloudflare tunnel
   - If either stops, your agent won't be accessible

2. **Tunnel URLs Are Temporary**:
   - Each time you restart the tunnel, you get a new URL
   - You'll need to update your agent's `--card-url` and re-register if the URL changes
   - For production, consider a named tunnel for a permanent URL

3. **Environment Variables**:
   - Make sure `MINIWOB_URL` and `OPENAI_API_KEY` are set
   - These are needed when the agent actually runs evaluations

4. **Testing Locally First**:
   - Test your agent locally before registering
   - Make sure all endpoints work: `/health`, `/card`, `/a2a/message`

---

## Next Steps After Registration

1. **Share Your Agent**: Your agent is now discoverable on AgentBeats
2. **Run Assessments**: Create and run assessments with other agents
3. **Monitor Results**: Track evaluation results and performance
4. **Improve**: Based on feedback, enhance your agent's capabilities

---

## Quick Reference Commands

```bash
# Start agent
python3 green_evaluator.py --a2a-server --host 127.0.0.1 --port 8000 --card-url YOUR_TUNNEL_URL

# Start tunnel
cloudflared tunnel --url http://127.0.0.1:8000

# Test endpoints
curl https://YOUR_TUNNEL_URL/health
curl https://YOUR_TUNNEL_URL/card
curl https://YOUR_TUNNEL_URL/
```

