import argparse
import asyncio
import json
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

from white_agent import WhiteAgentArgs
from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result


def parse_args():
    parser = argparse.ArgumentParser(description="BrowserGym Green Evaluator")
    parser.add_argument("--a2a-server", action="store_true", help="Run as A2A server")
    parser.add_argument("--port", type=int, default=8000, help="Port for A2A server")
    parser.add_argument("--card-url", type=str, default=None, help="Public URL for agent card")
    parser.add_argument("--task", type=str, default=None, help="Task to run (e.g., miniwob.click-test)")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum steps per task")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="Model name for agent")
    return parser.parse_args()


class SendMessageRequest(BaseModel):
    message: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


def create_a2a_app(card_url: str) -> FastAPI:
    """Create FastAPI app with A2A protocol endpoints for green evaluator."""
    app = FastAPI(title="BrowserGym Green Evaluator", version="1.0.0")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    tasks = {}

    @app.get("/.well-known/agent-card.json")
    async def get_agent_card():
        return {
            "protocolVersion": "0.3.0",
            "preferredTransport": "JSONRPC",
            "capabilities": {
                "streaming": False
            },
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
            "name": "BrowserGym Green Evaluator",
            "description": "Assessment hosting agent for BrowserGym web automation benchmarks",
            "url": card_url,
            "version": "1.0.0",
            "skills": [{
                "id": "host_assess_browsergym",
                "name": "BrowserGym Assessment Hosting",
                "description": "Evaluate white agents on web automation benchmarks including MiniWoB, WebArena, WorkArena, and VisualWebArena",
                "examples": [
                    "Assess agent on miniwob.click-test with max 50 steps",
                    "Evaluate agent on webarena task with shopping environment",
                    "Run WorkArena ServiceNow task evaluation",
                    "Test agent on VisualWebArena visual reasoning tasks"
                ],
                "tags": ["green agent", "assessment hosting", "browsergym", "web automation"]
            }]
        }

    @app.get("/")
    async def root():
        return {"name": "BrowserGym Green Evaluator", "version": "1.0.0", "role": "green_agent", "status": "ready"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/status")
    async def status():
        return {"status": "ok"}

    @app.get("/info")
    async def info():
        return {
            "name": "BrowserGym Green Evaluator",
            "version": "1.0.0",
            "type": "green_agent",
            "role": "assessor",
            "description": "Green agent that evaluates white agents on BrowserGym benchmarks",
            "capabilities": [
                "miniwob_evaluation",
                "workarena_evaluation",
                "webarena_evaluation",
                "assessment_orchestration"
            ],
            "url": card_url,
            "endpoints": {
                "agent_card": "/.well-known/agent-card.json",
                "send_message": "/sendMessage",
                "send_message_stream": "/sendMessageStream",
                "get_task": "/getTask",
                "cancel_task": "/cancelTask",
                "status": "/status",
                "health": "/health"
            }
        }

    @app.post("/sendMessage")
    async def send_message(request: SendMessageRequest):
        try:
            task_id = str(uuid.uuid4())
            message = request.message
            
            parts = message.get("parts", [])
            text_content = "".join(
                part.get("text", "") 
                for part in parts 
                if part.get("kind") == "text"
            )
            
            assessment_data = {}
            try:
                assessment_data = json.loads(text_content)
            except:
                pass
            
            participants = assessment_data.get("participants", {})
            config = assessment_data.get("config", {})
            
            tasks[task_id] = {
                "id": task_id,
                "status": "submitted",
                "participants": participants,
                "config": config,
                "artifacts": []
            }
            
            print(f"Assessment {task_id} received with {len(participants)} participants")
            asyncio.create_task(run_assessment(task_id, tasks))
            
            return {
                "task": {
                    "id": task_id,
                    "status": "submitted"
                }
            }
        except Exception as e:
            print(f"Error handling assessment: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/sendMessageStream")
    async def send_message_stream(request: SendMessageRequest):
        try:
            task_id = str(uuid.uuid4())
            message = request.message
            
            parts = message.get("parts", [])
            text_content = "".join(
                part.get("text", "") 
                for part in parts 
                if part.get("kind") == "text"
            )
            
            assessment_data = {}
            try:
                assessment_data = json.loads(text_content)
            except:
                pass
            
            participants = assessment_data.get("participants", {})
            config = assessment_data.get("config", {})
            
            tasks[task_id] = {
                "id": task_id,
                "status": "submitted",
                "participants": participants,
                "config": config,
                "artifacts": []
            }
            
            async def event_stream():
                yield f"data: {json.dumps({'task': {'id': task_id, 'status': 'submitted'}})}\n\n"
                
                await asyncio.sleep(0.1)
                tasks[task_id]["status"] = "working"
                yield f"data: {json.dumps({'task': {'id': task_id, 'status': 'working'}})}\n\n"
                
                await run_assessment(task_id, tasks, stream_updates=True)
                
                for update in tasks[task_id].get("updates", []):
                    yield f"data: {json.dumps(update)}\n\n"
                
                final_status = tasks[task_id]["status"]
                yield f"data: {json.dumps({'task': {'id': task_id, 'status': final_status, 'artifacts': tasks[task_id].get('artifacts', [])}})}\n\n"
            
            return StreamingResponse(event_stream(), media_type="text/event-stream")
        except Exception as e:
            print(f"Error in stream: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/getTask")
    async def get_task(request: Request):
        body = await request.json()
        task_id = body.get("taskId")
        
        if task_id not in tasks:
            return JSONResponse(status_code=404, content={"error": "Task not found"})
        
        task = tasks[task_id]
        return {
            "task": {
                "id": task_id,
                "status": task["status"],
                "artifacts": task.get("artifacts", [])
            }
        }

    @app.post("/cancelTask")
    async def cancel_task(request: Request):
        body = await request.json()
        task_id = body.get("taskId")
        
        if task_id in tasks:
            tasks[task_id]["status"] = "cancelled"
            return {"task": {"id": task_id, "status": "cancelled"}}
        
        return JSONResponse(status_code=404, content={"error": "Task not found"})

    return app


async def run_assessment(task_id: str, tasks: dict, stream_updates: bool = False):
    """Run assessment using BrowserGym and produce A2A artifacts."""
    try:
        task = tasks[task_id]
        task["status"] = "working"
        
        if stream_updates:
            task["updates"] = []
            task["updates"].append({"type": "log", "message": "Starting assessment"})
        
        config = task["config"]
        task_name = config.get("task_name", "miniwob.click-test")
        max_steps = config.get("max_steps", 50)
        model_name = config.get("model_name", "gpt-4o-mini")
        
        print(f"Running assessment {task_id} on {task_name}")
        
        if stream_updates:
            task["updates"].append({"type": "log", "message": f"Evaluating on task: {task_name}"})
        
        agent_args = WhiteAgentArgs(
            model_name=model_name,
            chat_mode=False,
            demo_mode="off",
            use_html=False,
            use_axtree=True,
            use_screenshot=False,
        )
        
        env_args = EnvArgs(
            task_name=task_name,
            task_seed=None,
            max_steps=max_steps,
            headless=True,
        )
        
        exp_args = ExpArgs(env_args=env_args, agent_args=agent_args)
        exp_args.prepare(f"./results/a2a_{task_id}")
        
        if stream_updates:
            task["updates"].append({"type": "log", "message": "Running white agent on benchmark"})
        
        exp_args.run()
        
        exp_result = get_exp_result(exp_args.exp_dir)
        exp_record = exp_result.get_exp_record()
        
        results = {
            "task_name": task_name,
            "success": exp_record.get("cum_reward", 0) > 0,
            "reward": exp_record.get("cum_reward", 0),
            "steps": exp_record.get("n_steps", 0),
            "max_steps": max_steps,
            "model": model_name,
            "details": exp_record
        }
        
        artifacts = [{
            "name": "assessment_results",
            "mimeType": "application/json",
            "data": results
        }]
        
        task["status"] = "completed"
        task["artifacts"] = artifacts
        
        if stream_updates:
            task["updates"].append({
                "type": "artifact",
                "artifact": artifacts[0]
            })
        
        print(f"Assessment {task_id} completed: Success={results['success']}, Reward={results['reward']}")
        
    except Exception as e:
        print(f"Assessment {task_id} failed: {e}")
        task["status"] = "failed"
        task["artifacts"] = [{
            "name": "error",
            "mimeType": "application/json",
            "data": {"error": str(e), "task_name": config.get("task_name", "unknown")}
        }]


def main():
    args = parse_args()

    agent_args = WhiteAgentArgs(
        model_name=args.model_name,
        chat_mode=False,
        demo_mode="off",
        use_html=False,
        use_axtree=True,
        use_screenshot=False,
    )

    if args.task:
        from browsergym.core.env import BrowserEnv
        
        env = BrowserEnv(
            task_name=args.task,
            headless=True,
            action_mapping=agent_args.make_agent().action_set.to_python_code,
        )

        agent = agent_args.make_agent()
        obs, info = env.reset()

        for step in range(args.max_steps):
            action_str, action_meta = agent.get_action(obs)
            print(f"\nStep {step+1}: {action_str}")

            obs, reward, terminated, truncated, info = env.step(action_str)

            if terminated or truncated:
                print(f"\nTask {'succeeded' if reward > 0 else 'failed'} after {step+1} steps")
                break

        env.close()
    else:
        env_args = EnvArgs(task_name="miniwob.click-test", max_steps=args.max_steps)
        exp_args = ExpArgs(env_args=env_args, agent_args=agent_args)
        exp_args.prepare("./results/green_eval")
        exp_args.run()
        
        exp_result = get_exp_result(exp_args.exp_dir)
        print(f"\nEvaluation complete: {exp_result.get_exp_record()}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    args = parse_args()
    
    if args.a2a_server:
        card_url = args.card_url or f"http://localhost:{args.port}"
        app = create_a2a_app(card_url)
        
        print(f"Starting BrowserGym Green Evaluator (A2A Protocol)")
        print(f"Server: {card_url}")
        print(f"Agent Card: {card_url}/.well-known/agent-card.json")
        print(f"Send assessment requests to: {card_url}/sendMessage")
        
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        main()
