import argparse
import uuid
import asyncio
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

from white_agent import WhiteAgentArgs
from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="OpenAI model name.")
    parser.add_argument("--task_name", type=str, default="openended", help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'")
    parser.add_argument("--start_url", type=str, default="https://www.google.com", help="Starting URL (only for the openended task).")
    parser.add_argument("--visual_effects", type=str2bool, default=True, help="Add visual effects when the agents performs actions.")
    parser.add_argument("--use_html", type=str2bool, default=False, help="Use HTML in the agent's observation space.")
    parser.add_argument("--use_axtree", type=str2bool, default=True, help="Use AXTree in the agent's observation space.")
    parser.add_argument("--use_screenshot", type=str2bool, default=False, help="Use screenshot in the agent's observation space.")
    parser.add_argument("--a2a-server", action="store_true", help="Run as A2A server")
    parser.add_argument("--port", type=int, default=8000, help="Port for A2A server")
    parser.add_argument("--card-url", type=str, default=None, help="Public URL for agent card")

    return parser.parse_args()


def main():
    args = parse_args()

    agent_args = WhiteAgentArgs(
        model_name=args.model_name,
        chat_mode=False,
        demo_mode="default" if args.visual_effects else "off",
        use_html=args.use_html,
        use_axtree=args.use_axtree,
        use_screenshot=args.use_screenshot,
    )

    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=None,
        max_steps=50,
        headless=False,
    )

    if args.task_name == "openended":
        agent_args.chat_mode = True
        env_args.wait_for_user_message = True
        env_args.task_kwargs = {"start_url": args.start_url}

    exp_args = ExpArgs(env_args=env_args, agent_args=agent_args)
    exp_args.prepare("./results")
    exp_args.run()

    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()

    for key, val in exp_record.items():
        print(f"{key}: {val}")


class AssessmentRequest(BaseModel):
    participants: Dict[str, str]
    config: Dict[str, Any] = {}


def create_a2a_app(card_url: str) -> FastAPI:
    """Create FastAPI app with A2A protocol endpoints for green evaluator."""
    app = FastAPI(title="BrowserGym Green Evaluator", version="1.0.0")
    tasks = {}

    @app.get("/.well-known/agent-card.json")
    async def get_agent_card():
        return {
            "name": "BrowserGym Green Evaluator",
            "description": "Evaluates web automation agents on BrowserGym benchmarks",
            "url": card_url,
            "role": "evaluator",
            "skills": [{
                "name": "agent_evaluation",
                "description": "Evaluate agent performance on web tasks"
            }],
            "benchmarks": ["miniwob", "workarena", "webarena", "visualwebarena"]
        }

    @app.get("/")
    async def root():
        return {"name": "BrowserGym Green Evaluator", "version": "1.0.0", "role": "green_agent", "status": "ready"}

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/assessment/request")
    async def handle_assessment(request: Request):
        try:
            body = await request.json()
            task_id = str(uuid.uuid4())
            
            tasks[task_id] = {
                "id": task_id,
                "status": "received",
                "participants": body.get("participants", {}),
                "config": body.get("config", {}),
                "results": None
            }
            
            print(f"Assessment {task_id} received")
            asyncio.create_task(run_assessment(task_id, tasks))
            
            return {"task_id": task_id, "status": "started", "message": "Assessment started"}
        except Exception as e:
            print(f"Error handling assessment: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/assessment/{task_id}")
    async def get_assessment_status(task_id: str):
        if task_id not in tasks:
            return JSONResponse(status_code=404, content={"error": "Task not found"})
        
        task = tasks[task_id]
        return {"task_id": task_id, "status": task["status"], "results": task.get("results")}

    @app.post("/message/send")
    async def handle_message(request: Request):
        try:
            body = await request.json()
            print(f"Received A2A message: {body}")
            return {"status": "received", "message": "Message processed"}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return app


async def run_assessment(task_id: str, tasks: dict):
    """Run assessment using existing BrowserGym functionality."""
    try:
        task = tasks[task_id]
        task["status"] = "running"
        
        config = task["config"]
        task_name = config.get("task_name", "miniwob.click-test")
        max_steps = config.get("max_steps", 50)
        model_name = config.get("model_name", "gpt-4o-mini")
        
        print(f"Running assessment {task_id} on {task_name}")
        
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
        
        task["status"] = "completed"
        task["results"] = results
        
        print(f"Assessment {task_id} completed: {results}")
        
    except Exception as e:
        print(f"Assessment {task_id} failed: {e}")
        task["status"] = "failed"
        task["results"] = {"error": str(e), "task_name": config.get("task_name", "unknown")}


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    args = parse_args()
    
    if args.a2a_server:
        card_url = args.card_url or f"http://localhost:{args.port}"
        app = create_a2a_app(card_url)
        
        print(f"Starting BrowserGym Green Evaluator")
        print(f"Server: {card_url}")
        print(f"Agent Card: {card_url}/.well-known/agent-card.json")
        print(f"Assessment endpoint: {card_url}/assessment/request")
        
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        main()
