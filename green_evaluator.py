#!/usr/bin/env python3
"""
Green Evaluator Agent - A Benchmark Testing Framework
====================================================

This is a bare bones green evaluator agent that acts as a wrapper for testing
other web agents (white/demo agents) on BrowserGym benchmarks, including MiniWoB, WorkArena, and WebArena.

The green evaluator:
1. Takes other agents as input
2. Runs them through BrowserGym benchmark tasks (MiniWoB, WorkArena, WebArena, etc.)
3. Evaluates their performance
4. Generates evaluation reports

Usage (Standalone):
    python3 green_evaluator.py --agent_path demo_agent/agent.py --task miniwob.click-dialog
    python3 green_evaluator.py --agent_path demo_agent/agent.py --task workarena.servicenow.order-standard-laptop
    python3 green_evaluator.py --agent_path demo_agent/agent.py --task webarena.4

Usage (A2A Server for AgentBeats):
    python3 green_evaluator.py --a2a-server --host 0.0.0.0 --port 8000 --card-url http://your-public-url:8000
"""

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    FastAPI = None
    HTTPException = None
    JSONResponse = None
    BaseModel = None
    uvicorn = None

# BrowserGym imports
from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result
from browsergym.experiments.agent import Agent


class GreenEvaluator:
    """
    Green Evaluator Agent - The Benchmark Testing Framework
    
    This class implements the evaluation logic for testing other web agents.
    It acts as a wrapper around BrowserGym's experiment framework to provide
    a clean interface for evaluating agent performance.
    """
    
    def __init__(self, results_dir: str = "./green_evaluation_results"):
        """
        Initialize the Green Evaluator.
        
        Args:
            results_dir: Directory to save evaluation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Load required environment variables
        self._load_environment_variables()
        
        # Evaluation metrics we'll track
        self.evaluation_metrics = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_steps": 0,
            "average_reward": 0,
            "task_results": []
        }
        
        self.logger.info("Green Evaluator initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the evaluator."""
        logger = logging.getLogger("GreenEvaluator")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = self.results_dir / "green_evaluator.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_environment_variables(self):
        """
        Load required environment variables for BrowserGym benchmarks.
        
        This method automatically loads the .env file if it exists, which contains
        the MINIWOB_URL and other required environment variables.
        """
        # Look for .env file in the current directory and parent directories
        current_dir = Path.cwd()
        env_file = None
        
        # Check current directory and parent directories
        for directory in [current_dir, current_dir.parent]:
            potential_env = directory / ".env"
            if potential_env.exists():
                env_file = potential_env
                break
        
        if env_file:
            self.logger.info(f"Loading environment variables from: {env_file}")
            try:
                # Load environment variables from .env file
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            # Remove quotes if present
                            value = value.strip('"').strip("'")
                            os.environ[key] = value
                            self.logger.debug(f"Set {key}={value}")
                
                # Verify critical environment variables
                if os.getenv('MINIWOB_URL'):
                    self.logger.info(f"MINIWOB_URL loaded: {os.getenv('MINIWOB_URL')}")
                else:
                    self.logger.warning("MINIWOB_URL not found in .env file")
                
                if os.getenv('OPENAI_API_KEY'):
                    self.logger.info("OPENAI_API_KEY loaded")
                else:
                    self.logger.warning("OPENAI_API_KEY not found in .env file")

                if os.getenv('HUGGING_FACE_HUB_TOKEN'):
                    self.logger.info("HUGGING_FACE_HUB_TOKEN loaded")
                else:
                    self.logger.warning("HUGGING_FACE_HUB_TOKEN not found in .env file (Required for WorkArena)")
                    
            except Exception as e:
                self.logger.error(f"Failed to load .env file: {e}")
        else:
            self.logger.warning("No .env file found. Make sure to set MINIWOB_URL and OPENAI_API_KEY manually")
    
    def load_agent(self, agent_path: str) -> Agent:
        """
        Load an agent from a Python file.
        
        Args:
            agent_path: Path to the agent Python file
            
        Returns:
            Loaded agent instance
        """
        self.logger.info(f"Loading agent from: {agent_path}")
        
        try:
            # Load the agent module
            spec = importlib.util.spec_from_file_location("agent_module", agent_path)
            agent_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(agent_module)
            
            # Look for agent classes in the module
            agent_classes = []
            for name, obj in agent_module.__dict__.items():
                if (isinstance(obj, type) and 
                    issubclass(obj, Agent) and 
                    obj != Agent):
                    agent_classes.append(obj)
            
            if not agent_classes:
                raise ValueError(f"No agent classes found in {agent_path}")
            
            # Use the first agent class found (assuming it's the main one)
            agent_class = agent_classes[0]
            self.logger.info(f"Found agent class: {agent_class.__name__}")
            
            # Try to create agent instance with default parameters
            # This is a simplified approach - in practice, you'd need to handle
            # different agent constructors more carefully
            try:
                # For DemoAgent, try with default parameters
                if hasattr(agent_class, '__init__'):
                    # Get default parameters from the class
                    import inspect
                    sig = inspect.signature(agent_class.__init__)
                    params = {}
                    
                    # Set reasonable defaults for common parameters
                    defaults = {
                        'model_name': 'gpt-4o-mini',
                        'chat_mode': False,  # Benchmark mode
                        'demo_mode': 'off',
                        'use_html': False,
                        'use_axtree': True,
                        'use_screenshot': False
                    }
                    
                    for param_name, param in sig.parameters.items():
                        if param_name != 'self' and param_name in defaults:
                            params[param_name] = defaults[param_name]
                    
                    agent = agent_class(**params)
                    self.logger.info(f"Created agent instance with parameters: {params}")
                    return agent
                else:
                    agent = agent_class()
                    self.logger.info("Created agent instance with no parameters")
                    return agent
                    
            except Exception as e:
                self.logger.error(f"Failed to create agent instance: {e}")
                raise
                
        except Exception as e:
            self.logger.error(f"Failed to load agent from {agent_path}: {e}")
            raise
    
    def evaluate_agent_on_task(self, agent: Agent, task_name: str, max_steps: int = 50) -> Dict[str, Any]:
        """
        Evaluate a single agent on a single benchmark task.
        
        Args:
            agent: The agent to evaluate
            task_name: Name of the benchmark task (e.g., 'miniwob.click-dialog')
            max_steps: Maximum number of steps for the task
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info(f"Evaluating agent on task: {task_name}")
        
        # Set up environment arguments for the benchmark task
        env_args = EnvArgs(
            task_name=task_name,
            task_seed=None,  # Random seed
            max_steps=max_steps,
            headless=True,  # Run in headless mode for evaluation
            record_video=False,
            wait_for_user_message=False,  # No human interaction
        )
        
        # Set up experiment arguments
        exp_args = ExpArgs(
            env_args=env_args,
            agent_args=None,  # We'll handle the agent directly
        )
        
        # Create a unique experiment directory
        timestamp = int(time.time())
        exp_dir = self.results_dir / f"eval_{task_name}_{timestamp}"
        exp_args.exp_dir = str(exp_dir)
        exp_args.exp_name = f"GreenEval_{task_name}_{timestamp}"
        
        try:
            # Run the evaluation
            self.logger.info(f"Starting evaluation in: {exp_dir}")
            
            # Import and prepare backend if needed (WorkArena, WebArena require backend preparation)
            if task_name.startswith("workarena"):
                try:
                    # Import WorkArena to register the environments
                    import browsergym.workarena
                    from browsergym.experiments.benchmark.utils import prepare_backend
                    self.logger.info("Preparing WorkArena backend...")
                    prepare_backend("workarena")
                    self.logger.info("WorkArena backend ready")
                except Exception as e:
                    self.logger.warning(f"Backend preparation warning (may continue anyway): {e}")
            elif task_name.startswith("webarena"):
                try:
                    # Import WebArena to register the environments
                    import browsergym.webarena
                    from browsergym.experiments.benchmark.utils import prepare_backend
                    self.logger.info("Preparing WebArena backend...")
                    prepare_backend("webarena")
                    self.logger.info("WebArena backend ready")
                except Exception as e:
                    self.logger.warning(f"Backend preparation warning (may continue anyway): {e}")
            
            # Prepare the experiment
            exp_args.prepare(str(self.results_dir))
            
            # Create environment
            env = env_args.make_env(
                action_mapping=agent.action_set.to_python_code,
                exp_dir=exp_dir,
            )
            
            # Run the agent through the task
            step_count = 0
            total_reward = 0
            success = False
            
            # Reset environment
            obs, env_info = env.reset()
            obs = agent.obs_preprocessor(obs)
            
            # Run the agent
            while step_count < max_steps:
                try:
                    # Get agent's action
                    action, agent_info = agent.get_action(obs.copy())
                    
                    if action is None:
                        self.logger.info("Agent returned None action, ending evaluation")
                        break
                    
                    # Execute action in environment
                    obs, reward, terminated, truncated, env_info = env.step(action)
                    obs = agent.obs_preprocessor(obs)
                    
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
                        
                except Exception as e:
                    self.logger.error(f"Error during evaluation: {e}")
                    break
            
            # Close environment
            env.close()
            
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
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            self.logger.error(f"Evaluation failed: {e}")
            self.logger.error(f"Full traceback:\n{error_traceback}")
            return {
                "task_name": task_name,
                "success": False,
                "steps_taken": 0,
                "total_reward": 0,
                "max_steps": max_steps,
                "error": str(e),
                "error_traceback": error_traceback,
                "exp_dir": str(exp_dir),
                "timestamp": timestamp
            }
    
    def evaluate_agent_on_benchmark_suite(self, agent: Agent, task_list: List[str]) -> Dict[str, Any]:
        """
        Evaluate an agent on multiple benchmark tasks.
        
        Args:
            agent: The agent to evaluate
            task_list: List of task names to evaluate on
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        self.logger.info(f"Starting benchmark suite evaluation with {len(task_list)} tasks")
        
        results = []
        successful_tasks = 0
        total_steps = 0
        total_reward = 0
        
        for i, task_name in enumerate(task_list, 1):
            self.logger.info(f"Task {i}/{len(task_list)}: {task_name}")
            
            # Evaluate on this task
            result = self.evaluate_agent_on_task(agent, task_name)
            results.append(result)
            
            # Update statistics
            if result["success"]:
                successful_tasks += 1
            total_steps += result["steps_taken"]
            total_reward += result["total_reward"]
            
            # Small delay between tasks
            time.sleep(1)
        
        # Calculate final metrics
        final_results = {
            "agent_evaluated": str(agent.__class__.__name__),
            "total_tasks": len(task_list),
            "successful_tasks": successful_tasks,
            "failed_tasks": len(task_list) - successful_tasks,
            "success_rate": successful_tasks / len(task_list) if task_list else 0,
            "average_steps": total_steps / len(task_list) if task_list else 0,
            "average_reward": total_reward / len(task_list) if task_list else 0,
            "task_results": results,
            "evaluation_timestamp": int(time.time())
        }
        
        # Save results
        results_file = self.results_dir / f"benchmark_evaluation_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        self.logger.info(f"Benchmark evaluation complete!")
        self.logger.info(f"Success rate: {final_results['success_rate']:.2%}")
        self.logger.info(f"Average steps: {final_results['average_steps']:.1f}")
        self.logger.info(f"Average reward: {final_results['average_reward']:.2f}")
        self.logger.info(f"Results saved to: {results_file}")
        
        return final_results


def get_miniwob_task_list() -> List[str]:
    """
    Get a list of MiniWoB benchmark tasks for evaluation.
    
    Returns:
        List of MiniWoB task names
    """
    # These are some common MiniWoB tasks for evaluation
    # In practice, you might want to load this from a configuration file
    # or query the available tasks dynamically
    return [
        "miniwob.click-dialog",
        "miniwob.choose-list", 
        "miniwob.click-checkboxes",
        "miniwob.choose-date",
        "miniwob.choose-date-easy",
        "miniwob.ascending-numbers",
        "miniwob.bisect-angle",
        "miniwob.book-flight",
        "miniwob.buy-ticket"
    ]


def get_workarena_task_list(level: str = "l1", max_tasks: Optional[int] = None) -> List[str]:
    """
    Get a list of WorkArena benchmark tasks for evaluation.
    
    Args:
        level: WorkArena level to use ("l1", "l2", or "l3"). Defaults to "l1".
        max_tasks: Maximum number of tasks to return. If None, returns all tasks for the level.
    
    Returns:
        List of WorkArena task names
    """
    # Try to load from the metadata CSV file
    try:
        import pandas as pd
        # Try multiple possible paths
        possible_paths = [
            Path(__file__).parent / "browsergym" / "experiments" / "src" / "browsergym" / "experiments" / "benchmark" / "metadata" / "workarena.csv",
            Path(__file__).parent.parent / "browsergym" / "experiments" / "src" / "browsergym" / "experiments" / "benchmark" / "metadata" / "workarena.csv",
        ]
        
        metadata_path = None
        for path in possible_paths:
            if path.exists():
                metadata_path = path
                break
        
        if metadata_path:
            df = pd.read_csv(metadata_path)
            # Filter by level
            level_tasks = df[df["level"] == level]["task_name"].tolist()
            
            if max_tasks:
                return level_tasks[:max_tasks]
            return level_tasks
    except Exception as e:
        # Fallback to hardcoded list if CSV loading fails
        pass
    
    # Fallback: return a curated list of common L1 WorkArena tasks
    l1_tasks = [
        "workarena.servicenow.order-standard-laptop",
        "workarena.servicenow.order-ipad-pro",
        "workarena.servicenow.order-developer-laptop",
        "workarena.servicenow.create-incident",
        "workarena.servicenow.create-change-request",
        "workarena.servicenow.filter-asset-list",
        "workarena.servicenow.sort-asset-list",
        "workarena.servicenow.single-chart-value-retrieval",
    ]
    
    if level == "l1":
        return l1_tasks[:max_tasks] if max_tasks else l1_tasks
    else:
        # For l2/l3, return empty list in fallback mode (should use CSV)
        return []


def get_webarena_task_list(max_tasks: Optional[int] = None) -> List[str]:
    """
    Get a list of WebArena benchmark tasks for evaluation.
    
    Args:
        max_tasks: Maximum number of tasks to return. If None, returns a curated list.
    
    Returns:
        List of WebArena task names
    """
    # Try to load from the metadata CSV file
    try:
        import pandas as pd
        # Try multiple possible paths
        possible_paths = [
            Path(__file__).parent / "browsergym" / "experiments" / "src" / "browsergym" / "experiments" / "benchmark" / "metadata" / "webarena.csv",
            Path(__file__).parent.parent / "browsergym" / "experiments" / "src" / "browsergym" / "experiments" / "benchmark" / "metadata" / "webarena.csv",
        ]
        
        metadata_path = None
        for path in possible_paths:
            if path.exists():
                metadata_path = path
                break
        
        if metadata_path:
            df = pd.read_csv(metadata_path)
            # Get all task names
            all_tasks = df["task_name"].tolist()
            
            if max_tasks:
                return all_tasks[:max_tasks]
            return all_tasks
    except Exception as e:
        # Fallback to hardcoded list if CSV loading fails
        pass
    
    # Fallback: return a curated list of common WebArena tasks
    # These are some representative tasks from different sites
    webarena_tasks = [
        "webarena.4",      # shopping_admin
        "webarena.7",      # map
        "webarena.21",     # shopping
        "webarena.27",     # reddit
        "webarena.410",    # reddit (commonly used)
        "webarena.533",    # gitlab (commonly used)
        "webarena.561",    # gitlab wiki
        "webarena.562",    # gitlab reddit
        "webarena.574",    # shopping
        "webarena.640",    # reddit
        "webarena.680",    # shopping_admin
        "webarena.740",    # wiki map
    ]
    
    return webarena_tasks[:max_tasks] if max_tasks else webarena_tasks


def get_task_list_by_benchmark(benchmark: str) -> List[str]:
    """
    Get a task list for a specific benchmark.
    
    Args:
        benchmark: Benchmark name ("miniwob", "workarena", or "webarena")
    
    Returns:
        List of task names for the benchmark
    """
    if benchmark.lower() == "miniwob":
        return get_miniwob_task_list()
    elif benchmark.lower() == "workarena":
        return get_workarena_task_list()
    elif benchmark.lower() == "webarena":
        return get_webarena_task_list()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}. Supported: 'miniwob', 'workarena', 'webarena'")


# ============================================================================
# A2A Server Implementation for AgentBeats
# ============================================================================

if A2A_AVAILABLE:
    # A2A Protocol Models
    class A2AMessage(BaseModel):
        """Base A2A message structure"""
        type: str
        task_id: Optional[str] = None
        content: Optional[Dict[str, Any]] = None

    class AssessmentRequest(BaseModel):
        """Assessment request message from AgentBeats"""
        type: str = "assessment_request"
        task_id: str
        participants: Dict[str, str]  # role -> endpoint mapping
        config: Dict[str, Any]  # assessment configuration

    class TaskUpdate(BaseModel):
        """Task update message for A2A protocol"""
        type: str = "task_update"
        task_id: str
        content: Dict[str, Any]

    class Artifact(BaseModel):
        """A2A artifact for results"""
        type: str = "artifact"
        task_id: str
        name: str
        content: Dict[str, Any]
        mime_type: str = "application/json"

    class A2AServer:
        """A2A Server wrapper for Green Evaluator"""
        
        def __init__(self, evaluator: GreenEvaluator, card_url: Optional[str] = None):
            self.evaluator = evaluator
            self.app = FastAPI(title="BrowserGym Green Evaluator", version="1.0.0")
            self.card_url = card_url
            self.active_tasks: Dict[str, Dict[str, Any]] = {}
            
            # Setup routes
            self._setup_routes()
        
        def _setup_routes(self):
            """Setup A2A protocol routes"""
            
            @self.app.get("/")
            async def root():
                return {
                    "name": "BrowserGym Green Evaluator",
                    "description": "A green agent that evaluates web agents on BrowserGym benchmarks",
                    "version": "1.0.0",
                    "a2a_protocol": True
                }
            
            @self.app.get("/card")
            async def get_card():
                """Return agent card information"""
                return {
                    "name": "BrowserGym Green Evaluator",
                    "description": "Evaluates web agents on BrowserGym benchmarks (MiniWoB, WorkArena, WebArena, etc.)",
                    "url": self.card_url or "http://localhost:8000",
                    "capabilities": [
                        "miniwob_benchmark",
                        "workarena_benchmark",
                        "webarena_benchmark",
                        "agent_evaluation"
                    ]
                }
            
            @self.app.post("/a2a/message")
            async def handle_a2a_message(request: Request):
                """Handle incoming A2A messages"""
                try:
                    body = await request.json()
                    msg_type = body.get("type")
                    
                    if msg_type == "assessment_request":
                        return await self._handle_assessment_request(body)
                    elif msg_type == "task_message":
                        return await self._handle_task_message(body)
                    else:
                        return JSONResponse(
                            status_code=400,
                            content={"error": f"Unknown message type: {msg_type}"}
                        )
                except Exception as e:
                    self.evaluator.logger.error(f"Error handling A2A message: {e}")
                    return JSONResponse(
                        status_code=500,
                        content={"error": str(e)}
                    )
            
            @self.app.get("/status")
            async def status():
                """Health check endpoint"""
                return {"status": "healthy"}

            @self.app.get("/health")
            async def health():
                """Health check endpoint"""
                return {"status": "healthy"}
        
        async def _handle_assessment_request(self, request_body: Dict[str, Any]) -> JSONResponse:
            """Handle assessment_request message"""
            try:
                task_id = request_body.get("task_id", str(uuid.uuid4()))
                participants = request_body.get("participants", {})
                config = request_body.get("config", {})
                
                self.evaluator.logger.info(f"Received assessment_request: task_id={task_id}")
                self.evaluator.logger.info(f"Participants: {participants}")
                self.evaluator.logger.info(f"Config: {config}")
                
                # Store task info
                self.active_tasks[task_id] = {
                    "participants": participants,
                    "config": config,
                    "status": "running"
                }
                
                # Start assessment asynchronously
                asyncio.create_task(self._run_assessment(task_id, participants, config))
                
                # Return acknowledgment
                return JSONResponse(content={
                    "type": "assessment_ack",
                    "task_id": task_id,
                    "status": "accepted"
                })
                
            except Exception as e:
                self.evaluator.logger.error(f"Error handling assessment_request: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )
        
        async def _handle_task_message(self, request_body: Dict[str, Any]) -> JSONResponse:
            """Handle task messages from participants"""
            task_id = request_body.get("task_id")
            # For now, just acknowledge
            return JSONResponse(content={
                "type": "task_message_ack",
                "task_id": task_id
            })
        
        async def _run_assessment(self, task_id: str, participants: Dict[str, str], config: Dict[str, Any]):
            """Run the actual assessment"""
            try:
                # Emit task update: starting
                await self._emit_task_update(task_id, {
                    "status": "starting",
                    "message": "Starting BrowserGym evaluation"
                })
                
                # Get task list from config or use default
                # Support both explicit task lists and benchmark names
                tasks_config = config.get("tasks")
                if tasks_config is None:
                    # Default to MiniWoB if no tasks specified
                    task_list = get_miniwob_task_list()
                elif isinstance(tasks_config, str):
                    # Check if it's a benchmark name or a single task
                    if tasks_config.lower() in ["miniwob", "workarena", "webarena"]:
                        task_list = get_task_list_by_benchmark(tasks_config)
                    else:
                        # Single task name
                        task_list = [tasks_config]
                elif isinstance(tasks_config, list):
                    task_list = tasks_config
                else:
                    raise ValueError(f"Invalid tasks config: {tasks_config}")
                
                # Also support benchmark name in config
                benchmark_name = config.get("benchmark")
                if benchmark_name and not tasks_config:
                    task_list = get_task_list_by_benchmark(benchmark_name)
                
                # Determine which backends need to be prepared based on task list
                backends_to_prepare = set()
                for task in task_list:
                    if isinstance(task, str):
                        if task.startswith("miniwob"):
                            backends_to_prepare.add("miniwob")
                        elif task.startswith("workarena"):
                            backends_to_prepare.add("workarena")
                        elif task.startswith("webarena"):
                            backends_to_prepare.add("webarena")
                
                # Prepare backends
                if backends_to_prepare:
                    await self._emit_task_update(task_id, {
                        "status": "preparing",
                        "message": f"Preparing backends: {', '.join(backends_to_prepare)}"
                    })
                    self._prepare_backends(backends_to_prepare)
                
                # For now, we'll evaluate using local agent loading
                # In a full implementation, you'd interact with purple agents via A2A
                agent_url = participants.get("purple_agent") or participants.get("agent")
                
                if not agent_url:
                    # Fallback: try to load from config
                    agent_path = config.get("agent_path")
                    if agent_path:
                        await self._evaluate_local_agent(task_id, agent_path, task_list, config)
                    else:
                        raise ValueError("No agent URL or path provided")
                else:
                    # TODO: Implement A2A client to interact with purple agent
                    # For now, emit an error
                    await self._emit_task_update(task_id, {
                        "status": "error",
                        "message": "A2A client integration not yet implemented. Please provide agent_path in config."
                    })
                    self.active_tasks[task_id]["status"] = "error"
                    
            except Exception as e:
                self.evaluator.logger.error(f"Assessment failed: {e}")
                await self._emit_task_update(task_id, {
                    "status": "error",
                    "message": f"Assessment failed: {str(e)}"
                })
                self.active_tasks[task_id]["status"] = "error"
        
        async def _evaluate_local_agent(self, task_id: str, agent_path: str, task_list: List[str], config: Dict[str, Any]):
            """Evaluate a local agent (fallback mode)"""
            try:
                max_steps = config.get("max_steps", 50)
                
                # Load agent
                await self._emit_task_update(task_id, {
                    "status": "loading_agent",
                    "message": f"Loading agent from {agent_path}"
                })
                
                agent = self.evaluator.load_agent(agent_path)
                
                # Run evaluation
                if len(task_list) == 1:
                    await self._emit_task_update(task_id, {
                        "status": "evaluating",
                        "message": f"Evaluating on task: {task_list[0]}"
                    })
                    result = self.evaluator.evaluate_agent_on_task(agent, task_list[0], max_steps)
                    results = {
                        "agent_evaluated": str(agent.__class__.__name__),
                        "total_tasks": 1,
                        "successful_tasks": 1 if result["success"] else 0,
                        "failed_tasks": 0 if result["success"] else 1,
                        "success_rate": 1.0 if result["success"] else 0.0,
                        "average_steps": result["steps_taken"],
                        "average_reward": result["total_reward"],
                        "task_results": [result]
                    }
                else:
                    await self._emit_task_update(task_id, {
                        "status": "evaluating",
                        "message": f"Evaluating on {len(task_list)} tasks"
                    })
                    results = self.evaluator.evaluate_agent_on_benchmark_suite(agent, task_list)
                
                # Create artifact with results
                artifact = {
                    "type": "artifact",
                    "task_id": task_id,
                    "name": "evaluation_results",
                    "mime_type": "application/json",
                    "content": results
                }
                
                await self._emit_task_update(task_id, {
                    "status": "complete",
                    "message": "Evaluation complete",
                    "artifact": artifact
                })
                
                self.active_tasks[task_id]["status"] = "complete"
                self.active_tasks[task_id]["results"] = results
                
            except Exception as e:
                self.evaluator.logger.error(f"Local agent evaluation failed: {e}")
                await self._emit_task_update(task_id, {
                    "status": "error",
                    "message": f"Evaluation failed: {str(e)}"
                })
                raise
        
        def _prepare_backends(self, backends: set):
            """Prepare benchmark backends"""
            try:
                from browsergym.experiments.benchmark.utils import prepare_backend
                
                for backend in backends:
                    try:
                        self.evaluator.logger.info(f"Preparing {backend} backend...")
                        prepare_backend(backend)
                        self.evaluator.logger.info(f"{backend} backend ready")
                    except Exception as e:
                        self.evaluator.logger.warning(f"Failed to prepare {backend} backend: {e}")
                        # Continue with other backends
            except ImportError:
                self.evaluator.logger.warning("Could not import prepare_backend, skipping backend preparation")
        
        async def _emit_task_update(self, task_id: str, content: Dict[str, Any]):
            """Emit a task update (in a real implementation, this would be sent via A2A)"""
            self.evaluator.logger.info(f"Task {task_id} update: {content}")
            # In a full implementation, you'd send this to the A2A task update endpoint
            # For now, we just log it


def create_agent_card(card_path: str, card_url: str):
    """Create an agent card TOML file"""
    card_data = {
        "name": "BrowserGym Green Evaluator",
        "description": "A green agent that evaluates web agents on BrowserGym benchmarks including MiniWoB, WorkArena, and WebArena",
        "url": card_url,
        "capabilities": [
            "miniwob_benchmark",
            "workarena_benchmark",
            "webarena_benchmark",
            "agent_evaluation"
        ]
    }
    
    # Write as JSON (always works)
    card_json_path = card_path.replace('.toml', '.json')
    with open(card_json_path, 'w') as f:
        json.dump(card_data, f, indent=2)
    print(f"Agent card created as JSON: {card_json_path}")
    
    # Also try to write TOML if tomli_w is available
    try:
        import tomli_w
        with open(card_path, 'wb') as f:
            tomli_w.dump(card_data, f)
        print(f"Agent card created as TOML: {card_path}")
    except ImportError:
        print(f"Note: TOML support not available (install tomli-w for TOML support). Using JSON instead.")


def main():
    """Main function to run the Green Evaluator."""
    parser = argparse.ArgumentParser(
        description="Green Evaluator - Test web agents on BrowserGym benchmarks"
    )
    
    # A2A Server mode
    parser.add_argument(
        "--a2a-server",
        action="store_true",
        help="Run as A2A server for AgentBeats (instead of standalone mode)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind A2A server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind A2A server to (default: 8000)"
    )
    parser.add_argument(
        "--card-url",
        type=str,
        default=None,
        help="Public URL for agent card (e.g., https://your-tunnel-url:8000)"
    )
    parser.add_argument(
        "--card-path",
        type=str,
        default="green_evaluator_card.toml",
        help="Path to save agent card file (default: green_evaluator_card.toml)"
    )
    
    # Standalone mode arguments
    parser.add_argument(
        "--agent_path", 
        type=str, 
        default=None,
        help="Path to the agent Python file to evaluate (standalone mode)"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default=None,
        help="Single task to evaluate (e.g., 'miniwob.click-dialog'). If not provided, runs full benchmark suite."
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=50,
        help="Maximum number of steps per task"
    )
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="./green_evaluation_results",
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    # A2A Server mode
    if args.a2a_server:
        if not A2A_AVAILABLE:
            print("ERROR: A2A dependencies not available. Install with: pip install fastapi uvicorn")
            sys.exit(1)
        
        # Initialize the Green Evaluator
        evaluator = GreenEvaluator(results_dir=args.results_dir)
        
        # Determine card URL
        card_url = args.card_url
        if not card_url:
            card_url = f"http://{args.host}:{args.port}"
            if args.host == "0.0.0.0":
                card_url = f"http://localhost:{args.port}"
        
        # Create agent card
        create_agent_card(args.card_path, card_url)
        
        # Create and run A2A server
        server = A2AServer(evaluator, card_url=card_url)
        
        evaluator.logger.info(f"Starting A2A server on {args.host}:{args.port}")
        evaluator.logger.info(f"Agent card URL: {card_url}")
        evaluator.logger.info(f"Agent card saved to: {args.card_path}")
        
        uvicorn.run(server.app, host=args.host, port=args.port, log_level="debug")
        return
    
    # Standalone mode (original functionality)
    if not args.agent_path:
        parser.error("--agent_path is required in standalone mode (or use --a2a-server)")
    
    # Initialize the Green Evaluator
    evaluator = GreenEvaluator(results_dir=args.results_dir)
    
    # Determine which task(s) will be evaluated
    task_to_check = args.task
    if not task_to_check:
        # Default to MiniWoB if no task specified
        task_to_check = "miniwob.click-dialog"  # Just for env var checking
    
    # Check required environment variables based on task type
    if task_to_check.startswith("miniwob"):
        if not os.getenv('MINIWOB_URL'):
            evaluator.logger.error("MINIWOB_URL environment variable not set!")
            evaluator.logger.error("Please run: source .env")
            sys.exit(1)
    elif task_to_check.startswith("workarena"):
        required_vars = ['SNOW_INSTANCE_URL', 'SNOW_INSTANCE_UNAME', 'SNOW_INSTANCE_PWD']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            evaluator.logger.error(f"Missing required environment variables for WorkArena: {', '.join(missing_vars)}")
            evaluator.logger.error("Please set SNOW_INSTANCE_URL, SNOW_INSTANCE_UNAME, and SNOW_INSTANCE_PWD")
            sys.exit(1)
    elif task_to_check.startswith("webarena"):
        required_vars = ['WA_SHOPPING', 'WA_SHOPPING_ADMIN', 'WA_REDDIT', 'WA_GITLAB', 'WA_WIKIPEDIA', 'WA_MAP', 'WA_HOMEPAGE']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            evaluator.logger.error(f"Missing required environment variables for WebArena: {', '.join(missing_vars)}")
            evaluator.logger.error("Please set WA_SHOPPING, WA_SHOPPING_ADMIN, WA_REDDIT, WA_GITLAB, WA_WIKIPEDIA, WA_MAP, and WA_HOMEPAGE")
            evaluator.logger.error("These should point to your GCP VM instance where WebArena Docker containers are running")
            sys.exit(1)
    
    if not os.getenv('OPENAI_API_KEY'):
        evaluator.logger.error("OPENAI_API_KEY environment variable not set!")
        evaluator.logger.error("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    try:
        # Load the agent to evaluate
        agent = evaluator.load_agent(args.agent_path)
        
        if args.task:
            # Evaluate on single task
            evaluator.logger.info(f"Single task evaluation: {args.task}")
            result = evaluator.evaluate_agent_on_task(agent, args.task, args.max_steps)
            print(f"\nSingle Task Results:")
            print(f"Task: {result['task_name']}")
            print(f"Success: {result['success']}")
            print(f"Steps: {result['steps_taken']}")
            print(f"Reward: {result['total_reward']}")
            
        else:
            # Evaluate on full benchmark suite (default to MiniWoB)
            evaluator.logger.info("Full benchmark suite evaluation (defaulting to MiniWoB)")
            evaluator.logger.info("Use --task to specify a specific task or modify code to use different benchmark")
            task_list = get_miniwob_task_list()
            results = evaluator.evaluate_agent_on_benchmark_suite(agent, task_list)
            
            print(f"\nBenchmark Suite Results:")
            print(f"Agent: {results['agent_evaluated']}")
            print(f"Success Rate: {results['success_rate']:.2%}")
            print(f"Average Steps: {results['average_steps']:.1f}")
            print(f"Average Reward: {results['average_reward']:.2f}")
            print(f"Successful Tasks: {results['successful_tasks']}/{results['total_tasks']}")
    
    except Exception as e:
        evaluator.logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()