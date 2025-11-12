#!/usr/bin/env python3
"""
Green Evaluator Agent

Evaluator agent that acts as a wrapper for testing
other web agents (white/demo agents) on BrowserGym benchmarks, specifically MiniWoB.

The green evaluator:
1. Takes other agents as input
2. Runs them through MiniWoB benchmark tasks
3. Evaluates their performance
4. Generates evaluation reports

Supports two modes:
- Direct Import Mode: Loads agent Python files directly (for local testing)
- A2A Mode: Communicates with agents via HTTP API (for AgentBeats platform)

Usage (Direct Import):
    python3 green_evaluator.py --agent_path demo_agent/agent.py --task miniwob.click-dialog

Usage (A2A Mode):
    python3 green_evaluator.py --agent_url http://localhost:5002 --task miniwob.click-dialog

Usage (Run as A2A Server):
    python3 green_evaluator.py --server_mode --port 5001
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod

# BrowserGym imports
from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result
from browsergym.experiments.agent import Agent, DEFAULT_ACTION_SET
from browsergym.core.action.base import AbstractActionSet

# A2A imports
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not available. A2A server mode will not work.")

from green_evaluator_a2a_client import A2AAgentClient
import requests


class AgentAdapter(ABC):
    """
    Adapter interface to make both direct Agent instances and A2A clients
    work seamlessly with the evaluator.
    """
    
    @abstractmethod
    def get_action(self, obs: Any) -> tuple[str, Dict[str, Any]]:
        """Get action from agent given observation."""
        pass
    
    @abstractmethod
    def obs_preprocessor(self, obs: dict) -> Any:
        """Preprocess observation before sending to agent."""
        pass
    
    @property
    @abstractmethod
    def action_set(self) -> AbstractActionSet:
        """Get the agent's action set."""
        pass


class DirectAgentAdapter(AgentAdapter):
    """Adapter for directly imported Agent instances."""
    
    def __init__(self, agent: Agent):
        self.agent = agent
    
    def get_action(self, obs: Any) -> tuple[str, Dict[str, Any]]:
        return self.agent.get_action(obs)
    
    def obs_preprocessor(self, obs: dict) -> Any:
        return self.agent.obs_preprocessor(obs)
    
    @property
    def action_set(self) -> AbstractActionSet:
        return self.agent.action_set


class A2AAgentAdapter(AgentAdapter):
    """Adapter for A2A client connections."""
    
    def __init__(self, client: A2AAgentClient):
        self.client = client
        # We'll use default action set if we can't get it from agent
        self._action_set = DEFAULT_ACTION_SET
        # Try to get action set info
        self._try_get_action_set_info()
    
    def _try_get_action_set_info(self):
        """Try to get action set information from the agent."""
        try:
            action_set_info = self.client.get_action_set_info()
            if action_set_info:
                # For now, we'll use default action set
                # In a full implementation, you'd reconstruct the action set from info
                pass
        except Exception as e:
            logging.getLogger("A2AAgentAdapter").warning(
                f"Could not get action set info: {e}"
            )
    
    def get_action(self, obs: Any) -> tuple[str, Dict[str, Any]]:
        # A2A clients expect the preprocessed observation
        return self.client.get_action(obs)
    
    def obs_preprocessor(self, obs: dict) -> Any:
        # For A2A, we'll do basic preprocessing here
        # In practice, you might want to delegate to the agent's preprocessor via API
        # For now, we'll do minimal preprocessing
        return obs
    
    @property
    def action_set(self) -> AbstractActionSet:
        return self._action_set


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
                    
            except Exception as e:
                self.logger.error(f"Failed to load .env file: {e}")
        else:
            self.logger.warning("No .env file found. Make sure to set MINIWOB_URL and OPENAI_API_KEY manually")
    
    def load_agent(self, agent_path: str) -> AgentAdapter:
        """
        Load an agent from a Python file (Direct Import Mode).
        
        Args:
            agent_path: Path to the agent Python file
            
        Returns:
            AgentAdapter wrapping the loaded agent
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
                    return DirectAgentAdapter(agent)
                else:
                    agent = agent_class()
                    self.logger.info("Created agent instance with no parameters")
                    return DirectAgentAdapter(agent)
                    
            except Exception as e:
                self.logger.error(f"Failed to create agent instance: {e}")
                raise
                
        except Exception as e:
            self.logger.error(f"Failed to load agent from {agent_path}: {e}")
            raise
    
    def load_agent_a2a(self, agent_url: str) -> AgentAdapter:
        """
        Load an agent via A2A protocol (A2A Mode).
        
        Args:
            agent_url: URL of the white agent (e.g., "http://localhost:5002")
            
        Returns:
            AgentAdapter wrapping the A2A client
        """
        self.logger.info(f"Loading agent via A2A from: {agent_url}")
        
        try:
            client = A2AAgentClient(agent_url)
            
            # Perform health check
            if not client.health_check():
                self.logger.error(f"Agent at {agent_url} failed health check")
                self.logger.error(f"Make sure the white agent server is running.")
                self.logger.error(f"Start it with: python3 example_white_agent_server.py --port {agent_url.split(':')[-1] if ':' in agent_url else '5002'}")
                raise ValueError(f"Agent at {agent_url} failed health check. Is the white agent server running?")
            
            self.logger.info(f"Successfully connected to agent at {agent_url}")
            return A2AAgentAdapter(client)
            
        except requests.exceptions.ConnectionError as e:
            port = agent_url.split(':')[-1] if ':' in agent_url else '5002'
            self.logger.error(f"Failed to connect to agent at {agent_url}")
            self.logger.error(f"Connection refused - the white agent server is not running.")
            self.logger.error(f"Start it with: python3 example_white_agent_server.py --port {port}")
            raise ValueError(f"Could not connect to agent at {agent_url}. Start the white agent server first.")
        except Exception as e:
            self.logger.error(f"Failed to load agent via A2A from {agent_url}: {e}")
            raise
    
    def evaluate_agent_on_task(self, agent: AgentAdapter, task_name: str, max_steps: int = 50) -> Dict[str, Any]:
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
            obs_preprocessed = agent.obs_preprocessor(obs)
            
            # Run the agent
            while step_count < max_steps:
                try:
                    # Get agent's action
                    action, agent_info = agent.get_action(obs_preprocessed.copy())
                    
                    if action is None:
                        self.logger.info("Agent returned None action, ending evaluation")
                        break
                    
                    # Execute action in environment
                    obs, reward, terminated, truncated, env_info = env.step(action)
                    obs_preprocessed = agent.obs_preprocessor(obs)
                    
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
            self.logger.error(f"Evaluation failed: {e}")
            return {
                "task_name": task_name,
                "success": False,
                "steps_taken": 0,
                "total_reward": 0,
                "max_steps": max_steps,
                "error": str(e),
                "exp_dir": str(exp_dir),
                "timestamp": timestamp
            }
    
    def evaluate_agent_on_benchmark_suite(self, agent: AgentAdapter, task_list: List[str]) -> Dict[str, Any]:
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
        agent_name = "A2AAgent" if isinstance(agent, A2AAgentAdapter) else str(agent.agent.__class__.__name__)
        final_results = {
            "agent_evaluated": agent_name,
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


def create_a2a_server(evaluator: GreenEvaluator, port: int = 5001):
    """
    Create and configure the Flask server for A2A protocol.
    
    Args:
        evaluator: The GreenEvaluator instance
        port: Port to run the server on
    """
    if not FLASK_AVAILABLE:
        raise ImportError("Flask is required for A2A server mode. Install it with: pip install flask")
    
    app = Flask(__name__)
    
    # Store evaluator instance for use in routes
    app.config['evaluator'] = evaluator
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "service": "GreenEvaluator",
            "version": "1.0.0"
        }), 200
    
    @app.route('/evaluate_task', methods=['POST'])
    def evaluate_task():
        """
        A2A endpoint to evaluate an agent on a single task.
        
        Expected JSON payload:
        {
            "agent_url": "http://localhost:5002",  # URL of white agent (A2A mode)
            "agent_path": "/path/to/agent.py",       # OR path to agent file (Direct mode)
            "task_name": "miniwob.click-dialog",
            "max_steps": 50,
            "mode": "a2a"  # or "direct"
        }
        """
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            task_name = data.get('task_name')
            max_steps = data.get('max_steps', 50)
            mode = data.get('mode', 'a2a')  # 'a2a' or 'direct'
            agent_url = data.get('agent_url')
            agent_path = data.get('agent_path')
            
            if not task_name:
                return jsonify({"error": "task_name is required"}), 400
            
            evaluator = app.config['evaluator']
            
            # Load agent based on mode
            if mode == 'a2a':
                if not agent_url:
                    return jsonify({"error": "agent_url is required for A2A mode"}), 400
                agent = evaluator.load_agent_a2a(agent_url)
            else:
                if not agent_path:
                    return jsonify({"error": "agent_path is required for direct mode"}), 400
                agent = evaluator.load_agent(agent_path)
            
            # Run evaluation
            result = evaluator.evaluate_agent_on_task(agent, task_name, max_steps)
            
            return jsonify(result), 200
            
        except Exception as e:
            evaluator = app.config.get('evaluator')
            if evaluator:
                evaluator.logger.error(f"Error in /evaluate_task: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/evaluate_suite', methods=['POST'])
    def evaluate_suite():
        """
        A2A endpoint to evaluate an agent on multiple tasks.
        
        Expected JSON payload:
        {
            "agent_url": "http://localhost:5002",
            "agent_path": "/path/to/agent.py",
            "task_list": ["miniwob.click-dialog", "miniwob.choose-list"],
            "max_steps": 50,
            "mode": "a2a"
        }
        """
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            task_list = data.get('task_list')
            max_steps = data.get('max_steps', 50)
            mode = data.get('mode', 'a2a')
            agent_url = data.get('agent_url')
            agent_path = data.get('agent_path')
            
            if not task_list:
                return jsonify({"error": "task_list is required"}), 400
            
            evaluator = app.config['evaluator']
            
            # Load agent based on mode
            if mode == 'a2a':
                if not agent_url:
                    return jsonify({"error": "agent_url is required for A2A mode"}), 400
                agent = evaluator.load_agent_a2a(agent_url)
            else:
                if not agent_path:
                    return jsonify({"error": "agent_path is required for direct mode"}), 400
                agent = evaluator.load_agent(agent_path)
            
            # Run evaluation
            result = evaluator.evaluate_agent_on_benchmark_suite(agent, task_list)
            
            return jsonify(result), 200
            
        except Exception as e:
            evaluator = app.config.get('evaluator')
            if evaluator:
                evaluator.logger.error(f"Error in /evaluate_suite: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/info', methods=['GET'])
    def info():
        """Get information about the Green Evaluator service."""
        return jsonify({
            "service": "GreenEvaluator",
            "version": "1.0.0",
            "modes": ["direct", "a2a"],
            "endpoints": {
                "health": "/health",
                "evaluate_task": "/evaluate_task",
                "evaluate_suite": "/evaluate_suite",
                "info": "/info"
            }
        }), 200
    
    return app, port


def main():
    """Main function to run the Green Evaluator."""
    parser = argparse.ArgumentParser(
        description="Green Evaluator - Test web agents on BrowserGym benchmarks"
    )
    
    # Mode selection
    parser.add_argument(
        "--server_mode",
        action="store_true",
        help="Run as A2A server (for AgentBeats platform integration)"
    )
    
    # Agent loading (mutually exclusive)
    agent_group = parser.add_mutually_exclusive_group(required=False)
    agent_group.add_argument(
        "--agent_path", 
        type=str, 
        help="Path to the agent Python file to evaluate (Direct Import Mode)"
    )
    agent_group.add_argument(
        "--agent_url",
        type=str,
        help="URL of the agent to evaluate (A2A Mode, e.g., 'http://localhost:5002')"
    )
    
    # Server options
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port for A2A server (default: 5001)"
    )
    
    # Evaluation options
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
    
    # Initialize the Green Evaluator
    evaluator = GreenEvaluator(results_dir=args.results_dir)
    
    # Check if required environment variables are set
    if not os.getenv('MINIWOB_URL'):
        evaluator.logger.error("MINIWOB_URL environment variable not set!")
        evaluator.logger.error("Please run: source .env")
        sys.exit(1)
    
    # Note: OPENAI_API_KEY is only needed by the white agent, not the green evaluator
    # We'll only warn, not fail, since the agent might handle its own API key
    if not os.getenv('OPENAI_API_KEY'):
        evaluator.logger.warning("OPENAI_API_KEY environment variable not set!")
        evaluator.logger.warning("This is needed if your white agent uses OpenAI. The green evaluator will continue.")
        evaluator.logger.warning("If evaluation fails, set it with: export OPENAI_API_KEY='your-key-here'")
    
    # Server mode: Run as A2A server
    if args.server_mode:
        if not FLASK_AVAILABLE:
            evaluator.logger.error("Flask is required for server mode. Install it with: pip install flask")
            sys.exit(1)
        
        app, port = create_a2a_server(evaluator, args.port)
        evaluator.logger.info(f"Starting Green Evaluator A2A server on port {port}")
        evaluator.logger.info(f"Server endpoints:")
        evaluator.logger.info(f"  - GET  /health")
        evaluator.logger.info(f"  - POST /evaluate_task")
        evaluator.logger.info(f"  - POST /evaluate_suite")
        evaluator.logger.info(f"  - GET  /info")
        app.run(host='0.0.0.0', port=port, debug=False)
        return
    
    # Client mode: Run evaluation
    if not args.agent_path and not args.agent_url:
        parser.error("Either --agent_path or --agent_url must be provided (or use --server_mode)")
    
    try:
        # Load the agent to evaluate
        if args.agent_url:
            # A2A Mode
            evaluator.logger.info("Running in A2A Mode")
            agent = evaluator.load_agent_a2a(args.agent_url)
        else:
            # Direct Import Mode
            evaluator.logger.info("Running in Direct Import Mode")
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
            print(f"\nFull JSON Result:")
            print(json.dumps(result, indent=2))
            
        else:
            # Evaluate on full benchmark suite
            evaluator.logger.info("Full benchmark suite evaluation")
            task_list = get_miniwob_task_list()
            results = evaluator.evaluate_agent_on_benchmark_suite(agent, task_list)
            
            print(f"\nBenchmark Suite Results:")
            print(f"Agent: {results['agent_evaluated']}")
            print(f"Success Rate: {results['success_rate']:.2%}")
            print(f"Average Steps: {results['average_steps']:.1f}")
            print(f"Average Reward: {results['average_reward']:.2f}")
            print(f"Successful Tasks: {results['successful_tasks']}/{results['total_tasks']}")
            print(f"\nFull JSON Result:")
            print(json.dumps(results, indent=2))
    
    except Exception as e:
        evaluator.logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
