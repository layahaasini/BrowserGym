#!/usr/bin/env python3
"""
Example White Agent A2A Server

This script demonstrates how to create a white agent that exposes
an A2A-compatible HTTP interface for use with the Green Evaluator.

Usage:
    python3 example_white_agent_server.py --port 5002
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    from flask import Flask, request, jsonify
except ImportError:
    print("Error: Flask is required. Install it with: pip install flask")
    sys.exit(1)

# Add demo_agent to path if needed
sys.path.insert(0, str(Path(__file__).parent))

try:
    from demo_agent.agent import DemoAgent
except ImportError:
    print("Error: Could not import DemoAgent. Make sure demo_agent is available.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WhiteAgentServer")


def create_white_agent_server(port: int = 5002):
    """Create and configure the white agent A2A server."""
    
    app = Flask(__name__)
    
    # Initialize the agent
    logger.info("Initializing DemoAgent...")
    agent = DemoAgent(
        model_name='gpt-4o-mini',
        chat_mode=False,  # Benchmark mode
        demo_mode='off',
        use_html=False,
        use_axtree=True,
        use_screenshot=False
    )
    logger.info("Agent initialized successfully")
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "service": "WhiteAgent",
            "agent_type": agent.__class__.__name__
        }), 200
    
    @app.route('/get_action', methods=['POST'])
    def get_action():
        """
        A2A endpoint to get an action from the agent.
        
        Expected JSON payload:
        {
            "obs": {
                "chat_messages": [...],
                "screenshot": "...",
                "goal_object": {...},
                ...
            }
        }
        """
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400
            
            obs = data.get('obs', {})
            
            if not obs:
                return jsonify({"error": "No observation provided"}), 400
            
            # Preprocess observation using agent's preprocessor
            logger.debug("Preprocessing observation...")
            obs_preprocessed = agent.obs_preprocessor(obs)
            
            # Get action from agent
            logger.debug("Getting action from agent...")
            action, agent_info = agent.get_action(obs_preprocessed)
            
            logger.debug(f"Action received: {action[:50] if action else 'None'}...")
            
            # Convert agent_info to dict if it's not already
            if hasattr(agent_info, '__dict__'):
                agent_info = agent_info.__dict__
            elif not isinstance(agent_info, dict):
                agent_info = {}
            
            return jsonify({
                "action": action or "",
                "agent_info": agent_info
            }), 200
            
        except Exception as e:
            logger.error(f"Error in /get_action: {e}", exc_info=True)
            return jsonify({
                "error": str(e),
                "action": "",
                "agent_info": {}
            }), 500
    
    @app.route('/action_set', methods=['GET'])
    def action_set():
        """Get information about the agent's action set."""
        try:
            return jsonify({
                "action_set_type": type(agent.action_set).__name__,
                "description": agent.action_set.describe() if hasattr(agent.action_set, 'describe') else "N/A"
            }), 200
        except Exception as e:
            logger.error(f"Error in /action_set: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/info', methods=['GET'])
    def info():
        """Get information about the white agent service."""
        return jsonify({
            "service": "WhiteAgent",
            "agent_type": agent.__class__.__name__,
            "endpoints": {
                "health": "/health",
                "get_action": "/get_action",
                "action_set": "/action_set",
                "info": "/info"
            }
        }), 200
    
    return app, port


def main():
    """Main function to run the white agent server."""
    parser = argparse.ArgumentParser(
        description="White Agent A2A Server - Example implementation"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5002,
        help="Port to run the server on (default: 5002)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    
    # Check for required environment variables
    import os
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable not set!")
        logger.error("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    try:
        app, port = create_white_agent_server(args.port)
        logger.info(f"Starting White Agent A2A server on {args.host}:{port}")
        logger.info(f"Server endpoints:")
        logger.info(f"  - GET  /health")
        logger.info(f"  - POST /get_action")
        logger.info(f"  - GET  /action_set")
        logger.info(f"  - GET  /info")
        logger.info(f"\nTo test with Green Evaluator:")
        logger.info(f"  python3 green_evaluator.py --agent_url http://localhost:{port} --task miniwob.click-dialog")
        
        app.run(host=args.host, port=port, debug=False)
        
    except KeyboardInterrupt:
        logger.info("\nShutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

