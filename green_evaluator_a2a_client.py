#!/usr/bin/env python3
"""
A2A Client Module for Green Evaluator

This module provides functionality to communicate with White Agents via HTTP
using the Agent-to-Agent (A2A) protocol.
"""

import json
import logging
import requests
from typing import Dict, Any, Optional, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("A2AClient")


class A2AAgentClient:
    """
    Client for communicating with White Agents via A2A protocol.
    
    This class handles HTTP communication with white agents that expose
    an A2A-compliant API.
    """
    
    def __init__(self, agent_url: str, timeout: int = 30):
        """
        Initialize the A2A client.
        
        Args:
            agent_url: Base URL of the white agent (e.g., "http://localhost:5002")
            timeout: Request timeout in seconds
        """
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
        
        logger.info(f"A2A Client initialized for agent: {self.agent_url}")
    
    def health_check(self) -> bool:
        """
        Check if the white agent is healthy and responding.
        
        Returns:
            True if agent is healthy, False otherwise
        """
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
    
    def get_action(self, obs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Send observation to white agent and get action back.
        
        This is the core A2A communication method.
        
        Args:
            obs: Observation dictionary from the environment
            
        Returns:
            Tuple of (action_string, agent_info_dict)
            
        Raises:
            requests.RequestException: If communication fails
        """
        try:
            # Prepare request payload
            # Convert numpy arrays to lists for JSON serialization
            payload = self._prepare_payload(obs)
            
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
            
            logger.debug(f"Received action from {self.agent_url}: {action[:50]}...")
            
            return action, agent_info
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get action from {self.agent_url}: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response from {self.agent_url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error communicating with {self.agent_url}: {e}")
            raise
    
    def get_action_set_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the white agent's action set.
        
        Returns:
            Dictionary with action set information, or None if not available
        """
        try:
            response = self.session.get(
                f"{self.agent_url}/action_set",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.warning(f"Could not get action set info: {e}")
            return None
    
    def _prepare_payload(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare observation for JSON serialization.
        
        Converts numpy arrays and other non-serializable types to JSON-compatible formats.
        
        Args:
            obs: Raw observation dictionary
            
        Returns:
            JSON-serializable observation dictionary
        """
        import numpy as np
        from PIL import Image
        
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
            elif isinstance(value, list):
                # Process lists (may contain numpy arrays or nested dicts)
                processed_list = []
                for item in value:
                    if isinstance(item, dict):
                        processed_list.append(self._prepare_payload(item))
                    elif isinstance(item, np.ndarray):
                        if item.ndim == 0:
                            processed_list.append(item.item())
                        else:
                            processed_list.append(item.tolist())
                    elif isinstance(item, (Image.Image,)):
                        import base64
                        import io
                        buffer = io.BytesIO()
                        if item.mode in ("RGBA", "LA"):
                            item = item.convert("RGB")
                        item.save(buffer, format="JPEG")
                        image_base64 = base64.b64encode(buffer.getvalue()).decode()
                        processed_list.append(f"data:image/jpeg;base64,{image_base64}")
                    else:
                        processed_list.append(item)
                payload[key] = processed_list
            else:
                # For other types (str, int, float, bool, None), keep as is
                payload[key] = value
        
        return payload

