import base64
import dataclasses
import io
import logging
import time

import numpy as np
import openai
from PIL import Image

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import AbstractAgentArgs, Agent
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html

logger = logging.getLogger(__name__)


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


class WhiteAgent(Agent):
    """
    White Agent for BrowserGym benchmarks.
    
    This agent uses OpenAI's GPT models to interact with web environments.
    It implements a chain-of-thought reasoning process to decide on actions.
    """

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

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        chat_mode: bool = False,
        demo_mode: str = "off",
        use_html: bool = False,
        use_axtree: bool = True,
        use_screenshot: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.chat_mode = chat_mode
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot

        if not (use_html or use_axtree):
            raise ValueError(f"Either use_html or use_axtree must be set to True.")

        self.openai_client = openai.OpenAI()

        self.action_set = HighLevelActionSet(
            subsets=["chat", "tab", "nav", "bid", "infeas"],
            strict=False,
            multiaction=False,
            demo_mode=demo_mode,
        )

        self.action_history = []

    def get_action(self, obs: dict) -> tuple[str, dict]:
        system_msgs = []
        user_msgs = []

        # System Instruction with CoT emphasis
        system_msgs.append(
            {
                "type": "text",
                "text": """\
# Instructions

You are a capable UI Assistant. Your goal is to help the user perform tasks using a web browser.
You have access to a web browser that you can interact with via specific commands.

## Reasoning Strategy
1. Analyze the user's goal and the current page state (URL, content, accessibility tree).
2. Review past actions and any errors to avoid repeating mistakes.
3. Formulate a step-by-step plan to achieve the goal.
4. Select the best single next action from the available action space.

Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.
""",
            }
        )

        if self.chat_mode:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Chat Messages
""",
                }
            )
            for msg in obs["chat_messages"]:
                if msg["role"] in ("user", "assistant", "infeasible"):
                    user_msgs.append(
                        {
                            "type": "text",
                            "text": f"""\
- [{msg['role']}] {msg['message']}
""",
                        }
                    )
                elif msg["role"] == "user_image":
                    user_msgs.append({"type": "image_url", "image_url": msg["message"]})
                else:
                    raise ValueError(f"Unexpected chat message role {repr(msg['role'])}")

        else:
            assert obs["goal_object"], "The goal is missing."
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Goal
""",
                }
            )
            user_msgs.extend(obs["goal_object"])

        # append url of all open tabs
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Currently open tabs
""",
            }
        )
        for page_index, (page_url, page_title) in enumerate(
            zip(obs["open_pages_urls"], obs["open_pages_titles"])
        ):
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
Tab {page_index}{" (active tab)" if page_index == obs["active_page_index"] else ""}
  Title: {page_title}
  URL: {page_url}
""",
                }
            )

        # append page AXTree (if asked)
        if self.use_axtree:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Current page Accessibility Tree

{obs["axtree_txt"]}

""",
                }
            )
        # append page HTML (if asked)
        if self.use_html:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Current page DOM

{obs["pruned_html"]}

""",
                }
            )

        # append page screenshot (if asked)
        if self.use_screenshot:
            user_msgs.append(
                {
                    "type": "text",
                    "text": """\
# Current page Screenshot
""",
                }
            )
            user_msgs.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_jpg_base64_url(obs["screenshot"]),
                        "detail": "auto",
                    },
                }
            )

        # append action space description
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Action Space

{self.action_set.describe(with_long_description=False, with_examples=True)}

Here are examples of actions with chain-of-thought reasoning:

I now need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.
```click("12")```

I found the information requested by the user, I will send it to the chat.
```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```

""",
            }
        )

        # append past actions
        if self.action_history:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# History of past actions
""",
                }
            )
            user_msgs.extend(
                [
                    {
                        "type": "text",
                        "text": f"""\

{action}
""",
                    }
                    for action in self.action_history
                ]
            )

            if obs["last_action_error"]:
                user_msgs.append(
                    {
                        "type": "text",
                        "text": f"""\
# Error message from last action

{obs["last_action_error"]}

""",
                    }
                )

        # ask for the next action
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Next action

You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the page before deciding on your next action.
""",
            }
        )

        prompt_text_strings = []
        for message in system_msgs + user_msgs:
            if message["type"] == "text":
                prompt_text_strings.append(message["text"])

        full_prompt_txt = "\n".join(prompt_text_strings)
        logger.info(full_prompt_txt)

        # query OpenAI model
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msgs},
                {"role": "user", "content": user_msgs},
            ],
            temperature=0.0, # Deterministic for evaluation
        )
        action = response.choices[0].message.content

        self.action_history.append(action)

        return action, {}


@dataclasses.dataclass
class WhiteAgentArgs(AbstractAgentArgs):
    """
    Arguments for the White Agent.
    """

    model_name: str = "gpt-4o-mini"
    chat_mode: bool = False
    demo_mode: str = "off"
    use_html: bool = False
    use_axtree: bool = True
    use_screenshot: bool = False

    def make_agent(self):
        return WhiteAgent(
            model_name=self.model_name,
            chat_mode=self.chat_mode,
            demo_mode=self.demo_mode,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
        )

# ============================================================================
# A2A Server Implementation
# ============================================================================

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
    A2A_AVAILABLE = True
except ImportError:
    A2A_AVAILABLE = False
    logger.warning("FastAPI/Uvicorn not found. A2A server capabilities disabled.")

if A2A_AVAILABLE:
    class WhiteAgentServer:
        """A2A Server wrapper for White Agent"""
        
        def __init__(self, agent: WhiteAgent, port: int = 8000):
            self.agent = agent
            self.port = port
            self.app = FastAPI(title="BrowserGym White Agent", version="1.0.0")
            
            # Setup routes
            self._setup_routes()
        
        def _setup_routes(self):
            """Setup A2A protocol routes"""
            
            @self.app.get("/")
            async def root():
                return {
                    "name": "BrowserGym White Agent",
                    "description": "Generalist Web Agent using GPT-4o-mini",
                    "version": "1.0.0",
                    "a2a_protocol": True
                }
            
            @self.app.get("/card")
            async def get_card():
                """Return agent card information"""
                return {
                    "name": "BrowserGym White Agent",
                    "description": "Generalist Web Agent using GPT-4o-mini and CoT",
                    "url": f"http://localhost:{self.port}",
                    "capabilities": [
                        "miniwob_benchmark",
                        "workarena_benchmark",
                        "webarena_benchmark",
                        "agent_solver"
                    ]
                }
            
            @self.app.get("/status")
            async def status():
                """Health check endpoint"""
                return {"status": "healthy"}

            @self.app.get("/health")
            async def health():
                """Health check endpoint"""
                return {"status": "healthy"}
            
            @self.app.post("/a2a/message")
            async def handle_a2a_message(request: Request):
                """Handle incoming A2A messages - Basic logging for solver"""
                try:
                    body = await request.json()
                    logger.info(f"Received A2A message: {body.get('type')}")
                    return {"status": "received"}
                except Exception as e:
                    logger.error(f"Error handling A2A message: {e}")
                    return JSONResponse(status_code=500, content={"error": str(e)})

        def run(self):
            uvicorn.run(self.app, host="0.0.0.0", port=self.port)


def main():
    import argparse
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run White Agent")
    parser.add_argument("--a2a-server", action="store_true", help="Run as A2A server")
    parser.add_argument("--port", type=int, default=8000, help="Port for A2A server")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    
    args = parser.parse_args()
    
    # Initialize the agent
    agent_args = WhiteAgentArgs(model_name=args.model_name)
    agent = agent_args.make_agent()
    
    if args.a2a_server:
        if not A2A_AVAILABLE:
            print("Error: FastAPI/Uvicorn not installed. Cannot run A2A server.")
            return
            
        print(f"Starting White Agent A2A Server on port {args.port}...")
        server = WhiteAgentServer(agent, port=args.port)
        server.run()
    else:
        print("White Agent initialized. ready to be loaded by Green Evaluator.")

if __name__ == "__main__":
    main()
