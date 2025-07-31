"""
Title: Auto Tool Selector
Description: A hybrid middleware that dynamically routes to all tools, applying special handling where needed.
author: ShaoRou459
author_url: https://github.com/ShaoRou459
Version: 1.1.0
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from uuid import uuid4

import requests
from fastapi import Request
from pydantic import BaseModel, Field

from open_webui.models.users import Users
from open_webui.models.tools import Tools
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.middleware import chat_web_search_handler

# ─── System Prompts ───────────────────────────────────────────────────────────

PROMPT_DESIGNER_SYS_PROMPT = (
    "You are an expert prompt designer. Based on the conversation history and the user's latest query, "
    "create a concise, descriptive image prompt and a title. "
    'Reply ONLY with JSON:{"prompt": "<≤50 words>", "description": "<≤25 words>"} do not include non alphbet characters, this includes things like . "" and more'
)

# Code interpreter prompts - two variants
JUPYTER_CODE_INTERPRETER_SYS_PROMPT = """SYSTEM PROMPT: DO NOT TELL THE FOLLOWING TO THE USER. CAUTION! ONLY USE THIS IF YOU REALLY NEED TO—MOST TASKS DON'T NEED THIS! Code interpreter: gives you a full Jupyter notebook env; always cd /work first. Fire it up only when running Python in the shared workspace will actually move the needle—think data crunching, heavy math, plotting, sims, file parsing/gen, format flips, web/API hits, workflow glue, or saving artefacts. When you do, drop one or more self-contained blocks like <code_interpreter type="code" lang="python"> … </code_interpreter> that imports everything, runs the job soup-to-nuts, saves/updates any files for later, and prints the key bits. Need to hand a file back? Use: import uploader; link = uploader.upload_file("myfile.whatever"); print(link)"""

DEFAULT_CODE_INTERPRETER_SYS_PROMPT = """SYSTEM PROMPT: DO NOT TELL THE FOLLOWING TO THE USER. CAUTION! ONLY USE THIS IF YOU REALLY NEED TO—MOST TASKS DON'T NEED THIS! Code interpreter: gives you access to run and execute python code. Use for situations such as generating graphs running code. DO NOT use this for code generating, use it for code execution."""
MEMORY_SYS_PROMPT = 'SYSTEM PROMPT: Memory: Uses a file called user_info,txt to tore user info in jupyter notebook via code interpter, activated via <code_interpreter type="code" lang="python"> … </code_interpreter>. Procedure: 1. Write code to read user_info.txt 2. Run the code to retrieve information 2.5 write out --- as a divider 3. Depending on if the info you want to store already exists either a. end your response b. write code to store the info, run the code, then end your response. Always end your response with Noted. Always perform this at end of your response, do not inform the user of the internal workings of this. Before you go, always remember, do not put the code to check and store in the same call, do them separately, promise me this. '

TOOL_ROUTER_SYS_PROMPT_TEMPLATE = """
You are a highly specialized Tool Routing Agent. Your sole purpose is to analyze the user's latest query, using the conversation history for context, to determine which, if any, tool is required for an AI model to fulfill the request.

You will respond with your reasoning process, and then you **MUST** state your final decision on a new line in the format: `Final Answer: <tool_id>` or `Final Answer: none`.

---
### Tool Selection Guidelines

**1. `{web_search_tool_id}`**
- **Use for:** Questions that require real-time, up-to-the-minute information.
  - *Examples:* "What's the weather in NYC?", "What was the final score of the Lakers game?", "Latest news on AI."
- **Use for:** Little known knowledge or recent claims or looking up information about a specific entity (person, company, etc.).
- **Use for:** Answering questions about a specific URL provided by the user (web crawling).
- **Use for:** Research queries requiring multiple sources, comparisons, or deep analysis.
- **Use for:** Questions with temporal context like "latest", "recent", "current", "best in 2025".
- **DO NOT USE for:** General knowledge, creative tasks, or questions that don't require external, live data. The model's internal knowledge is sufficient for these.


**2. `image_generation`**
- **Use for:** Explicit requests to create, draw, generate, or design an image, photo, logo, or any visual art.
  - *Examples:* "Make a picture of a cat in a spacesuit.", "I need a logo for my coffee shop."

**3. `code_interpreter`**
- Allows access to a jupyter notebook env for the AI assistant to run code and get results. 
- **Use for:**
  - For when the user gives a task that requires the *excution* of python code.
  - For when the user gives a task that involves File manipulation WITHIN the JUPYTER ENV: Such as storing files, reading files etc. 
  - DO NOT USE FOR: Non python tasks, tasks that only involve code generation (code does not need to be excuted for AI assistant to see), file genereration tasks (not storing anything permantly)
**4. `memory`**
- **Use for:** Explicit requests to remember or recall specific, simple pieces of information within the current conversation.
  - *Examples:* "Remember my name is Peter.", "What was the idea I mentioned earlier about the marketing plan?"
- **DO NOT USE for:** File operations or complex data recall (that's for `code_interpreter`).

---
### **How to Use Conversation History**

The user's latest message might be short or use pronouns (like "it", "that", "them"). You **must** look at the conversation history to understand what they are referring to. The history provides the *subject* for the action in the latest query.

-   If the user says, "Graph that for me," check the history to see what data "that" refers to.
-   If the user says, "Save it as a text file," check the history to find the content for "it."

---

### **Core Principles (VERY IMPORTANT)**

1.  **Latest Query is the Trigger:** Your decision is always triggered by the user's **most recent message**. The history is for clarification ONLY. Do not call a tool just because it was used in a previous turn. Your choice must be for the user's LATEST query.
2.  **Default to `none`:** When in doubt, choose `none`. It is significantly better to not use a tool than to use one incorrectly. The model should answer directly if it can.
3.  **No Vague Guesses:** If the user's request is ambiguous or lacks context, even with history (e.g., "What about this?", "And what now?"), choose `none`. Do not try to guess.
4.  **Common Sense Is Key:** Always use common sense in making decision, sometimes a tool might seem to be the right call, but use common sense, what does this sutiation really require? What needs to be completed first?
---
### Final Output Format
<think>
- Identify user intent
- Identify user needs
- Pick the user need that must be stastifyed before the others
- Identify which tool best suits the user's need
- Double check if the tool needs to be used
- Done
</think>
Final Answer: <The chosen tool_id or none>
---
ONLY EVER RETURN Final Answer: THE TOOL'S ID, NEVER PUT IT IN ANY SORT OF FORMMATING OR QUOTES. 
"""


# ─── Debug System ────────────────────────────────────────────────────────────
class Debug:
    """Structured debug logging system for AutoToolSelector."""

    # ANSI color codes
    _COLORS = {
        "RESET": "\x1b[0m",
        "BOLD": "\x1b[1m",
        "DIM": "\x1b[2m",
        "CYAN": "\x1b[96m",
        "GREEN": "\x1b[92m",
        "YELLOW": "\x1b[93m",
        "RED": "\x1b[91m",
        "MAGENTA": "\x1b[95m",
        "BLUE": "\x1b[94m",
    }

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def _format_msg(self, category: str, message: str, color: str = "CYAN") -> str:
        """Format a debug message with consistent styling."""
        if not self.enabled:
            return ""

        timestamp = ""  # Can add timestamp if needed
        prefix = f"{self._COLORS['MAGENTA']}{self._COLORS['BOLD']}[AutoToolSelector]{self._COLORS['RESET']}"
        cat_colored = f"{self._COLORS[color]}{self._COLORS['BOLD']}{category}{self._COLORS['RESET']}"
        msg_colored = f"{self._COLORS[color]}{message}{self._COLORS['RESET']}"

        return f"{prefix} {cat_colored}: {msg_colored}"

    def _log(self, category: str, message: str, color: str = "CYAN") -> None:
        """Internal logging method."""
        if self.enabled:
            formatted = self._format_msg(category, message, color)
            if formatted:
                print(formatted, file=sys.stderr)

    def router(self, message: str) -> None:
        """Log router decision making."""
        self._log("ROUTER", message, "BLUE")

    def vision(self, message: str) -> None:
        """Log vision processing."""
        self._log("VISION", message, "GREEN")

    def tool(self, message: str) -> None:
        """Log tool activation."""
        self._log("TOOL", message, "YELLOW")

    def handler(self, message: str) -> None:
        """Log special handler activity."""
        self._log("HANDLER", message, "MAGENTA")

    def error(self, message: str) -> None:
        """Log errors and warnings."""
        self._log("ERROR", message, "RED")

    def flow(self, message: str) -> None:
        """Log general workflow steps."""
        self._log("FLOW", message, "CYAN")

    def data(self, label: str, data: Any, truncate: int = 80) -> None:
        """Log data with optional truncation."""
        if not self.enabled:
            return

        if isinstance(data, str) and len(data) > truncate:
            data_str = f"{data[:truncate]}..."
        else:
            data_str = str(data)

        self._log("DATA", f"{label} → {data_str}", "DIM")


# Legacy compatibility - will be replaced
def _debug(msg: str) -> None:
    """Legacy debug function - use Debug class instead."""
    print(
        f"{Debug._COLORS['MAGENTA']}{Debug._COLORS['BOLD']}[AutoToolSelector]{Debug._COLORS['RESET']}{Debug._COLORS['CYAN']} {msg}{Debug._COLORS['RESET']}",
        file=sys.stderr,
    )


# ─── Content Parsing Helpers ──────────────────────────────────────────────────
def _get_text_from_message(message_content: Any) -> str:
    """Extracts only the text part of a message, ignoring image data URLs."""
    if isinstance(message_content, list):
        text_parts = []
        for part in message_content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return " ".join(text_parts)
    elif isinstance(message_content, str):
        return message_content
    return ""


def _get_message_parts(message_content: Any) -> Tuple[str, List[str]]:
    """Extracts text and image data URLs from a message's content."""
    if isinstance(message_content, list):
        text_parts = []
        image_urls = []
        for part in message_content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    image_urls.append(part["image_url"]["url"])
        return " ".join(text_parts), image_urls
    elif isinstance(message_content, str):
        return message_content, []
    return "", []


def get_last_user_message_content(messages: List[Dict[str, Any]]) -> Any:
    """Gets the entire content object of the last user message."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def get_last_user_message(messages: List[Dict[str, Any]]) -> str:
    """Gets the last user message's text content, ignoring images."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return _get_text_from_message(message.get("content", ""))
    return ""


# ─── Regex & Keyword Helpers ──────────────────────────────────────────────────
_JSON_RE = re.compile(r"\{.*?\}", re.S)
_URL_RE = re.compile(r"https?://\S+")


# ─── Image Gen Helpers ────────────────────────────────────────────────────────
def _parse_json_fuzzy(text: str, debug: Debug = None) -> Dict[str, str]:
    raw = text.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = "\n".join(raw.splitlines()[1:-1])
    m = _JSON_RE.search(raw)
    if m:
        raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception as e:
        if debug:
            debug.error(f"JSON parse error → {e}. Raw: {raw[:80]}…")
        return {}


async def _generate_prompt_and_desc(
    request: Request,
    user: Any,
    model: str,
    convo_snippet: str,
    user_query: str,
    debug: Debug = None,
) -> Tuple[str, str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": PROMPT_DESIGNER_SYS_PROMPT},
            {
                "role": "user",
                "content": f"Conversation so far:\n{convo_snippet}\n\nUser query: {user_query}",
            },
        ],
        "stream": False,
    }

    try:
        res = await generate_chat_completion(
            request=request, form_data=payload, user=user
        )
        obj = _parse_json_fuzzy(res["choices"][0]["message"]["content"], debug)
        prompt = obj.get("prompt", user_query)
        description = obj.get("description", "Image generated from conversation.")
        if debug:
            debug.handler(f"Router prompt → {prompt[:60]}… | desc: {description}")
        return prompt, description
    except Exception as exc:
        if debug:
            debug.error(f"Prompt‑designer error → {exc}")
        return user_query, "Image generated."


# ─── Special Tool Handlers ────────────────────────────────────────────────────
async def flux_image_generation_handler(
    request: Any, body: dict, ctx: dict, user: Any, debug: Debug = None
) -> dict:
    prompt: str = ctx.get("prompt") or get_last_user_message(body["messages"])
    description: str = ctx.get("description", "Image generated.")
    emitter = ctx.get("__event_emitter__")

    placeholder_id = str(uuid4())
    placeholder = {"id": placeholder_id, "role": "assistant", "content": ""}
    body["messages"].append(placeholder)

    if emitter:
        await emitter({"type": "chat:message", "data": placeholder})
        await emitter(
            {
                "type": "status",
                "data": {
                    "message_id": placeholder_id,
                    "description": f'Generating image from prompt: "{prompt[:60]}..."',
                    "done": False,
                },
            }
        )

    if debug:
        debug.handler(f"Calling Flux with prompt → {prompt[:80]}…")
    try:
        resp = await generate_chat_completion(
            request=request,
            form_data={
                "model": "gpt-4o-image",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            user=user,
        )
        flux_reply = resp["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        if debug:
            debug.error(f"Flux error → {exc}")
        fail = f"❌ Image generation failed: {exc}"
        if emitter:
            await emitter(
                {
                    "type": "replace",
                    "data": {"message_id": placeholder_id, "content": fail},
                }
            )
            await emitter(
                {
                    "type": "status",
                    "data": {
                        "message_id": placeholder_id,
                        "description": "Failed",
                        "done": True,
                    },
                }
            )
        body["messages"].pop()
        return body

    url_match = _URL_RE.search(flux_reply)
    image_url = url_match.group(0) if url_match else flux_reply
    if debug:
        debug.handler(f"✅ Flux URL → {image_url}")

    if emitter:
        await emitter({"type": "delete", "data": {"message_id": placeholder_id}})
        await emitter(
            {
                "type": "status",
                "data": {"message_id": placeholder_id, "description": "", "done": True},
            }
        )

    body["messages"].pop()

    meta = (
        "[IMAGE_GENERATED]\n"
        f"url: {image_url}\n"
        f"prompt: {prompt}\n"
        f"description: {description}\n"
        f'[IMAGE_INSTRUCTION] Embed the generated image using ![description](url) and add a one-sentence caption."'
    )
    body["messages"].append({"role": "system", "content": meta})
    return body


async def code_interpreter_handler(
    request: Request,
    body: dict,
    ctx: dict,
    user: Any,
    debug: Debug = None,
    use_jupyter: bool = True,
) -> dict:
    emitter = ctx.get("__event_emitter__")
    if emitter:
        await emitter(
            {
                "type": "status",
                "data": {
                    "description": "Preparing Python environment...",
                    "done": False,
                },
            }
        )

    # Choose the appropriate prompt based on the valve setting
    prompt = (
        JUPYTER_CODE_INTERPRETER_SYS_PROMPT
        if use_jupyter
        else DEFAULT_CODE_INTERPRETER_SYS_PROMPT
    )

    sys_msg = {
        "role": "system",
        "content": prompt,
    }
    body["messages"].append(sys_msg)
    body.setdefault("features", {})["code_interpreter"] = True
    if debug:
        interpreter_type = "Jupyter notebook" if use_jupyter else "basic code execution"
        debug.handler(f"🔧 Code Interpreter enabled for this turn ({interpreter_type})")
    return body


async def memory_handler(
    request: Request, body: dict, ctx: dict, user: Any, debug: Debug = None
) -> dict:
    emitter = ctx.get("__event_emitter__")
    if emitter:
        await emitter(
            {
                "type": "status",
                "data": {
                    "description": "Accessing conversational memory...",
                    "done": False,
                },
            }
        )

    sys_msg = {
        "role": "system",
        "content": MEMORY_SYS_PROMPT,
    }
    body["messages"].append(sys_msg)
    body.setdefault("features", {})["code_interpreter"] = True
    if debug:
        debug.handler("🔧 Code Interpreter enabled for this turn --Via memory")
    return body


# ─── Main Filter Class ────────────────────────────────────────────────────────
class Filter:
    class Valves(BaseModel):
        helper_model: Optional[str] = Field(default=None)
        vision_model: Optional[str] = Field(
            default=None,
            description="Optional vision model to describe images for context. If empty, images are ignored.",
        )
        history_char_limit: int = Field(
            default=500,
            description="Max characters per message in the history snippet.",
        )
        vision_injection_models: List[str] = Field(
            default=[],
            description="List of non-vision model IDs that should receive the image analysis text.",
        )
        use_exa_router_search: bool = Field(
            default=True,
            description="Toggle to use exa_router_search instead of default web_search when available.",
        )
        debug_enabled: bool = Field(
            default=False,
            description="Enable detailed debug logging for troubleshooting.",
        )
        use_jupyter_code_interpreter: bool = Field(
            default=True,
            description="Use Jupyter notebook environment for code interpreter. If False, uses basic code execution.",
        )

    class UserValves(BaseModel):
        auto_tools: bool = Field(default=True)

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()
        self.debug = Debug(enabled=False)  # Will be updated when valves change
        self.special_handlers = {
            "image_generation": flux_image_generation_handler,
            "code_interpreter": code_interpreter_handler,
            "memory": memory_handler,
            "web_search": chat_web_search_handler,
        }

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __request__: Request,
        __user__: dict | None = None,
        __model__: dict | None = None,
    ) -> dict:
        # Update debug state based on current valve setting
        self.debug.enabled = self.valves.debug_enabled
        self.debug.flow("Starting AutoToolSelector processing")

        if not self.user_valves.auto_tools:
            self.debug.flow("Auto tools disabled, skipping processing")
            return body

        messages = body.get("messages", [])
        if not messages:
            self.debug.flow("No messages found, skipping processing")
            return body

        self.debug.flow(f"Processing {len(messages)} messages")

        last_user_content_obj = get_last_user_message_content(messages)
        user_message_text, image_urls = _get_message_parts(last_user_content_obj)

        self.debug.data("User message text", user_message_text)
        self.debug.data("Image URLs found", len(image_urls))

        last_user_message_idx = next(
            (
                i
                for i, m in reversed(list(enumerate(messages)))
                if m.get("role") == "user"
            ),
            -1,
        )

        user_obj = Users.get_user_by_id(__user__["id"]) if __user__ else None

        routing_query = user_message_text
        image_analysis_started = False

        if self.valves.vision_model and image_urls:
            image_analysis_started = True
            self.debug.vision(
                f"Found {len(image_urls)} image(s). Sending to vision model: {self.valves.vision_model}"
            )
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Analyzing image...", "done": False},
                }
            )

            image_descriptions = []
            for idx, url in enumerate(image_urls):
                self.debug.vision(f"Analyzing image {idx + 1}/{len(image_urls)}...")
                vision_payload = {
                    "model": self.valves.vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this image in detail for a search or routing agent. What is it? What's happening? What text is visible?",
                                },
                                {"type": "image_url", "image_url": {"url": url}},
                            ],
                        }
                    ],
                }
                try:
                    res = await generate_chat_completion(
                        request=__request__, form_data=vision_payload, user=user_obj
                    )
                    description = res["choices"][0]["message"]["content"]
                    image_descriptions.append(
                        f"[Image {idx + 1} Context: {description}]"
                    )
                except Exception as e:
                    self.debug.error(f"Vision model failed for image {idx + 1}: {e}.")
                    image_descriptions.append(
                        f"[Image {idx + 1} Context: Analysis failed.]"
                    )

            if image_descriptions:
                full_image_context = "\n\n".join(image_descriptions)
                # This query is temporary for the router model only
                routing_query = f"{user_message_text}\n\n{full_image_context}"

        # Always strip images from non-vision models for ALL messages in history
        if __model__ and __model__.get("id") in self.valves.vision_injection_models:
            self.debug.vision(
                f"Stripping images from all messages for non-vision model: {__model__.get('id')}"
            )

            # Process all messages in the conversation history
            for msg_idx, message in enumerate(body["messages"]):
                if message.get("role") == "user":
                    msg_content = message.get("content", "")
                    msg_text, msg_image_urls = _get_message_parts(msg_content)

                    # If this message has images, strip them
                    if msg_image_urls:
                        # For the current/last user message, include vision analysis if available
                        if (
                            msg_idx == last_user_message_idx
                            and "full_image_context" in locals()
                        ):
                            final_text = f"{msg_text}\n\n{full_image_context}"
                        else:
                            final_text = msg_text

                        # Replace content with text-only version
                        body["messages"][msg_idx]["content"] = [
                            {
                                "type": "text",
                                "text": final_text,
                            }
                        ]
                        self.debug.vision(f"Stripped images from message {msg_idx}")

            # Only process current message images if they exist
            if last_user_message_idx != -1 and image_urls:
                self.debug.vision("Processed current message with images")

        all_tools = [
            {"id": tool.id, "description": getattr(tool.meta, "description", "")}
            for tool in Tools.get_tools()
        ]
        tool_ids = [tool["id"] for tool in all_tools]
        self.debug.data("Available tools", tool_ids)

        history_messages = messages[-6:]
        convo_snippet_parts = []
        for m in history_messages:
            content = _get_text_from_message(m.get("content", ""))
            role = m.get("role", "unknown").upper()
            if len(content) > self.valves.history_char_limit:
                content = content[: self.valves.history_char_limit] + "..."
            convo_snippet_parts.append(f"{role}: {content!r}")
        convo_snippet = "\n".join(convo_snippet_parts)

        # Determine web search tool based on valve setting
        if self.valves.use_exa_router_search and "exa_router_search" in tool_ids:
            web_search_tool_id = "exa_router_search"
        else:
            web_search_tool_id = "web_search"
        self.debug.data("Selected web search tool", web_search_tool_id)

        router_sys = TOOL_ROUTER_SYS_PROMPT_TEMPLATE.format(
            web_search_tool_id=web_search_tool_id
        )

        router_payload = {
            "model": self.valves.helper_model or body["model"],
            "messages": [
                {"role": "system", "content": router_sys},
                {
                    "role": "user",
                    "content": f"Recent Conversation History:\n{convo_snippet}\n\nLatest User Query (with image context if any):\n{routing_query}",
                },
            ],
            "stream": False,
        }

        self.debug.router(f"Sending routing query to model: {router_payload['model']}")
        self.debug.data("Routing query", routing_query, truncate=120)

        try:
            res = await generate_chat_completion(
                request=__request__, form_data=router_payload, user=user_obj
            )
            llm_response_text = res["choices"][0]["message"]["content"]
            self.debug.data("Router full response", llm_response_text, truncate=200)

            decision = "none"
            for line in llm_response_text.splitlines():
                if line.lower().strip().startswith("final answer:"):
                    decision = (
                        line.split(":", 1)[1]
                        .strip()
                        .lower()
                        .replace("'", "")
                        .replace('"', "")
                    )
                    break

            if decision == "none":
                last_line = llm_response_text.strip().splitlines()[-1].strip().lower()
                if last_line in tool_ids:
                    self.debug.router("Found tool ID on the last line as a fallback.")
                    decision = last_line

            self.debug.router(f"Extracted decision → {decision}")

        except Exception as exc:
            self.debug.error(f"Router error → {exc}")
            return body

        if decision == "none" and image_analysis_started:
            self.debug.flow(
                "No tool selected but image analysis was performed, completing"
            )
            await __event_emitter__(
                {"type": "status", "data": {"description": "", "done": True}}
            )
            return body

        # This is the main body that will be returned and used for the next turn's history.
        # We will create a separate, temporary body for the tool call.
        tool_body = body.copy()

        if decision != "none":
            if last_user_message_idx != -1:
                # Create a temporary copy of messages for the tool call
                tool_messages = [m.copy() for m in messages]

                # FIX: Instead of replacing with a string, create a valid text-only content structure.
                # This ensures the message format is always a list of parts, which is safe.
                tool_messages[last_user_message_idx]["content"] = [
                    {"type": "text", "text": routing_query}
                ]

                # Assign the safe, temporary messages to the tool_body
                tool_body["messages"] = tool_messages

        if decision in self.special_handlers:
            self.debug.handler(f"Activating special handler for '{decision}'")
            handler = self.special_handlers[decision]
            ctx = {"__event_emitter__": __event_emitter__}

            if decision == "image_generation":
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Developing creative concept...",
                            "done": False,
                        },
                    }
                )
                prompt, desc = await _generate_prompt_and_desc(
                    __request__,
                    user_obj,
                    router_payload["model"],
                    convo_snippet,
                    user_message_text,
                    self.debug,
                )
                ctx["prompt"] = prompt
                ctx["description"] = desc
            elif decision == "code_interpreter":
                # Pass the valve setting to determine which code interpreter to use
                return await handler(
                    __request__,
                    tool_body,
                    ctx,
                    user_obj,
                    self.debug,
                    self.valves.use_jupyter_code_interpreter,
                )
            elif decision == "web_search" and "full_image_context" in locals():
                # Pass image analysis to web search for better context
                ctx["image_context"] = locals()["full_image_context"]
                self.debug.handler("Passing image context to web search")

            # The handler receives the temporary tool_body, leaving the original `body` untouched.
            return await handler(__request__, tool_body, ctx, user_obj, self.debug)

        elif decision and decision != "none" and decision in tool_ids:
            self.debug.tool(f"Activating standard tool with ID → {decision}")
            # For standard tools, we modify the main body that gets passed on.
            # Special case for web_search: use handler if valve is set to default, otherwise use tool ID
            if decision == "web_search" and not self.valves.use_exa_router_search:
                # Use the special handler for default web search
                handler = self.special_handlers["web_search"]
                ctx = {"__event_emitter__": __event_emitter__}
                return await handler(__request__, tool_body, ctx, user_obj, self.debug)
            else:
                # Use the tool ID approach for exa_router_search or other tools
                body["tool_ids"] = [decision]
                if "messages" in tool_body:
                    body["messages"] = tool_body["messages"]

        # Return the original, unmodified body unless vision injection occurred.
        # The vision injection logic already correctly modifies the main `body`.
        self.debug.flow("AutoToolSelector processing completed")
        return body
