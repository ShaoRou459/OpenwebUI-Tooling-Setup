"""
Title: Auto Tool Selector
Description: A hybrid middleware that dynamically routes to all tools, applying special handling where needed.
author: ShaoRou459
author_url: https://github.com/ShaoRou459
Version: 1.2.5
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import copy
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from uuid import uuid4
from datetime import datetime
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
You are a Tool Routing Agent. Decide which single tool to use — or `none` — by considering BOTH:
- the user's CURRENT QUERY, and
- the RECENT CONVERSATION HISTORY (use history to infer intent, continuity, constraints, and previously provided details; do not re-answer here).

You must show brief reasoning, then output exactly one final line: `Final Answer: <tool_id>` or `Final Answer: none`.

---
### How to Use Context
- Resolve ambiguity in the current query using the history.
- Detect follow-ups (e.g., "do it again", "use the same method", "as before").
- Infer missing parameters from prior turns (e.g., topic, format, constraints).
- If history indicates a non-web creative/coding task, prefer those tools over web search.

---
### Pick a Tool

1) `{web_search_tool_id}`
- Use when info must be fetched from the web: live/unknown facts, specific URL given (crawl it), multi-source research/comparisons, or temporal queries (latest/recent/current year).
- Do NOT use for knowledge that AI can answer without web or creative tasks answerable without web.
- SEARCH LEVEL (append after tool id based on query complexity):
  - CRAWL → user provided a specific URL to read
  - STANDARD → single-topic lookup, simple facts, quick answers
  - COMPLETE → complex multi-part questions, comparisons, research requiring synthesis of diverse sources, comprehensive how‑tos

2) `image_generation`
- Use for explicit requests to create/design an image or logo.

3) `code_interpreter`
- Use only to execute Python or manipulate files inside the notebook env. Not for pure code generation or non-Python tasks.

4) `memory`
- Use to remember/recall simple facts within this conversation. Not for file ops or complex data recall.

---
### Smart Search Level Selection
- STANDARD: "What is X?", "When did Y happen?", "Who is Z?", simple factual queries
- COMPLETE: "Compare A vs B", "How to do X comprehensively", "Explain the full process of Y", "Research Z and provide detailed analysis", multi‑faceted questions, technical tutorials

---
### Core Rules
- Latest user message triggers the decision; HISTORY MODIFIES/INFORMS the decision.
- When in doubt, choose `none` and answer directly.
- Be intelligent about search depth — complex questions deserve COMPLETE research.
- Use common sense and err toward deeper research for substantive queries.

---
### Output Format (strict)
<think>
- Query complexity (simple/complex)
- Need (≤8 words)
- Best approach using context (≤8 words)
</think>
Final Answer: <tool_id or none>
If `{web_search_tool_id}` is chosen, you MUST append a LEVEL token: CRAWL | STANDARD | COMPLETE.
Example: `Final Answer: {web_search_tool_id} COMPLETE`
ONLY return the Final Answer line exactly as shown (no quotes/formatting).
"""


# ─── Enhanced Debug System ──────────────────────────────────────────────────
from dataclasses import dataclass, field
from contextlib import contextmanager

@dataclass
class DebugMetrics:
    """Collects and tracks metrics throughout the debug session."""
    
    # Timing metrics
    start_time: float = field(default_factory=time.perf_counter)
    operation_times: Dict[str, float] = field(default_factory=dict)
    total_operations: int = 0
    
    # Tool routing metrics
    tool_decisions: int = 0
    tool_activations: int = 0
    handler_calls: int = 0
    
    # Vision processing metrics
    images_processed: int = 0
    vision_calls: int = 0
    vision_total_time: float = 0.0
    
    # LLM metrics
    llm_calls: int = 0
    llm_total_time: float = 0.0
    llm_failures: int = 0
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_operation_time(self, operation: str, duration: float) -> None:
        """Add timing data for an operation."""
        self.operation_times[operation] = self.operation_times.get(operation, 0) + duration
        self.total_operations += 1
    
    def add_error(self, error: str) -> None:
        """Add an error to tracking."""
        self.errors.append(f"[{datetime.now().strftime('%H:%M:%S')}] {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to tracking."""
        self.warnings.append(f"[{datetime.now().strftime('%H:%M:%S')}] {warning}")
    
    def get_total_time(self) -> float:
        """Get total elapsed time since start."""
        return time.perf_counter() - self.start_time


class Debug:
    """Enhanced structured debug logging system for AutoToolSelector with metrics collection."""

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
        "WHITE": "\x1b[97m",
        "ORANGE": "\x1b[38;5;208m",
        "PURPLE": "\x1b[38;5;129m",
    }

    def __init__(self, enabled: bool = False, tool_name: str = "AutoToolSelector"):
        self.enabled = enabled
        self.tool_name = tool_name
        self.metrics = DebugMetrics()
        self._session_id = str(int(time.time()))[-6:]  # Last 6 digits of timestamp

    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds

    def _format_msg(self, category: str, message: str, color: str = "CYAN", include_timestamp: bool = True) -> str:
        """Format a debug message with consistent styling and optional timestamp."""
        if not self.enabled:
            return ""

        timestamp = f"{self._COLORS['DIM']}[{self._get_timestamp()}]{self._COLORS['RESET']} " if include_timestamp else ""
        prefix = f"{self._COLORS['MAGENTA']}{self._COLORS['BOLD']}[{self.tool_name}:{self._session_id}]{self._COLORS['RESET']}"
        cat_colored = f"{self._COLORS[color]}{self._COLORS['BOLD']}{category:<12}{self._COLORS['RESET']}"
        msg_colored = f"{self._COLORS[color]}{message}{self._COLORS['RESET']}"

        return f"{timestamp}{prefix} {cat_colored}: {msg_colored}"

    def _log(self, category: str, message: str, color: str = "CYAN", track_metric: bool = True) -> None:
        """Internal logging method with optional metrics tracking."""
        if self.enabled:
            formatted = self._format_msg(category, message, color)
            if formatted:
                print(formatted, file=sys.stderr)
            
            if track_metric:
                self.metrics.total_operations += 1

    @contextmanager
    def timer(self, operation_name: str):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.metrics.add_operation_time(operation_name, duration)
            if self.enabled:
                self._log("TIMING", f"{operation_name} completed in {duration:.3f}s", "ORANGE", track_metric=False)

    def start_session(self, description: str = "") -> None:
        """Start a new debug session."""
        self.metrics = DebugMetrics()  # Reset metrics
        session_msg = f"Debug session started" + (f": {description}" if description else "")
        self._log("SESSION", session_msg, "PURPLE", track_metric=False)
        self._log("SESSION", f"Session ID: {self._session_id}", "DIM", track_metric=False)

    def router(self, message: str) -> None:
        """Log router decision making."""
        self._log("ROUTER", message, "BLUE")
        self.metrics.tool_decisions += 1

    def vision(self, message: str) -> None:
        """Log vision processing."""
        self._log("VISION", message, "GREEN")
        self.metrics.vision_calls += 1

    def tool(self, message: str) -> None:
        """Log tool activation."""
        self._log("TOOL", message, "YELLOW")
        self.metrics.tool_activations += 1

    def handler(self, message: str) -> None:
        """Log special handler activity."""
        self._log("HANDLER", message, "MAGENTA")
        self.metrics.handler_calls += 1

    def error(self, message: str) -> None:
        """Log errors and warnings."""
        self._log("ERROR", message, "RED")
        self.metrics.add_error(message)

    def warning(self, message: str) -> None:
        """Log warnings."""
        self._log("WARNING", message, "YELLOW")
        self.metrics.add_warning(message)

    def flow(self, message: str) -> None:
        """Log general workflow steps."""
        self._log("FLOW", message, "CYAN")

    def data(self, label: str, data: Any, truncate: Optional[int] = None) -> None:
        """Log data with optional truncation. Set truncate=None to disable."""
        if not self.enabled:
            return
        data_str = str(data)
        if truncate is not None and isinstance(data, str) and len(data_str) > truncate:
            data_str = f"{data_str[:truncate]}..."
        self._log("DATA", f"{label} → {data_str}", "DIM")

    def llm_call(self, model: str, success: bool = True, duration: float = 0.0) -> None:
        """Track LLM call metrics."""
        self.metrics.llm_calls += 1
        self.metrics.llm_total_time += duration
        if not success:
            self.metrics.llm_failures += 1
        
        status = "✓" if success else "✗"
        self._log("LLM", f"{status} {model} ({duration:.3f}s)", "GREEN" if success else "RED")

    def vision_metrics(self, images: int = 0, duration: float = 0.0) -> None:
        """Update vision-related metrics."""
        self.metrics.images_processed += images
        self.metrics.vision_total_time += duration

    def metrics_summary(self) -> None:
        """Display comprehensive metrics summary at the end of execution."""
        if not self.enabled:
            return
        
        total_time = self.metrics.get_total_time()
        
        # Build metrics report
        report_lines = [
            "",
            "═" * 80,
            f"📊 EXECUTION METRICS SUMMARY - {self.tool_name} (Session: {self._session_id})",
            "═" * 80,
            "",
            "⏱️  TIMING METRICS:",
            f"   Total Execution Time: {total_time:.3f}s",
            f"   Total Operations: {self.metrics.total_operations}",
        ]
        
        if self.metrics.operation_times:
            report_lines.append("   Operation Breakdown:")
            for op, duration in sorted(self.metrics.operation_times.items(), key=lambda x: x[1], reverse=True):
                percentage = (duration / total_time) * 100 if total_time > 0 else 0
                report_lines.append(f"     • {op}: {duration:.3f}s ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "🔧 TOOL ROUTING METRICS:",
            f"   Tool Decisions Made: {self.metrics.tool_decisions}",
            f"   Tool Activations: {self.metrics.tool_activations}",
            f"   Handler Calls: {self.metrics.handler_calls}",
        ])
        
        if self.metrics.vision_calls > 0:
            report_lines.extend([
                "",
                "👁️  VISION METRICS:",
                f"   Vision Calls: {self.metrics.vision_calls}",
                f"   Images Processed: {self.metrics.images_processed}",
                f"   Vision Total Time: {self.metrics.vision_total_time:.3f}s",
                f"   Average Vision Time: {(self.metrics.vision_total_time / self.metrics.vision_calls):.3f}s" if self.metrics.vision_calls > 0 else "   Average Vision Time: N/A",
            ])
        
        if self.metrics.llm_calls > 0:
            report_lines.extend([
                "",
                "🤖 LLM METRICS:",
                f"   Total LLM Calls: {self.metrics.llm_calls}",
                f"   LLM Total Time: {self.metrics.llm_total_time:.3f}s",
                f"   LLM Failures: {self.metrics.llm_failures}",
                f"   Average LLM Time: {(self.metrics.llm_total_time / self.metrics.llm_calls):.3f}s" if self.metrics.llm_calls > 0 else "   Average LLM Time: N/A",
            ])
        
        if self.metrics.errors or self.metrics.warnings:
            report_lines.extend([
                "",
                "⚠️  ISSUES SUMMARY:",
                f"   Errors: {len(self.metrics.errors)}",
                f"   Warnings: {len(self.metrics.warnings)}",
            ])
            
            if self.metrics.errors:
                report_lines.append("   Recent Errors:")
                for error in self.metrics.errors[-3:]:  # Show last 3 errors
                    report_lines.append(f"     • {error}")
            
            if self.metrics.warnings:
                report_lines.append("   Recent Warnings:")
                for warning in self.metrics.warnings[-3:]:  # Show last 3 warnings
                    report_lines.append(f"     • {warning}")
        
        report_lines.extend([
            "",
            "═" * 80,
            ""
        ])
        
        # Print the metrics report
        metrics_report = "\n".join(report_lines)
        formatted = self._format_msg("METRICS", metrics_report, "PURPLE", include_timestamp=False)
        if formatted:
            print(formatted, file=sys.stderr)


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

    start_time = time.perf_counter()
    try:
        res = await generate_chat_completion(
            request=request, form_data=payload, user=user
        )
        duration = time.perf_counter() - start_time
        
        if debug:
            debug.llm_call(model, success=True, duration=duration)
        
        obj = _parse_json_fuzzy(res["choices"][0]["message"]["content"], debug)
        prompt = obj.get("prompt", user_query)
        description = obj.get("description", "Image generated from conversation.")
        if debug:
            debug.handler(f"Router prompt → {prompt[:60]}… | desc: {description}")
        return prompt, description
    except Exception as exc:
        duration = time.perf_counter() - start_time
        if debug:
            debug.llm_call(model, success=False, duration=duration)
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


async def default_web_search_handler(
    request: Request, body: dict, ctx: dict, user: Any, debug: Debug = None
) -> dict:
    """
    Thin wrapper around OpenWebUI's chat_web_search_handler that normalizes the
    call signature from (request, body, ctx, user[, debug]) to the expected
    (request, body, __event_emitter__, user).

    This prevents signature mismatches when other handlers accept an optional
    debug parameter while the default web search handler does not.
    """
    if debug:
        debug.handler("Routing to OpenWebUI default web_search handler")

    # Delegate to the OpenWebUI middleware handler with the correct parameters
    extra_params = ctx if isinstance(ctx, dict) else {}
    return await chat_web_search_handler(request, body, extra_params, user)

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
    # Clear status immediately; the main model will proceed to respond
    if emitter:
        try:
            await emitter(
                {
                    "type": "status",
                    "data": {"description": "", "done": True},
                }
            )
        except Exception:
            pass
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
    # Clear status immediately; the main model will proceed to respond
    if emitter:
        try:
            await emitter(
                {
                    "type": "status",
                    "data": {"description": "", "done": True},
                }
            )
        except Exception:
            pass
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
            # Use a local wrapper to normalize signature and prevent arg mismatches
            "web_search": default_web_search_handler,
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
        
        if self.debug.enabled:
            self.debug.start_session(f"User message: {user_message_text[:50]}...")
        self.debug.flow("Starting AutoToolSelector processing")

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

            start_vision = time.perf_counter()

            async def describe_image(idx: int, url: str) -> str:
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
                    return f"[Image {idx + 1} Context: {description}]"
                except Exception as e:
                    self.debug.error(f"Vision model failed for image {idx + 1}: {e}.")
                    return f"[Image {idx + 1} Context: Analysis failed.]"

            # Limit concurrency to avoid overwhelming the vision model/backend
            sem = asyncio.Semaphore(3)

            async def sem_wrapper(i: int, u: str) -> str:
                async with sem:
                    self.debug.vision(f"Analyzing image {i + 1}/{len(image_urls)}...")
                    return await describe_image(i, u)

            image_descriptions = await asyncio.gather(
                *[sem_wrapper(i, u) for i, u in enumerate(image_urls)]
            )

            elapsed_vision = time.perf_counter() - start_vision
            self.debug.flow(f"Vision analysis completed in {elapsed_vision:.2f}s for {len(image_urls)} image(s)")

            # Clear the vision analysis status now that it's complete
            try:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "", "done": True},
                })
            except Exception:
                pass

            if image_descriptions:
                full_image_context = "\n\n".join(image_descriptions)
                # This query is temporary for the router model only
                routing_query = f"{user_message_text}\n\n{full_image_context}"
                
                if self.debug.enabled:
                    self.debug.vision_metrics(images=len(image_urls), duration=elapsed_vision)

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
            start_router = time.perf_counter()
            res = await generate_chat_completion(
                request=__request__, form_data=router_payload, user=user_obj
            )
            elapsed_router = time.perf_counter() - start_router
            self.debug.llm_call(router_payload['model'], success=True, duration=elapsed_router)
            llm_response_text = res["choices"][0]["message"]["content"]
            self.debug.data("Router full response", llm_response_text, truncate=200)

            decision = "none"
            decision_tool = "none"
            decision_mode = None
            for line in llm_response_text.splitlines():
                if line.lower().strip().startswith("final answer:"):
                    raw = (
                        line.split(":", 1)[1]
                        .strip()
                        .replace("'", "")
                        .replace('"', "")
                    )
                    # Allow optional mode token after tool id
                    parts = raw.split()
                    if parts:
                        decision_tool = parts[0].lower()
                        decision = decision_tool
                        if len(parts) > 1:
                            maybe_mode = parts[-1].upper()
                            if maybe_mode in {"CRAWL", "STANDARD", "COMPLETE"}:
                                decision_mode = maybe_mode
                    break

            if decision == "none":
                last_line = llm_response_text.strip().splitlines()[-1].strip().lower()
                if last_line in tool_ids:
                    self.debug.router("Found tool ID on the last line as a fallback.")
                    decision = last_line
                    decision_tool = last_line

            if decision_mode:
                self.debug.router(
                    f"Extracted decision → {decision} with mode {decision_mode}"
                )
            else:
                self.debug.router(f"Extracted decision → {decision}")

        except Exception as exc:
            elapsed_router = time.perf_counter() - start_router if 'start_router' in locals() else 0
            self.debug.llm_call(router_payload.get('model', 'unknown'), success=False, duration=elapsed_router)
            self.debug.error(f"Router error → {exc}")
            if self.debug.enabled:
                self.debug.metrics_summary()
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
                # Create a deep copy of messages for the tool call to avoid side-effects
                tool_messages = copy.deepcopy(messages)

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
            # Special case: default web_search handler only takes 4 parameters
            if decision == "web_search":
                self.debug.handler("Calling default web_search handler with 4 parameters")
                return await handler(__request__, tool_body, ctx, user_obj)
            else:
                self.debug.handler(f"Calling {decision} handler with 5 parameters (including debug)")
                return await handler(__request__, tool_body, ctx, user_obj, self.debug)

        elif decision and decision != "none" and decision in tool_ids:
            self.debug.tool(f"Activating standard tool with ID → {decision}")
            # For standard tools, we modify the main body that gets passed on.
            # Special case for web_search: use handler if valve is set to default, otherwise use tool ID
            if decision == "web_search" and not self.valves.use_exa_router_search:
                # Use the special handler for default web search
                self.debug.handler("Using default web_search handler via tool_ids path")
                handler = self.special_handlers["web_search"]
                ctx = {"__event_emitter__": __event_emitter__}
                # Default chat_web_search_handler only takes 4 parameters (no debug)
                return await handler(__request__, tool_body, ctx, user_obj)
            else:
                # Use the tool ID approach for exa_router_search or other tools
                # Inject per-call mode for exa_router_search so the tool can skip its own router
                if decision == "exa_router_search" and decision_mode:
                    # Ensure tool_body has messages and append a system control message
                    tool_body.setdefault("messages", messages)
                    tool_body["messages"].append(
                        {"role": "system", "content": f"[EXA_SEARCH_MODE] {decision_mode}"}
                    )
                body["tool_ids"] = [decision]
                if "messages" in tool_body:
                    body["messages"] = tool_body["messages"]

        # Return the original, unmodified body unless vision injection occurred.
        # The vision injection logic already correctly modifies the main `body`.
        self.debug.flow("AutoToolSelector processing completed")
        if self.debug.enabled:
            self.debug.metrics_summary()
        # Safety: ensure any transient status is cleared before handing back to model
        try:
            await __event_emitter__({
                "type": "status",
                "data": {"description": "", "done": True},
            })
        except Exception:
            pass
        return body
