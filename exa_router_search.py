"""
Title: Search Router Tool
Description: An advanced research tool with a robust retry and graceful failure mechanism.
author: ShaoRou459
author_url: https://github.com/ShaoRou459
Version: 1.2.5
Requirements: exa_py, open_webui
"""

from __future__ import annotations

import os
import re
import sys
import json
import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from open_webui.utils.chat import generate_chat_completion
from open_webui.models.users import Users
from open_webui.utils.misc import get_last_user_message

try:
    from exa_py import Exa

    EXA_AVAILABLE = True
except ImportError:
    Exa = None
    EXA_AVAILABLE = False


# ─── System Prompts ───────────────────────────────────────────────────────────

SEARCH_STRATEGY_ROUTER_PROMPT_TEMPLATE = """
You are a search strategy router. Analyze the user's query and conversation context to determine the best search approach.

Strategies:
- CRAWL → only if the user provided a specific URL to read.
- STANDARD → default; quick lookup answerable with a brief web search (~5 min).
- COMPLETE → deep, multi-source research or explicit request for in-depth analysis.

Respond with your decision on a line starting with "ANSWER: " followed by CRAWL, STANDARD, or COMPLETE.
"""

IMPROVED_SQR_PROMPT_TEMPLATE = f"""
You are a search query optimizer. Convert the user's question into a natural, effective search query that would work well on Google or similar search engines.

Guidelines:
- Write like a human would search Google - natural phrases, not keyword soup
- For news/current events: use "latest", "today", "{datetime.now().year}", "recent" naturally
- For technical topics: include key terms but keep it readable
- For comparisons: use "vs" or "compared to" 
- For how-to: start with "how to" or include "guide", "tutorial"
- Keep it under 10 words when possible
- Avoid excessive OR operators and site: filters unless truly needed

Examples:
- User: "latest news about AI" → "latest AI news {datetime.now().year}"
- User: "how do I fix my car engine" → "how to fix car engine problems"
- User: "compare React vs Vue" → "React vs Vue comparison {datetime.now().year}"
- User: "what is quantum computing" → "quantum computing explained"

Output your optimized search query on a line starting with "ANSWER: "
"""

QUICK_SUMMARIZER_PROMPT = (
    "Using ONLY the provided context, produce a clear, organized summary that answers the user's request. "
    "Do NOT include explicit citations, reference markers, or raw URLs unless the user explicitly asked for citations."
)

# ── New Iterative Research System Prompts ────────────────────────────────────

INTRODUCTORY_QUERY_PROMPT = """
You are an information-seeking specialist. Your job is to generate an introductory search query that helps understand the context and background of what the user is asking about.

CURRENT DATE: {current_date}

This query should be INFORMATIONAL, not trying to answer their question directly. Think of it as "What do I need to know about this topic first?"

Examples:
- User: "How do I optimize my React app performance?" → "React application performance optimization techniques"
- User: "What's the latest news about OpenAI?" → "OpenAI company recent developments {current_year}"
- User: "Compare investment strategies for 2024" → "investment strategies types overview {current_year}"

Output your introductory query on a line starting with "QUERY: "
"""

OBJECTIVE_SETTING_PROMPT = """
You are a research strategist. Based on the user's request and the introductory information gathered, set clear research objectives.

CURRENT DATE: {current_date}

Analyze:
1. What exactly is the user asking for?
2. What are the key components of their request?
3. What direction should the research take?

Output a structured analysis with:
OBJECTIVES: [List 3-5 specific research objectives]
RESEARCH_DIRECTION: [Brief description of the overall research approach]
KEY_COMPONENTS: [List the main parts of the user's request that need to be addressed]
"""

ITERATION_REASONING_PROMPT = """
You are a research iteration planner. Based on your current knowledge and what you've found so far, reason about what to search for next.

CURRENT DATE: {current_date}

Current situation:
- Research objectives: {objectives}
- Previous findings summary: {previous_findings}
- Iteration: {current_iteration} of {max_iterations}

Your task:
1. Analyze what you've learned so far
2. Identify what's still missing
3. Reason about the best search approach for this iteration
4. Generate {query_count} diverse, specific search queries

Note: For time-sensitive topics, include {current_year} in your queries when relevant.

Output format:
ANALYSIS: [What you've learned and what's missing]
REASONING: [Why these specific searches will help]
QUERIES: ["query1", "query2", "query3", ...]
"""

ITERATION_CONCLUSION_PROMPT = """
You are a research analyst. Summarize what you found in this iteration and determine next steps.

CURRENT DATE: {current_date}
ITERATION: {current_iteration} of {max_iterations}

Provide:
FINDINGS_SUMMARY: [Key information discovered this iteration - be concise but comprehensive]
PROGRESS_ASSESSMENT: [How much closer are you to answering the user's question?]
NEXT_STEPS: [What should the next iteration focus on, or should research conclude?]
DECISION: [CONTINUE or FINISH]

Note: If this is iteration {max_iterations}, you must decide FINISH unless critical information is still missing.
"""

FINAL_SYNTHESIS_PROMPT = """
You are an information organizer. Your job is to structure the research findings so the chat model can effectively answer the user's question.

CURRENT DATE: {current_date}

Using the research chain and findings summaries, organize the information into a clear, comprehensive knowledge base that covers:
- Key facts and findings relevant to the user's question
- Important context and background information  
- Relevant developments, especially recent ones when applicable
- Different perspectives or approaches discovered
- Any actionable insights or recommendations found

Structure this as organized, factual information that provides the chat model with everything needed to give a complete response to the user's original question. Focus on being comprehensive and well-organized rather than directly answering.

Do include raw URLs or direct quotes from sources when needed.
"""

SYNTHESIS_DECIDER_PROMPT = """
You are an output formatting assistant. Decide whether to:
- RETURN_RAW → when the user wants to read/learn/review full context (docs, articles, longer materials).
- SYNTHESIZE → when the user asks for an answer/summary/comparison/explanation.
"""

COMPLETE_SUMMARIZER_PROMPT = """
You are an expert synthesizer. Provide a comprehensive, well-structured answer to the user's original question using the provided research context and agent notes.
- Choose the best format (brief summary, step-by-step, comparison, etc.) based on the question.
- Use notes to guide emphasis; pull specifics from context.
Important: Do NOT include explicit citations, reference markers, or raw URLs unless the user explicitly asked for citations.
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
    
    # API/LLM metrics
    llm_calls: int = 0
    llm_total_time: float = 0.0
    llm_failures: int = 0
    
    # Search metrics
    search_queries: int = 0
    urls_found: int = 0
    urls_crawled: int = 0
    urls_successful: int = 0
    urls_failed: int = 0
    
    # Content metrics
    total_content_chars: int = 0
    context_truncations: int = 0
    
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
    """Enhanced structured debug logging system for SearchRouterTool with metrics collection."""

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

    def __init__(self, enabled: bool = False, tool_name: str = "SearchRouterTool"):
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
        """Log search strategy routing decisions."""
        self._log("ROUTER", message, "BLUE")

    def search(self, message: str) -> None:
        """Log search operations."""
        self._log("SEARCH", message, "GREEN")
        self.metrics.search_queries += 1

    def crawl(self, message: str) -> None:
        """Log crawling operations."""
        self._log("CRAWL", message, "YELLOW")

    def agent(self, message: str) -> None:
        """Log agentic operations in COMPLETE mode."""
        self._log("AGENT", message, "MAGENTA")

    def synthesis(self, message: str) -> None:
        """Log synthesis and summarization."""
        self._log("SYNTHESIS", message, "WHITE")

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

    def data(self, label: str, data: Any, truncate: int = 80) -> None:
        """Log data with optional truncation."""
        if not self.enabled:
            return

        if isinstance(data, str) and len(data) > truncate:
            data_str = f"{data[:truncate]}..."
            self.metrics.context_truncations += 1
        else:
            data_str = str(data)

        self._log("DATA", f"{label} → {data_str}", "DIM")

    def query(self, message: str) -> None:
        """Log query refinement operations."""
        self._log("QUERY", message, "BLUE")

    def iteration(self, message: str) -> None:
        """Log research iterations in COMPLETE mode."""
        self._log("ITERATION", message, "CYAN")

    def llm_call(self, model: str, success: bool = True, duration: float = 0.0) -> None:
        """Track LLM call metrics."""
        self.metrics.llm_calls += 1
        self.metrics.llm_total_time += duration
        if not success:
            self.metrics.llm_failures += 1
        
        status = "✓" if success else "✗"
        self._log("LLM", f"{status} {model} ({duration:.3f}s)", "GREEN" if success else "RED")

    def url_metrics(self, found: int = 0, crawled: int = 0, successful: int = 0, failed: int = 0) -> None:
        """Update URL-related metrics."""
        self.metrics.urls_found += found
        self.metrics.urls_crawled += crawled
        self.metrics.urls_successful += successful
        self.metrics.urls_failed += failed

    def content_metrics(self, chars: int, truncated: bool = False) -> None:
        """Update content-related metrics."""
        self.metrics.total_content_chars += chars
        if truncated:
            self.metrics.context_truncations += 1

    def report(self, message: str) -> None:
        """Log full debug reports without truncation."""
        if self.enabled:
            # Split long reports into chunks to avoid terminal truncation
            lines = message.split("\n")
            chunk_size = 25  # Lines per chunk

            for i in range(0, len(lines), chunk_size):
                chunk = lines[i : i + chunk_size]
                chunk_text = "\n".join(chunk)

                if i == 0:
                    # First chunk gets the REPORT prefix
                    formatted = self._format_msg(
                        "REPORT", f"Part {(i//chunk_size)+1}:\n{chunk_text}", "WHITE", include_timestamp=False
                    )
                else:
                    # Subsequent chunks get REPORT-CONT prefix
                    formatted = self._format_msg(
                        "REPORT-CONT",
                        f"Part {(i//chunk_size)+1}:\n{chunk_text}",
                        "WHITE",
                        include_timestamp=False
                    )

                if formatted:
                    print(formatted, file=sys.stderr)

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
            "🤖 LLM METRICS:",
            f"   Total LLM Calls: {self.metrics.llm_calls}",
            f"   LLM Total Time: {self.metrics.llm_total_time:.3f}s",
            f"   LLM Failures: {self.metrics.llm_failures}",
            f"   Average LLM Time: {(self.metrics.llm_total_time / self.metrics.llm_calls):.3f}s" if self.metrics.llm_calls > 0 else "   Average LLM Time: N/A",
        ])
        
        if self.metrics.search_queries > 0:
            report_lines.extend([
                "",
                "🔍 SEARCH METRICS:",
                f"   Search Queries: {self.metrics.search_queries}",
                f"   URLs Found: {self.metrics.urls_found}",
                f"   URLs Crawled: {self.metrics.urls_crawled}",
                f"   URLs Successful: {self.metrics.urls_successful}",
                f"   URLs Failed: {self.metrics.urls_failed}",
                f"   Success Rate: {(self.metrics.urls_successful / self.metrics.urls_crawled * 100):.1f}%" if self.metrics.urls_crawled > 0 else "   Success Rate: N/A",
            ])
        
        if self.metrics.total_content_chars > 0:
            report_lines.extend([
                "",
                "📄 CONTENT METRICS:",
                f"   Total Content Characters: {self.metrics.total_content_chars:,}",
                f"   Context Truncations: {self.metrics.context_truncations}",
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
        f"{Debug._COLORS['MAGENTA']}{Debug._COLORS['BOLD']}[SearchRouterTool]{Debug._COLORS['RESET']}{Debug._COLORS['CYAN']} {msg}{Debug._COLORS['RESET']}",
        file=sys.stderr,
    )


# ─── Constants & Helpers ──────────────────────────────────────────────────────
URL_RE = re.compile(r"https?://\S+")


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


async def generate_with_retry(
    max_retries: int = 3, delay: int = 3, debug: Debug = None, **kwargs: Any
) -> Dict[str, Any]:
    """
    A wrapper for generate_chat_completion that includes a retry mechanism with exponential backoff.
    """
    def _normalize_llm_response(res: Any) -> Dict[str, Any]:
        # Already a dict
        if isinstance(res, dict):
            return res
        
        # Debug what we actually got
        if debug:
            debug.error(f"LLM response normalization needed. Type: {type(res)}, Dir: {[attr for attr in dir(res) if not attr.startswith('_')]}")
        
        # If it's a JSONResponse (Starlette/FastAPI), extract the body and parse
        try:
            if hasattr(res, 'body') and hasattr(res, 'status_code'):
                import json
                body_bytes = res.body
                if isinstance(body_bytes, bytes):
                    parsed = json.loads(body_bytes.decode('utf-8'))
                    if debug:
                        debug.flow(f"Successfully parsed JSONResponse body")
                        debug.data("Full parsed response", str(parsed)[:500] + "..." if len(str(parsed)) > 500 else str(parsed))
                        debug.data("Response keys", list(parsed.keys()) if isinstance(parsed, dict) else "Not a dict")
                    return parsed
                elif isinstance(body_bytes, str):
                    parsed = json.loads(body_bytes)
                    if debug:
                        debug.flow(f"Successfully parsed JSONResponse string body")
                        debug.data("Full parsed response", str(parsed)[:500] + "..." if len(str(parsed)) > 500 else str(parsed))
                        debug.data("Response keys", list(parsed.keys()) if isinstance(parsed, dict) else "Not a dict")
                    return parsed
        except Exception as e:
            if debug:
                debug.error(f"Failed to parse JSONResponse body: {e}")
        
        # Try to call render() method if available (for Response objects)
        try:
            if hasattr(res, 'render'):
                import json
                rendered = res.render(None)  # Pass None as scope if not needed
                if isinstance(rendered, bytes):
                    parsed = json.loads(rendered.decode('utf-8'))
                    if debug:
                        debug.flow(f"Successfully parsed rendered response")
                    return parsed
        except Exception as e:
            if debug:
                debug.error(f"Failed to render and parse response: {e}")
        
        # Check if it's already JSON-like but not a dict
        try:
            if hasattr(res, '__dict__'):
                return res.__dict__
        except Exception:
            pass
            
        # Last resort: if it has a dict-like interface
        try:
            return dict(res)  # may raise
        except Exception as e:
            if debug:
                debug.error(f"Failed to convert to dict: {e}")
            raise TypeError(f"LLM response is not a dict and could not be normalized. Type: {type(res)}, Value: {str(res)[:200]}")

    model_name = kwargs.get('form_data', {}).get('model', 'unknown')
    last_exception = None
    
    for attempt in range(max_retries):
        start_time = time.perf_counter()
        try:
            raw = await generate_chat_completion(**kwargs)
            result = _normalize_llm_response(raw)
            duration = time.perf_counter() - start_time
            
            if debug:
                debug.llm_call(model_name, success=True, duration=duration)
                if attempt > 0:
                    debug.flow(f"LLM call succeeded on attempt {attempt + 1}")
            return result
        except Exception as e:
            duration = time.perf_counter() - start_time
            last_exception = e
            
            if debug:
                debug.llm_call(model_name, success=False, duration=duration)
                debug.error(
                    f"LLM call failed on attempt {attempt + 1}/{max_retries}: {str(e)[:100]}..."
                )

            # Don't wait on the last attempt
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = delay * (2**attempt) + (attempt * 0.5)
                await asyncio.sleep(wait_time)

    if debug:
        debug.error(
            f"LLM call failed after {max_retries} retries. Last error: {str(last_exception)[:100]}..."
        )
    raise last_exception


async def generate_with_parsing_retry(
    max_retries: int = 3, delay: int = 3, debug: Debug = None, 
    expected_keys: List[str] = None, **kwargs: Any
) -> Dict[str, Any]:
    """
    A wrapper that combines generate_with_retry with parsing retry logic.
    Retries both API failures and response parsing failures.
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            # First, try the API call with retry
            result = await generate_with_retry(max_retries=max_retries, delay=delay, debug=debug, **kwargs)
            
            # Then validate the response format
            if expected_keys:
                if isinstance(result, dict):
                    # Check if any expected key exists
                    if any(key in result for key in expected_keys):
                        return result
                    else:
                        # Log the parsing issue and retry
                        if debug:
                            debug.warning(f"Response missing expected keys {expected_keys}. Got keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}. Attempt {attempt + 1}/{max_retries}")
                        raise ValueError(f"Response missing expected keys {expected_keys}. Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                else:
                    if debug:
                        debug.warning(f"Response is not a dict. Type: {type(result)}. Attempt {attempt + 1}/{max_retries}")
                    raise ValueError(f"Response is not a dict. Type: {type(result)}")
            else:
                # No specific validation needed, return result
                return result
                
        except Exception as e:
            last_exception = e
            
            if debug:
                debug.error(f"Generate with parsing retry failed on attempt {attempt + 1}/{max_retries}: {str(e)[:100]}...")
            
            # Don't wait on the last attempt
            if attempt < max_retries - 1:
                # Use same exponential backoff as generate_with_retry
                wait_time = delay * (2**attempt) + (attempt * 0.5)
                if debug:
                    debug.flow(f"Waiting {wait_time:.1f}s before retry attempt {attempt + 2}")
                await asyncio.sleep(wait_time)
    
    if debug:
        debug.error(f"Generate with parsing retry failed after {max_retries} retries. Last error: {str(last_exception)[:100]}...")
    raise last_exception


# Debug Report Dataclasses


@dataclass
class QuickDebugReport:
    """Enhanced structured report for debugging the STANDARD search process."""

    initial_query: str = ""
    refined_query: str = ""
    urls_found: List[str] = field(default_factory=list)
    urls_crawled: List[str] = field(default_factory=list)
    urls_successful: List[str] = field(default_factory=list)
    urls_failed: List[str] = field(default_factory=list)
    final_prompt: str = ""
    final_output: str = ""

    # Valve settings for comparison
    valve_urls_to_search: int = 0
    valve_queries_to_crawl: int = 0
    valve_max_context_chars: int = 0

    # Actual metrics
    context_length: int = 0
    was_truncated: bool = False

    def format_report(self) -> str:
        report_parts = [
            "\n\n" + "=" * 30 + " STANDARD SEARCH DEBUG REPORT " + "=" * 30,
            f"INITIAL USER QUERY: {self.initial_query}",
            f"REFINED SEARCH QUERY: {self.refined_query}",
            "",
            "─── VALVE SETTINGS vs ACTUAL RESULTS ───",
            f"URLs to Search (valve): {self.valve_urls_to_search} | Found: {len(self.urls_found)}",
            f"URLs to Crawl (valve): {self.valve_queries_to_crawl} | Attempted: {len(self.urls_crawled)} | Successful: {len(self.urls_successful)}",
            f"Max Context Chars (valve): {self.valve_max_context_chars} | Used: {self.context_length} {'(TRUNCATED)' if self.was_truncated else ''}",
            "",
            "─── SEARCH RESULTS ───",
            f"URLs Found ({len(self.urls_found)}):",
        ]

        for i, url in enumerate(self.urls_found, 1):
            report_parts.append(f"  {i}. {url}")

        report_parts.extend(
            [
                "",
                "─── CRAWL RESULTS ───",
                f"Successfully Crawled ({len(self.urls_successful)}):",
            ]
        )

        for i, url in enumerate(self.urls_successful, 1):
            report_parts.append(f"  ✓ {i}. {url}")

        if self.urls_failed:
            report_parts.extend(
                [
                    f"Failed to Crawl ({len(self.urls_failed)}):",
                ]
            )
            for i, url in enumerate(self.urls_failed, 1):
                report_parts.append(f"  ✗ {i}. {url}")

        report_parts.extend(
            [
                "",
                "─── SYNTHESIS ───",
                f"Final Output Preview:\n{self.final_output[:400]}{'...' if len(self.final_output) > 400 else ''}",
                "",
                "=" * 91 + "\n",
            ]
        )
        return "\n".join(report_parts)


@dataclass
class CompleteDebugReport:
    """Enhanced structured report for debugging the COMPLETE search process."""

    initial_user_query: str = ""
    refined_initial_query: str = ""
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    final_decision: str = ""
    final_payload: str = ""
    final_output: str = ""

    # Valve settings for comparison
    valve_urls_per_query: int = 0
    valve_queries_to_crawl: int = 0
    valve_queries_to_generate: int = 0
    valve_max_iterations: int = 0

    # Total metrics
    total_sources_found: int = 0

    def add_iteration(
        self,
        iteration_number: int,
        continue_decision: str,
        reasoning_notes: str,
        generated_queries: List[str],
    ):
        self.iterations.append(
            {
                "iteration": iteration_number,
                "continue_decision": continue_decision,
                "reasoning_notes": reasoning_notes,
                "generated_queries": generated_queries,
                "searches": [],
            }
        )

    def add_search_to_iteration(
        self, iteration_number: int, query: str, crawled_urls: List[str]
    ):
        for iter_data in self.iterations:
            if iter_data["iteration"] == iteration_number:
                iter_data["searches"].append(
                    {"query": query, "crawled_urls": crawled_urls}
                )
                return

    def format_report(self) -> str:
        # Calculate total metrics
        total_queries_executed = sum(
            len(iter_data.get("searches", [])) for iter_data in self.iterations
        )
        total_sources_crawled = sum(
            len(search.get("crawled_urls", []))
            for iter_data in self.iterations
            for search in iter_data.get("searches", [])
        )
        actual_iterations = len(
            [i for i in self.iterations if i.get("iteration", 0) > 0]
        )

        report_parts = [
            "\n\n" + "=" * 30 + " COMPLETE SEARCH DEBUG REPORT " + "=" * 30,
            f"INITIAL USER QUERY: {self.initial_user_query}",
            f"REFINED SEARCH QUERY: {self.refined_initial_query}",
            "",
            "─── VALVE SETTINGS vs ACTUAL RESULTS ───",
            f"Max Iterations (valve): {self.valve_max_iterations} | Executed: {actual_iterations}",
            f"URLs per Query (valve): {self.valve_urls_per_query} | Queries to Crawl (valve): {self.valve_queries_to_crawl}",
            f"Queries to Generate (valve): {self.valve_queries_to_generate}",
            f"Total Queries Executed: {total_queries_executed} | Total Sources Gathered: {total_sources_crawled}",
            f"Unique Sources in Notepad: {self.total_sources_found}",
            "",
            "─── RESEARCH ITERATIONS ───",
        ]

        for iter_data in self.iterations:
            iteration_num = iter_data.get("iteration", "N/A")

            if iteration_num == 0:
                report_parts.append("INITIAL SEARCH:")
                if iter_data.get("searches"):
                    initial_search = iter_data["searches"][0]
                    query = initial_search.get("query", "N/A")
                    crawled_urls = initial_search.get("crawled_urls", [])
                    report_parts.append(f'  Query: "{query}"')
                    report_parts.append(f"  Sources Found: {len(crawled_urls)}")
                    for i, url in enumerate(crawled_urls, 1):
                        report_parts.append(f"    {i}. {url}")
            else:
                decision = iter_data.get("continue_decision", "N/A")
                notes = iter_data.get("reasoning_notes", "")
                queries = iter_data.get("generated_queries", [])

                report_parts.append(f"ITERATION {iteration_num}:")
                report_parts.append(f"  Decision: {decision}")

                if queries:
                    report_parts.append(f"  Generated Queries ({len(queries)}):")
                    for i, query in enumerate(queries, 1):
                        report_parts.append(f"    {i}. {query}")

                for search_data in iter_data.get("searches", []):
                    query = search_data.get("query", "N/A")
                    crawled_urls = search_data.get("crawled_urls", [])
                    report_parts.append(
                        f"  Search: \"{query[:60]}{'...' if len(query) > 60 else ''}\""
                    )
                    report_parts.append(f"    Sources: {len(crawled_urls)}")
                    for i, url in enumerate(crawled_urls, 1):
                        report_parts.append(f"      {i}. {url}")

            report_parts.append("")

        report_parts.extend(
            [
                "─── FINAL SYNTHESIS ───",
                f"Synthesis Decision: {self.final_decision}",
                f"Final Output Preview:\n{self.final_output[:400]}{'...' if len(self.final_output) > 400 else ''}",
                "",
                "=" * 91 + "\n",
            ]
        )
        return "\n".join(report_parts)


# Valves and core functionality
class ToolsInternal:

    class Valves(BaseModel):
        exa_api_key: str = Field(default="", description="Your Exa API key.")
        router_model: str = Field(
            default="gpt-4o-mini",
            description="LLM for the initial CRAWL/STANDARD/COMPLETE decision.",
        )
        quick_search_model: str = Field(
            default="gpt-4o-mini",
            description="Single 'helper' model for all tasks in the STANDARD path (refining, summarizing).",
        )
        complete_agent_model: str = Field(
            default="gpt-4-turbo",
            description="The 'smart' model for all agentic steps in the COMPLETE path (refining, deciding, query generation).",
        )
        complete_summarizer_model: str = Field(
            default="gpt-4-turbo",
            description="Dedicated high-quality model for the final summary in the COMPLETE path.",
        )
        quick_urls_to_search: int = Field(
            default=5, description="Number of URLs to fetch for STANDARD search."
        )
        quick_queries_to_crawl: int = Field(
            default=3, description="Number of top URLs to crawl for STANDARD search."
        )
        quick_max_context_chars: int = Field(
            default=8000,
            description="Maximum total characters of context to feed to the STANDARD search summarizer.",
        )
        complete_urls_to_search_per_query: int = Field(
            default=5,
            description="Number of URLs to fetch for each targeted query in COMPLETE search.",
        )
        complete_queries_to_crawl: int = Field(
            default=3,
            description="Number of top URLs to crawl for each targeted query in COMPLETE search.",
        )
        complete_queries_to_generate: int = Field(
            default=3,
            description="Number of new targeted queries to generate per iteration.",
        )
        complete_max_search_iterations: int = Field(
            default=2, description="Maximum number of research loops for the agent."
        )
        debug_enabled: bool = Field(
            default=False,
            description="Enable detailed debug logging for troubleshooting search operations.",
        )
        show_sources: bool = Field(
            default=False,
            description="If true, return show_source=True so the UI can display sources. Prompts instruct the LLM not to include explicit citations in the text unless asked.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.debug = Debug(enabled=False)  # Will be updated when valves change
        self._exa: Optional[Exa] = None
        self._query_cache: Dict[str, Any] = {}  # Simple query caching
        self._cache_max_size = 100  # Limit cache size
        self._active_sessions: Dict[str, asyncio.Lock] = {}  # Session concurrency control
        self._session_lock = asyncio.Lock()  # Lock for managing session locks

    def _exa_client(self) -> Exa:
        if self._exa is None:
            if Exa is None:
                raise RuntimeError(
                    "exa_py not installed. Please install with: pip install exa_py"
                )
            key = self.valves.exa_api_key or os.getenv("EXA_API_KEY")
            if not key:
                raise RuntimeError(
                    "Exa API key missing. Please set exa_api_key in valves or EXA_API_KEY environment variable"
                )
            try:
                self._exa = Exa(key)
                self.debug.flow("🔑 Exa client initialised successfully")
            except Exception as e:
                self.debug.error(f"Failed to initialize Exa client: {e}")
                raise RuntimeError(f"Failed to initialize Exa client: {e}")
        return self._exa

    async def _safe_exa_search(
        self, query: str, num_results: int, debug_context: str = ""
    ) -> List[Any]:
        """Safely perform Exa search with error handling, caching, and latency metrics."""
        # Simple cache key based on query and num_results
        cache_key = f"{query}:{num_results}"

        # Check cache first
        if cache_key in self._query_cache:
            self.debug.search(f"Cache hit for {debug_context}: using cached results")
            return self._query_cache[cache_key]

        with self.debug.timer(f"exa_search_{debug_context}"):
            try:
                exa = self._exa_client()
                search_data = await asyncio.to_thread(
                    exa.search, query, num_results=num_results, use_autoprompt=True
                )
                results = search_data.results

                # Cache the results (with size limit)
                if len(self._query_cache) >= self._cache_max_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self._query_cache))
                    del self._query_cache[oldest_key]

                self._query_cache[cache_key] = results
                self.debug.search(
                    f"Exa search successful for {debug_context}: {len(results)} results (cached)"
                )
                self.debug.url_metrics(found=len(results))
                return results
            except Exception as e:
                self.debug.error(
                    f"Exa search failed for {debug_context}: {str(e)[:100]}..."
                )
                return []

    async def _safe_exa_crawl(
        self, ids_or_urls: List[str], debug_context: str = ""
    ) -> List[Any]:
        """Safely perform Exa content crawling with error handling, chunking, concurrency, and latency metrics."""
        if not ids_or_urls:
            return []

        with self.debug.timer(f"exa_crawl_{debug_context}"):
            try:
                exa = self._exa_client()
                # Chunk inputs to avoid oversized requests; run chunks concurrently
                chunk_size = 10
                chunks: List[List[str]] = [ids_or_urls[i:i+chunk_size] for i in range(0, len(ids_or_urls), chunk_size)]

                async def fetch_chunk(chunk: List[str]):
                    return await asyncio.to_thread(exa.get_contents, chunk)

                results_list = await asyncio.gather(*[fetch_chunk(c) for c in chunks], return_exceptions=True)

                combined = []
                total_requested = len(ids_or_urls)
                total_success = 0
                total_content_chars = 0
                
                for item in results_list:
                    if isinstance(item, Exception):
                        self.debug.error(f"Exa crawl chunk failed for {debug_context}: {str(item)[:100]}...")
                        continue
                    # item.results is expected from Exa
                    combined.extend(item.results)
                    total_success += len(item.results)
                    # Track content size
                    for result in item.results:
                        total_content_chars += len(getattr(result, 'text', ''))

                failed_count = total_requested - total_success
                
                if total_success < total_requested:
                    self.debug.error(
                        f"Exa crawl partial failure for {debug_context}: {total_success}/{total_requested} succeeded"
                    )
                else:
                    self.debug.crawl(
                        f"Exa crawl successful for {debug_context}: {total_success} sources"
                    )

                # Update metrics
                self.debug.url_metrics(crawled=total_requested, successful=total_success, failed=failed_count)
                self.debug.content_metrics(total_content_chars)
                
                return combined
            except Exception as e:
                self.debug.error(f"Exa crawl failed for {debug_context}: {str(e)[:100]}...")
                self.debug.url_metrics(crawled=len(ids_or_urls), failed=len(ids_or_urls))
                return []

    async def _extract_with_correction(
        self,
        *,
        request: Any,
        user_obj: Any,
        model: str,
        original_text: str,
        prefix: str,
        validate: Callable[[str], bool],
        correction_instructions: str,
        max_attempts: int = 2,
    ) -> str:
        """
        Extracts the substring following a required prefix from LLM output. If missing or invalid,
        performs up to `max_attempts` corrective re-prompts to coerce the model to output the correct format.

        Returns the best-effort extracted value. Callers should still defensively validate the return value.
        """
        def _extract(text: str) -> str:
            for line in text.splitlines():
                if line.strip().startswith(prefix):
                    return line.split(prefix, 1)[1].strip()
            return ""

        # First, try to extract from the original text
        extracted = _extract(original_text)
        if not extracted:
            # As a fallback, if there's only one non-empty line, use it
            lines = [ln.strip() for ln in original_text.splitlines() if ln.strip()]
            if len(lines) == 1:
                extracted = lines[0]

        if extracted and validate(extracted):
            return extracted

        # Attempt corrective re-prompts
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            self.debug.warning(
                f"Output missing/invalid '{prefix}' prefix. Attempting corrective re-prompt ({attempt}/{max_attempts})."
            )
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": correction_instructions,
                    },
                    {
                        "role": "user",
                        "content": f"Original response to fix:\n{original_text}",
                    },
                ],
                "stream": False,
            }

            try:
                fixed = await generate_with_retry(
                    request=request, form_data=payload, user=user_obj, debug=self.debug
                )
                fixed_text = fixed.get("choices", [{}])[0].get("message", {}).get("content", "")
                self.debug.data("Correction full response", fixed_text, truncate=200)

                extracted = _extract(fixed_text) or fixed_text.strip()
                if extracted and validate(extracted):
                    return extracted
            except Exception as e:
                self.debug.error(f"Corrective re-prompt failed: {e}")

        # Last resort: return a trimmed original if it passes, else empty string
        fallback = original_text.strip()
        return fallback if validate(fallback) else ""

    # Main
    async def _get_session_lock(self, user_id: str, query_hash: str) -> asyncio.Lock:
        """Get or create a session lock for concurrent call protection."""
        session_key = f"{user_id}_{query_hash}"
        
        async with self._session_lock:
            if session_key not in self._active_sessions:
                self._active_sessions[session_key] = asyncio.Lock()
            return self._active_sessions[session_key]
    
    async def _cleanup_session_lock(self, user_id: str, query_hash: str) -> None:
        """Clean up session lock after completion."""
        session_key = f"{user_id}_{query_hash}"
        
        async with self._session_lock:
            if session_key in self._active_sessions:
                del self._active_sessions[session_key]

    async def routed_search(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __request__: Optional[Any] = None,
        __user__: Optional[Dict] = None,
        __messages__: Optional[List[Dict]] = None,
        image_context: Optional[str] = None,
    ) -> dict:
        # Guard: some environments may not have show_sources in valves
        show_sources = bool(getattr(self.valves, "show_sources", False))
        # Check if Exa is available first
        if not EXA_AVAILABLE:
            error_msg = "❌ Search tool unavailable: exa_py module not installed. Please install with: pip install exa_py"
            self.debug.error(error_msg)
            return {
                "content": error_msg,
                "show_source": show_sources,
            }

        # Generate session identifiers for concurrency control
        user_id = __user__.get("id", "unknown") if __user__ else "unknown"
        query_hash = str(hash(query))[-8:]  # Use last 8 chars of query hash
        
        # Get session lock to prevent concurrent calls
        session_lock = await self._get_session_lock(user_id, query_hash)
        
        # Check if another instance is already running for this user/query combo
        if session_lock.locked():
            self.debug.flow(f"Concurrent call detected for user {user_id}, query hash {query_hash}")
            return {
                "content": "⚠️ A search is already in progress for this query. Please wait for it to complete before starting a new search.",
                "show_source": show_sources,
            }
        
        # Acquire the lock for this session
        async with session_lock:
            # Update debug state based on current valve setting
            self.debug.enabled = self.valves.debug_enabled
            if self.debug.enabled:
                self.debug.start_session(f"Query: {query[:50]}...")
            self.debug.flow("Starting SearchRouterTool processing")
            
            try:
                return await self._execute_search(
                    query, __event_emitter__, __request__, __user__, __messages__, image_context, show_sources
                )
            finally:
                # Clean up the session lock
                await self._cleanup_session_lock(user_id, query_hash)
                # Safety: ensure any transient status is cleared at the end
                if __event_emitter__:
                    try:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": "", "done": True},
                        })
                    except Exception:
                        pass
    
    async def _execute_search(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __request__: Optional[Any] = None,
        __user__: Optional[Dict] = None,
        __messages__: Optional[List[Dict]] = None,
        image_context: Optional[str] = None,
        show_sources: bool = False,
    ) -> dict:

        async def _status(desc: str, done: bool = False) -> None:
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": desc, "done": done}}
                )

        # Add debug info about the tool being called
        self.debug.flow(f"ExaSearch tool called with query: {query[:100]}...")
        if __user__:
            self.debug.data("User ID", __user__.get("id", "unknown"))
        if __messages__:
            self.debug.data("Message count", len(__messages__))

        messages = __messages__ or []
        last_user_message = get_last_user_message(messages)
        if not last_user_message:
            self.debug.error("Could not find a user message to process")
            return {
                "content": "Could not find a user message to process. Please try again.",
                "show_source": show_sources,
            }

        self.debug.data("User query", query)
        self.debug.data("Last user message", last_user_message)

        # Build conversation history snippet for context
        history_messages = messages[-6:-1]
        convo_snippet_parts = []
        for m in history_messages:
            text_content = _get_text_from_message(m.get("content", ""))
            role = m.get("role", "").upper()
            convo_snippet_parts.append(f"{role}: {text_content!r}")
        convo_snippet = "\n".join(convo_snippet_parts)

        # Include image context if provided by autotoo
        if image_context:
            self.debug.flow("Image context provided by autotoo, enhancing search")
            convo_snippet += f"\n\nIMAGE CONTEXT: {image_context}"

        # The definitive query for the router now includes history
        router_query = f"Conversation History:\n{convo_snippet}\n\nLatest User Query:\n'{last_user_message}'"

        user_obj = Users.get_user_by_id(__user__["id"]) if __user__ else None
        self.debug.router(f"Router triggered with full query context")
        self.debug.data("Router query", router_query, truncate=150)
        await _status("Deciding search strategy…")

        # Check for per-call override from upstream router to skip internal strategy LLM
        override_decision = ""
        if __messages__:
            try:
                for m in __messages__:
                    if m.get("role") == "system" and isinstance(m.get("content"), str):
                        content = m.get("content", "")
                        if "[EXA_SEARCH_MODE]" in content:
                            # Expected formats:
                            # [EXA_SEARCH_MODE] STANDARD
                            # [EXA_SEARCH_MODE] CRAWL
                            # [EXA_SEARCH_MODE] COMPLETE
                            mode = (
                                content.split("[EXA_SEARCH_MODE]", 1)[1]
                                .strip()
                                .split()[0]
                                .upper()
                            )
                            if mode in {"CRAWL", "STANDARD", "COMPLETE"}:
                                override_decision = mode
                                self.debug.router(
                                    f"Override strategy detected via system message → {override_decision}"
                                )
                                break
            except Exception as _:
                # Swallow parsing errors to avoid breaking normal flow
                pass

        if override_decision:
            decision = override_decision
        else:
            router_payload = {
                "model": self.valves.router_model,
                "messages": [
                    {"role": "system", "content": SEARCH_STRATEGY_ROUTER_PROMPT_TEMPLATE},
                    {"role": "user", "content": router_query},  # Use the full context query
                ],
                "stream": False,
            }
            try:
                res = await generate_with_retry(
                    request=__request__,
                    form_data=router_payload,
                    user=user_obj,
                    debug=self.debug,
                )
                llm_response_text = res["choices"][0]["message"]["content"]
                self.debug.data("Router full response", llm_response_text, truncate=200)

                # Extract the decision using a robust prefixed-line correction helper
                decision = await self._extract_with_correction(
                    request=__request__,
                    user_obj=user_obj,
                    model=self.valves.router_model,
                    original_text=llm_response_text,
                    prefix="ANSWER:",
                    validate=lambda s: s in {"CRAWL", "STANDARD", "COMPLETE"},
                    correction_instructions=(
                        "Return exactly one line with the routing decision in the format 'ANSWER: CRAWL' or 'ANSWER: STANDARD' or 'ANSWER: COMPLETE'. "
                        "Do not include any other text."
                    ),
                )

                if decision not in {"CRAWL", "STANDARD", "COMPLETE"}:
                    self.debug.router(
                        "Could not parse router decision even after correction. Defaulting to STANDARD."
                    )
                    decision = "STANDARD"

            except Exception as exc:
                self.debug.error(
                    f"Router LLM failed after retries: {exc}. Defaulting to STANDARD."
                )
                decision = "STANDARD"

        self.debug.router(f"Router decision → {decision}")
        exa = self._exa_client()

        # Mode 1 - Crawl
        if decision == "CRAWL":
            urls = URL_RE.findall(last_user_message)
            if not urls:
                return {
                    "content": "You requested a crawl, but I could not find a URL in your message. Please provide a valid URL.",
                    "show_source": show_sources,
                }

            url_to_crawl = urls[0]
            await _status("Reading content from URL...")
            self.debug.crawl(f"Executing CRAWL on: {url_to_crawl}")
            try:
                start_crawl = time.perf_counter()
                crawled_results = await asyncio.to_thread(exa.get_contents, [url_to_crawl])
                elapsed_crawl = time.perf_counter() - start_crawl
                content = (
                    crawled_results.results[0].text
                    if crawled_results.results
                    else "Could not retrieve any text content from the URL."
                )
                await _status("Crawl complete.", done=True)
                self.debug.crawl(f"Successfully crawled {len(content)} characters in {elapsed_crawl:.2f}s")

                # Simple CRAWL mode debug report
                if self.debug.enabled:
                    crawl_report = [
                        "\n\n" + "=" * 35 + " CRAWL MODE DEBUG REPORT " + "=" * 35,
                        f"URL REQUESTED: {url_to_crawl}",
                        f"CRAWL SUCCESS: ✓ Yes",
                        f"CONTENT LENGTH: {len(content)} characters",
                        f"CONTENT PREVIEW:\n{content[:300]}{'...' if len(content) > 300 else ''}",
                        "",
                        "=" * 96 + "\n",
                    ]
                    self.debug.report("\n".join(crawl_report))
                    self.debug.content_metrics(len(content))
                    self.debug.metrics_summary()

                self.debug.synthesis("Crawl complete, returning content.")
                return {
                    "content": content,
                    "show_source": show_sources,
                }
            except Exception as e:
                self.debug.error(f"Crawl failed: {e}")

                # CRAWL mode failure debug report
                if self.debug.enabled:
                    crawl_report = [
                        "\n\n" + "=" * 35 + " CRAWL MODE DEBUG REPORT " + "=" * 35,
                        f"URL REQUESTED: {url_to_crawl}",
                        f"CRAWL SUCCESS: ✗ No",
                        f"ERROR: {str(e)}",
                        "",
                        "=" * 96 + "\n",
                    ]
                    self.debug.report("\n".join(crawl_report))
                    self.debug.url_metrics(failed=1)
                    self.debug.metrics_summary()

                return {
                    "content": f"I failed while trying to crawl the URL: {e}",
                    "show_source": show_sources,
                }

        # Mode 2 - Standard
        elif decision == "STANDARD":
            self.debug.flow("Starting STANDARD search mode")
            report = QuickDebugReport(
                initial_query=last_user_message,
                valve_urls_to_search=self.valves.quick_urls_to_search,
                valve_queries_to_crawl=self.valves.quick_queries_to_crawl,
                valve_max_context_chars=self.valves.quick_max_context_chars,
            )
            final_result = ""
            context = ""
            try:
                await _status("Formulating search plan...")

                refiner_user_prompt = f"## Conversation History:\n{convo_snippet}\n\n## User's Latest Query:\n'{last_user_message}'"
                refiner_payload = {
                    "model": self.valves.quick_search_model,
                    "messages": [
                        {"role": "system", "content": IMPROVED_SQR_PROMPT_TEMPLATE},
                        {"role": "user", "content": refiner_user_prompt},
                    ],
                    "stream": False,
                }

                res = await generate_with_retry(
                    request=__request__,
                    form_data=refiner_payload,
                    user=user_obj,
                    debug=self.debug,
                )
                llm_response_text = res["choices"][0]["message"]["content"].strip()
                self.debug.data("SQR full response", llm_response_text, truncate=200)

                # Extract refined query using correction helper
                refined_query = await self._extract_with_correction(
                    request=__request__,
                    user_obj=user_obj,
                    model=self.valves.quick_search_model,
                    original_text=llm_response_text,
                    prefix="ANSWER:",
                    validate=lambda s: len(s) > 0 and len(s) <= 1000,
                    correction_instructions=(
                        "Return exactly one line starting with 'ANSWER: ' followed by the optimized search query. "
                        "No explanations or quotes."
                    ),
                )
                self.debug.query(f"Refined STANDARD query: {refined_query}")
                report.refined_query = refined_query

                await _status(f'Searching for: "{refined_query}"')

                # Use safe search method
                search_results = await self._safe_exa_search(
                    refined_query, self.valves.quick_urls_to_search, "STANDARD search"
                )

                if not search_results:
                    final_result = "My search found no results to read. Please try a different query."
                    self.debug.search("No search results found")
                else:
                    report.urls_found = [res.url for res in search_results]
                    self.debug.search(f"Found {len(report.urls_found)} search results")

                    crawl_candidates = search_results[
                        : self.valves.quick_queries_to_crawl
                    ]

                    if not crawl_candidates:
                        final_result = "My search found no results to read. Please try a different query."
                        self.debug.search("No crawl candidates found")
                    else:
                        domains = [
                            urlparse(res.url).netloc.replace("www.", "")
                            for res in crawl_candidates
                        ]
                        await _status(f"Reading from: {', '.join(domains)}")
                        self.debug.crawl(
                            f"Crawling {len(crawl_candidates)} URLs from domains: {domains}"
                        )

                        ids_to_crawl = [res.id for res in crawl_candidates]
                        report.urls_crawled = [res.url for res in crawl_candidates]

                        # Use safe crawl method
                        crawled_results = await self._safe_exa_crawl(
                            ids_to_crawl, "STANDARD search"
                        )

                        # Populate report with success/failure data
                        successful_urls = [res.url for res in crawled_results]
                        report.urls_successful = successful_urls

                        failed_urls = [
                            url
                            for url in report.urls_crawled
                            if url not in successful_urls
                        ]
                        report.urls_failed = failed_urls

                        if crawled_results:
                            await _status(
                                f"Synthesizing answer from {len(crawled_results)} sources..."
                            )
                            context = "\n\n".join(
                                [
                                    f"## Source: {res.url}\n\n{res.text}"
                                    for res in crawled_results
                                ]
                            )
                            context = context[: self.valves.quick_max_context_chars]

                            # Populate report metrics
                            report.context_length = len(context)
                            report.was_truncated = (
                                len(
                                    "\n\n".join(
                                        [
                                            f"## Source: {res.url}\n\n{res.text}"
                                            for res in crawled_results
                                        ]
                                    )
                                )
                                > self.valves.quick_max_context_chars
                            )

                            self.debug.synthesis(
                                f"Generated context with {len(context)} characters from {len(crawled_results)} sources"
                            )

                            summarizer_user_prompt = f"## Context:\n{context}\n\n## User's Question:\n{last_user_message}"
                            report.final_prompt = f"SYSTEM: {QUICK_SUMMARIZER_PROMPT}\nUSER: {summarizer_user_prompt}"
                            summarizer_payload = {
                                "model": self.valves.quick_search_model,
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": QUICK_SUMMARIZER_PROMPT,
                                    },
                                    {"role": "user", "content": summarizer_user_prompt},
                                ],
                                "stream": False,
                            }
                            final_res = await generate_with_retry(
                                request=__request__,
                                form_data=summarizer_payload,
                                user=user_obj,
                                debug=self.debug,
                            )
                            final_result = final_res["choices"][0]["message"]["content"]
                            await _status("Standard search complete.", done=True)
                            self.debug.synthesis("STANDARD search synthesis complete")
                        else:
                            final_result = "I found search results but was unable to read any content from them. Please try a different query."

            except Exception as e:
                self.debug.error(f"STANDARD search path failed with an exception: {e}")
                if context:
                    final_result = f"I found some information but encountered an error while processing it. Here is the raw data I gathered:\n\n{context}"
                else:
                    final_result = f"I failed during the standard search: {e}"
            finally:
                report.final_output = final_result
                if self.debug.enabled:
                    self.debug.report(report.format_report())
                    self.debug.metrics_summary()
                return {"content": final_result, "show_source": show_sources}

        # Mode 3 - Complete (New Iterative Research System)
        elif decision == "COMPLETE":
            return await self._iterative_complete_search(
                last_user_message, convo_snippet, _status, __request__, user_obj, show_sources
            )

        self.debug.flow("SearchRouterTool processing completed")
        if self.debug.enabled:
            self.debug.metrics_summary()
        return {
            "content": f"Router chose '{decision}', but no corresponding action was taken.",
            "show_source": show_sources,
        }

    async def _iterative_complete_search(
        self, user_query: str, convo_snippet: str, status_func, request, user_obj, show_sources: bool
    ) -> dict:
        """New iterative complete search system based on user's vision"""
        self.debug.flow("Starting NEW iterative complete search mode")
        
        try:
            # Get current datetime for all prompts
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_year = datetime.now().year
            
            # Phase 1: Generate introductory query for context
            await status_func("Gathering initial context...")
            self.debug.agent("Phase 1: Generating introductory query")
            
            intro_payload = {
                "model": self.valves.complete_agent_model,
                "messages": [
                    {"role": "system", "content": INTRODUCTORY_QUERY_PROMPT.format(current_date=current_date, current_year=current_year)},
                    {"role": "user", "content": f"User's request: {user_query}"},
                ],
                "stream": False,
            }
            
            intro_res = await generate_with_parsing_retry(
                request=request, 
                form_data=intro_payload, 
                user=user_obj, 
                debug=self.debug,
                expected_keys=["choices", "content", "message"]
            )
            
            # Debug the response structure
            self.debug.data("Intro LLM raw response type", type(intro_res))
            self.debug.data("Intro LLM raw response keys", list(intro_res.keys()) if isinstance(intro_res, dict) else "Not a dict")
            self.debug.data("Intro LLM full response", str(intro_res)[:800] + "..." if len(str(intro_res)) > 800 else str(intro_res))
            
            # Handle different response formats
            if "choices" in intro_res and intro_res["choices"]:
                intro_response = intro_res["choices"][0]["message"]["content"]
            elif "content" in intro_res:
                intro_response = intro_res["content"]
            elif "message" in intro_res:
                intro_response = intro_res["message"]
            elif isinstance(intro_res, str):
                intro_response = intro_res
            else:
                raise ValueError(f"Unexpected intro LLM response format. Keys: {list(intro_res.keys()) if isinstance(intro_res, dict) else 'Not a dict'}")
            
            # Extract intro query using correction helper
            intro_query = await self._extract_with_correction(
                request=request,
                user_obj=user_obj,
                model=self.valves.complete_agent_model,
                original_text=intro_response,
                prefix="QUERY:",
                validate=lambda s: len(s) > 0,
                correction_instructions=(
                    "Return exactly one line starting with 'QUERY: ' followed by a concise introductory search query. "
                    "No additional text."
                ),
            )
            
            self.debug.data("Introductory query extracted", intro_query)
            
            # Search with the introductory query - with retry logic for failures
            intro_content = ""
            intro_search_attempts = 0
            max_intro_attempts = 3
            
            while intro_search_attempts < max_intro_attempts and not intro_content:
                intro_search_attempts += 1
                self.debug.search(f"Executing introductory search (attempt {intro_search_attempts}) — query: {intro_query}")
                
                try:
                    # Perform search + crawl using internal helpers (replaces missing _search_and_crawl)
                    search_results = await self._safe_exa_search(
                        intro_query, self.valves.complete_urls_to_search_per_query, "introductory"
                    )
                    intro_results = {"content": "", "sources": []}
                    if search_results:
                        ids_to_crawl = [res.id for res in search_results[: self.valves.complete_queries_to_crawl]]
                        crawled_results = await self._safe_exa_crawl(ids_to_crawl, "introductory")
                        if crawled_results:
                            texts = []
                            for res in crawled_results:
                                if getattr(res, "text", None):
                                    texts.append(' '.join(res.text.split()[:3000]))
                                if getattr(res, "url", None):
                                    intro_results["sources"].append(res.url)
                            intro_results["content"] = "\n\n".join(texts)
                    intro_content = intro_results.get("content", "")
                    
                    if intro_content and intro_content.strip():
                        if intro_results.get("sources"):
                            self.debug.data("Intro search sources found", len(intro_results["sources"]))
                        break
                    else:
                        self.debug.error(f"Introductory search attempt {intro_search_attempts} returned empty content")
                        if intro_search_attempts < max_intro_attempts:
                            # Modify query slightly for retry
                            intro_query = f"{intro_query} overview basics"
                            await status_func(f"Retrying initial context search (attempt {intro_search_attempts + 1})...")
                        
                except Exception as e:
                    self.debug.error(f"Introductory search attempt {intro_search_attempts} failed: {str(e)}")
                    if intro_search_attempts < max_intro_attempts:
                        await status_func(f"Retrying initial context search (attempt {intro_search_attempts + 1})...")
            
            if not intro_content:
                intro_content = "Unable to gather initial context after multiple attempts. Proceeding with user query directly."
                self.debug.warning("All introductory search attempts failed, using fallback")
            
            # Phase 2: Set objectives and research direction
            await status_func("Setting research objectives...")
            self.debug.agent("Phase 2: Setting research objectives")
            
            objectives_payload = {
                "model": self.valves.complete_agent_model,
                "messages": [
                    {"role": "system", "content": OBJECTIVE_SETTING_PROMPT.format(current_date=current_date)},
                    {"role": "user", "content": f"User's request: {user_query}\n\nConversation context: {convo_snippet}\n\nIntroductory information:\n{intro_content}"},
                ],
                "stream": False,
            }
            
            objectives_res = await generate_with_parsing_retry(
                request=request, 
                form_data=objectives_payload, 
                user=user_obj, 
                debug=self.debug,
                expected_keys=["choices", "content", "message"]
            )
            
            # Handle different response formats
            if "choices" in objectives_res and objectives_res["choices"]:
                objectives_response = objectives_res["choices"][0]["message"]["content"]
            elif "content" in objectives_res:
                objectives_response = objectives_res["content"]
            elif "message" in objectives_res:
                objectives_response = objectives_res["message"]
            elif isinstance(objectives_res, str):
                objectives_response = objectives_res
            else:
                raise ValueError(f"Unexpected objectives LLM response format. Keys: {list(objectives_res.keys()) if isinstance(objectives_res, dict) else 'Not a dict'}")
            self.debug.data("Research objectives", objectives_response, truncate=300)
            
            # Phase 3: Iterative search with reasoning
            research_chain = [f"INITIAL CONTEXT: {intro_content[:10000]}...", f"OBJECTIVES: {objectives_response}"]
            previous_findings = "Initial context gathered from introductory search."
            
            for iteration in range(1, self.valves.complete_max_search_iterations + 1):
                self.debug.iteration(f"Starting research iteration {iteration}/{self.valves.complete_max_search_iterations}")
                await status_func(f"Research iteration {iteration}/{self.valves.complete_max_search_iterations}...")
                
                # Generate reasoning and queries for this iteration
                reasoning_prompt = ITERATION_REASONING_PROMPT.format(
                    current_date=current_date,
                    current_year=current_year,
                    objectives=objectives_response[:6000],
                    previous_findings=previous_findings[:16000],
                    current_iteration=iteration,
                    max_iterations=self.valves.complete_max_search_iterations,
                    query_count=self.valves.complete_queries_to_generate
                )
                
                reasoning_payload = {
                    "model": self.valves.complete_agent_model,
                    "messages": [
                        {"role": "system", "content": "You are a research iteration planner."},
                        {"role": "user", "content": reasoning_prompt},
                    ],
                    "stream": False,
                }
                
                reasoning_res = await generate_with_parsing_retry(
                    request=request, 
                    form_data=reasoning_payload, 
                    user=user_obj, 
                    debug=self.debug,
                    expected_keys=["choices", "content", "message"]
                )
                
                # Handle different response formats
                if "choices" in reasoning_res and reasoning_res["choices"]:
                    reasoning_response = reasoning_res["choices"][0]["message"]["content"]
                elif "content" in reasoning_res:
                    reasoning_response = reasoning_res["content"]
                elif "message" in reasoning_res:
                    reasoning_response = reasoning_res["message"]
                elif isinstance(reasoning_res, str):
                    reasoning_response = reasoning_res
                else:
                    raise ValueError(f"Unexpected reasoning LLM response format. Keys: {list(reasoning_res.keys()) if isinstance(reasoning_res, dict) else 'Not a dict'}")
                self.debug.agent(f"Iteration {iteration} reasoning: {reasoning_response[:200]}...")
                
                # Extract queries from reasoning response with more robust parsing
                queries = []
                
                # Method 1: Look for QUERIES: line and extract from it
                for line in reasoning_response.split("\n"):
                    if "QUERIES:" in line.upper():
                        # Try to extract from same line if it contains array-like structure
                        if "[" in line and "]" in line:
                            try:
                                import json
                                query_part = line.split("QUERIES:")[-1].strip()
                                queries = json.loads(query_part)
                                break
                            except json.JSONDecodeError:
                                continue
                        
                        # Look at following lines for array structure
                        lines = reasoning_response.split("\n")
                        start_idx = lines.index(line)
                        for i in range(start_idx + 1, min(start_idx + 10, len(lines))):
                            current_line = lines[i].strip()
                            if current_line.startswith("[") and "]" in current_line:
                                try:
                                    import json
                                    queries = json.loads(current_line)
                                    break
                                except json.JSONDecodeError:
                                    continue
                        break
                
                # Method 2: Extract individual quoted strings
                if not queries:
                    import re
                    quoted_queries = re.findall(r'"([^"]*)"', reasoning_response)
                    if quoted_queries:
                        queries = [q for q in quoted_queries if len(q) > 5]  # Filter short quotes
                
                # Method 3: Look for list-style queries
                if not queries:
                    lines = reasoning_response.split("\n")
                    for line in lines:
                        if (line.strip().startswith("-") or line.strip().startswith("*") or 
                            line.strip().startswith("1.") or line.strip().startswith("2.")):
                            query = line.strip().lstrip("-*123456789. ").strip()
                            if len(query) > 10:
                                queries.append(query)
                
                # Final fallback
                if not queries:
                    queries = [f"{user_query} detailed research iteration {iteration}"]
                    self.debug.error(f"Could not extract queries from reasoning, using fallback")
                
                queries = queries[:self.valves.complete_queries_to_generate]  # Limit to valve setting
                self.debug.search(f"Iteration {iteration} queries: {queries}")
                
                # Execute searches for this iteration
                iteration_findings = []
                for query in queries:
                    await status_func(f"Searching: {query[:40]}...")
                    
                    search_results = await self._safe_exa_search(
                        query, self.valves.complete_urls_to_search_per_query, f"iteration {iteration} query"
                    )
                    
                    if search_results:
                        ids_to_crawl = [res.id for res in search_results[:self.valves.complete_queries_to_crawl]]
                        crawled_results = await self._safe_exa_crawl(ids_to_crawl, f"iteration {iteration} crawl")
                        
                        if crawled_results:
                            for res in crawled_results:
                                if res.text and res.text.strip():
                                    title = res.title or "Unknown Source"
                                    text_summary = ' '.join(res.text.split()[:3000])
                                    if text_summary:
                                        finding_summary = f"From {title}: {text_summary}"
                                        iteration_findings.append(finding_summary)
                
                # Conclude iteration
                await status_func(f"Analyzing iteration {iteration} findings...")
                
                iteration_content = "\n\n".join(iteration_findings) if iteration_findings else "No significant findings in this iteration."
                
                conclusion_prompt = f"""
                Research findings from iteration {iteration}:
                {iteration_content}
                
                User's original question: {user_query}
                Research objectives: {objectives_response}
                Previous findings summary: {previous_findings}
                
                Analyze these findings and determine next steps.
                """
                
                conclusion_payload = {
                    "model": self.valves.complete_agent_model,
                    "messages": [
                        {"role": "system", "content": ITERATION_CONCLUSION_PROMPT.format(
                            current_date=current_date,
                            current_iteration=iteration,
                            max_iterations=self.valves.complete_max_search_iterations
                        )},
                        {"role": "user", "content": conclusion_prompt},
                    ],
                    "stream": False,
                }
                
                conclusion_res = await generate_with_parsing_retry(
                    request=request, 
                    form_data=conclusion_payload, 
                    user=user_obj, 
                    debug=self.debug,
                    expected_keys=["choices", "content", "message"]
                )
                
                # Handle different response formats
                if "choices" in conclusion_res and conclusion_res["choices"]:
                    conclusion_response = conclusion_res["choices"][0]["message"]["content"]
                elif "content" in conclusion_res:
                    conclusion_response = conclusion_res["content"]
                elif "message" in conclusion_res:
                    conclusion_response = conclusion_res["message"]
                elif isinstance(conclusion_res, str):
                    conclusion_response = conclusion_res
                else:
                    raise ValueError(f"Unexpected conclusion LLM response format. Keys: {list(conclusion_res.keys()) if isinstance(conclusion_res, dict) else 'Not a dict'}")
                
                # Extract decision
                decision = "CONTINUE"
                if "DECISION:" in conclusion_response.upper():
                    decision_line = [line for line in conclusion_response.split("\n") if "DECISION:" in line.upper()]
                    if decision_line:
                        decision = decision_line[0].split(":")[-1].strip().upper()
                
                # Update research chain (pass down summaries, not raw content)
                findings_summary = "No summary available"
                for line in conclusion_response.split("\n"):
                    if "FINDINGS_SUMMARY:" in line.upper():
                        findings_summary = line.split(":", 1)[1].strip()
                        break
                
                research_chain.append(f"ITERATION {iteration}: {findings_summary}")
                previous_findings = findings_summary
                
                self.debug.agent(f"Iteration {iteration} decision: {decision}")
                
                if decision == "FINISH" or iteration == self.valves.complete_max_search_iterations:
                    self.debug.agent("Research concluded - synthesizing final answer")
                    break
            
            # Phase 4: Final synthesis
            await status_func("Synthesizing comprehensive answer...")
            self.debug.synthesis("Generating final synthesis from research chain")
            
            research_summary = "\n\n".join(research_chain)
            
            final_payload = {
                "model": self.valves.complete_summarizer_model,
                "messages": [
                    {"role": "system", "content": FINAL_SYNTHESIS_PROMPT.format(current_date=current_date)},
                    {"role": "user", "content": f"User's original question: {user_query}\n\nResearch progression and findings:\n{research_summary}"},
                ],
                "stream": False,
            }
            
            final_res = await generate_with_parsing_retry(
                request=request, 
                form_data=final_payload, 
                user=user_obj, 
                debug=self.debug,
                expected_keys=["choices", "content", "message"]
            )
            
            # Handle different response formats
            if "choices" in final_res and final_res["choices"]:
                final_answer = final_res["choices"][0]["message"]["content"]
            elif "content" in final_res:
                final_answer = final_res["content"]
            elif "message" in final_res:
                final_answer = final_res["message"]
            elif isinstance(final_res, str):
                final_answer = final_res
            else:
                raise ValueError(f"Unexpected final synthesis LLM response format. Keys: {list(final_res.keys()) if isinstance(final_res, dict) else 'Not a dict'}")
            
            await status_func("Research complete.", done=True)
            self.debug.synthesis("Iterative complete search finished successfully")
            
            if self.debug.enabled:
                self.debug.metrics_summary()
            
            return {"content": final_answer, "show_source": show_sources}
            
        except Exception as e:
            self.debug.error(f"Iterative complete search failed: {e}")
            if self.debug.enabled:
                self.debug.metrics_summary()
            # Safety: clear any lingering status on failure
            try:
                await status_func("", done=True)
            except Exception:
                pass
            return {
                "content": f"I encountered an error during the research process: {e}"
            }


# Final tool definition for OpenWebUI
class Tools:
    """Main class that OpenWebUI will use"""

    def __init__(self):
        self.tools_instance = ToolsInternal()
        # Expose valves directly at the top level for OpenWebUI
        self.valves = self.tools_instance.valves

    class Valves(BaseModel):
        exa_api_key: str = Field(default="", description="Your Exa API key.")
        router_model: str = Field(
            default="gpt-4o-mini",
            description="LLM for the initial CRAWL/STANDARD/COMPLETE decision.",
        )
        quick_search_model: str = Field(
            default="gpt-4o-mini",
            description="Single 'helper' model for all tasks in the STANDARD path (refining, summarizing).",
        )
        complete_agent_model: str = Field(
            default="gpt-4-turbo",
            description="The 'smart' model for all agentic steps in the COMPLETE path (refining, deciding, query generation).",
        )
        complete_summarizer_model: str = Field(
            default="gpt-4-turbo",
            description="Dedicated high-quality model for the final summary in the COMPLETE path.",
        )
        quick_urls_to_search: int = Field(
            default=5, description="Number of URLs to fetch for STANDARD search."
        )
        quick_queries_to_crawl: int = Field(
            default=3, description="Number of top URLs to crawl for STANDARD search."
        )
        quick_max_context_chars: int = Field(
            default=8000,
            description="Maximum total characters of context to feed to the STANDARD search summarizer.",
        )
        complete_urls_to_search_per_query: int = Field(
            default=5,
            description="Number of URLs to fetch for each targeted query in COMPLETE search.",
        )
        complete_queries_to_crawl: int = Field(
            default=3,
            description="Number of top URLs to crawl for each targeted query in COMPLETE search.",
        )
        complete_queries_to_generate: int = Field(
            default=3,
            description="Number of new targeted queries to generate per iteration.",
        )
        complete_max_search_iterations: int = Field(
            default=2, description="Maximum number of research loops for the agent."
        )
        debug_enabled: bool = Field(
            default=False,
            description="Enable detailed debug logging for troubleshooting search operations.",
        )

    async def routed_search(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __request__: Optional[Any] = None,
        __user__: Optional[Dict] = None,
        __messages__: Optional[List[Dict]] = None,
    ) -> dict:
        # Create debug instance with consistent formatting
        debug = Debug(enabled=self.valves.debug_enabled)
        debug.flow("routed_search function called")
        debug.data("Query", query[:50] + "..." if len(query) > 50 else query)

        # Sync valve settings to internal instance
        self.tools_instance.valves = self.valves

        return await self.tools_instance.routed_search(
            query, __event_emitter__, __request__, __user__, __messages__
        )
