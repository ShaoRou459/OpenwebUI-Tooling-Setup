"""
Title: Search Router Tool
Description: An advanced research tool with a robust retry and graceful failure mechanism.
author: ShaoRou459
author_url: https://github.com/ShaoRou459
Version: 1.0.0
Requirements: exa_py, open_webui
"""

from __future__ import annotations

import os
import re
import sys
import json
import asyncio
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
except ImportError:
    Exa = None


# ─── System Prompts ───────────────────────────────────────────────────────────

SEARCH_STRATEGY_PROMPT = """You are a meticulous Search Strategy Analyst. Your role is to analyze the user's query and determine the best search strategy. You must choose one of three strategies: CRAWL, STANDARD, or COMPLETE.\n\n"
## Strategies:
- **CRAWL**: Choose this ONLY if the user provides a specific URL and asks a question about it.
- **STANDARD**: Choose this for most questions, they are questions that can be answered within 5 minutes of a Google Search. This is good for 95% of queries (e.g., 'What is the capital of France?').
- **COMPLETE**: [DO NOT CHOOSE UNLESS USER EXPLICTLY SAYS TO DO DEEP RESEARCH] Choose this for complex, open-ended, or comparative questions that require gathering information from multiple sources and synthesizing a detailed answer (e.g., 'Compare the pros and cons of React vs. Vue.js').

## Response Format:
First, think through your reasoning inside a <think> block. Then, on a new line, state your final decision in the format: `Final Answer: <strategy>` where <strategy> is one of CRAWL, STANDARD, or COMPLETE.

### Example:
User: 'What are the main differences between ARM and x86? Do some deep research'

<think>
USER EXPLICTLY ASKED FOR A DEEP RESEARCH + The user is asking for a comparison between two complex topics. This is not a simple fact lookup. It will require searching for information on both ARM and x86, then analyzing and comparing them. This clearly falls under the COMPLETE strategy.
</think>
Final Answer: COMPLETE
"""

SQR_PROMPT_TEMPLATE = f"""
You are a **Search Query Refinement (SQR) bot**.

Your task is to turn a user's free-form question into the smallest, highest-yield keyword query.

First, think step-by-step inside a `<think>` block. In your thinking process, you MUST follow the SQR PROTOCOL to construct the query.
After the `<think>` block, on a new line, output *ONLY* the final, assembled query string.

-------------------------------------------------
SQR PROTOCOL (to be followed inside the `<think>` block)

STEP 0️⃣ — DATE Token  
If the user query includes any temporal cue («latest», «new in», «current», «today», «this week», etc.), append  
`(YYYY M D)`, using the current date of {datetime.now().year} {datetime.now().month} {datetime.now().day}.  
If not, omit the date.

STEP 1️⃣ — Keyword Extraction  
Pick the *core noun+verb pair* plus any *must-have context words* (brand, city, SKU ID, regulation, version number, etc.).

STEP 2️⃣ — Functional Modifiers  
Add the minimal disambiguation word(s) that narrow the search domain:  
- For factual answers → add «wikidata.org» or «government.ca etc»  
- For opinions/recs → add «Reddit OR StackOverflow OR Hacker News etc»  
- For price → add «price OR cost OR MSRP etc»  
- For download → add «download OR .pdf OR repo etc»  
- For news → add «NYTimes OR SCMP OR Reuters etc» (pick the most relevant subset).

STEP 3️⃣ — Breadth Tokens  
Append 2-4 comma-separated authoritative / crowd-sourced sources or high-coverage keywords to guarantee wide recall.

STEP 4️⃣ — Assembly (No Spaces After Commas)  
Assemble the parts in this order:  
[CORE QUERY] [DATE] [FUNCTION MOD] [BREADTH TOKENS]

STEP 5️⃣ — Final Review
Review the assembled query inside your `<think>` block before outputting the final result.

-------------------------------------------------
EXAMPLES

User: “Latest news on EU AI act”

<think>
STEP 0: User mentioned \"Latest\", so I'll add the current date: `2025 7 21`.
STEP 1: Core query is \"EU AI Act\".
STEP 2: This is news, so I'll add news sources.
STEP 3: Breadth tokens: NYTimes, Politico, Verge.
STEP 4: Assembling: `EU AI Act 2025 7 21 NYTimes,Politico,Verge`
STEP 5: The query looks good. It's specific and targets relevant sources.
</think>
EU AI Act 2025 7 21 NYTimes,Politico,Verge

-------------------------------------------------
END
dd «NYTimes OR SCMP OR Reuters etc» (pick the most relevant subset).

STEP 3️⃣ — Breadth Tokens  
Append 2-4 comma-separated authoritative / crowd-sourced sources or high-coverage keywords to guarantee wide recall.

STEP 4️⃣ — Assembly (No Spaces After Commas)  
Assemble the parts in this order:  
[CORE QUERY] [DATE] [FUNCTION MOD] [BREADTH TOKENS]

STEP 5️⃣ — Output  
Return *ONLY* the assembled string, no explanation.

-------------------------------------------------
EXAMPLES

User: “Latest news on EU AI act” →  
`EU AI Act 2025 7 19 NYTimes,Politico,Verge`

User: “Best phone deals July 2025” →  
`Phone Deals July 2025 price Reddit,BestBuy,Amazon,RedFlagDeals`

User: “Craigslist Vancouver furniture for sale under 100 CAD” →  
`Craigslist Vancouver furniture under 100 CAD price Reddit,Facebook Marketplace`

-------------------------------------------------
END
"""

QUICK_SUMMARIZER_PROMPT = "Based ONLY on the following context, organise and cleaup the information so that a person can answer the user's request based off these."

COMPLETE_DECIDER_PROMPT = """You are a meticulous fact-checker and research analyst. Your job is to evaluate if the research is truly complete by comparing the collected context against the user's original, multi-part query.
**RULES:**
- If even ONE item from your checklist is missing from the context, you MUST respond with CONTINUE.
- Only respond with FINISH if you can confidently answer EVERY part of the user's query using ONLY the provided context.
"""

COMPLETE_NOTES_PROMPT = """You are a senior research strategist. Your job is to analyze the research gathered so far and create a 'Research Analysis & Plan' to guide the next steps.

1. **Checklist**: Create a checklist of all parts of the user's original request and mark which items have been found in the 'Current Research Context'.
2. **Analysis**: For the missing items, write a brief analysis of what information is still needed.
3. **Plan**: Outline a high-level plan for what to search for next."""

COMPLETE_Q_GEN_PROMPT_TEMPLATE = """You are a search query generator. Based on the provided 'Research Analysis & Plan', your job is to create a JSON object with a list of search queries to find the missing information.

IMPORTANT: If the query seems time-sensitive (e.g., 'latest'), add the current date ({date_str}).
Examples of what you should do: (User input): Latest News --> Latest News ({date_str}) NY Times, South China Morning Post, DW News || (User input): Best Phone deals --> Phone Deals ({date_str}) Reddit, Bestbuy, Amazon, Red Flag Deals
Don't copy excatly what the example says, but follow the sprit of it! Ensure for whatever you are searching the user gets the widest view of that topic, across the best sources.
Generate a JSON object with a 'queries' key, containing a list of {query_count} new, distinct search queries based *only* on the plan. Example: `{{"queries": ["query 1", "query 2"]}}`"""

SYNTHESIS_DECIDER_PROMPT = """You are an output formatting assistant. Based on the user's original request, should the collected research be SYNTHESIZED into a coherent answer, or should the raw text be returned (RETURN_RAW) for the user to review in full?
- For requests asking the model to 'learn', 'read', 'get documentation', or other similar tasks that imply a need for full context, choose RETURN_RAW.
- For requests asking 'what is', 'compare', 'explain', or other questions that require a formulated answer, choose SYNTHESIZE.
"""

COMPLETE_SUMMARIZER_PROMPT = """You are an expert synthesizer. Your task is to provide a comprehensive, well-structured answer to the user's original question based on the provided research context and the agent's own notes.
Use the agent's notes to understand what was considered important, and use the full context to pull the details. First, analyze the user's question to determine the best format for the answer (e.g., a brief summary, a detailed step-by-step guide, a comparative analysis, etc.). Then, formulate your response in that format."""


# ─── Constants & Helpers ──────────────────────────────────────────────────────

_CYAN = "\x1b[96m"
_MAGENTA = "\x1b[95m"
_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
URL_RE = re.compile(r"https?://\S+")


def _debug(msg: str) -> None:
    """Lightweight stderr logger."""
    print(
        f"{_MAGENTA}{_BOLD}[SearchRouterTool]{_RESET}{_CYAN} {msg}{_RESET}",
        file=sys.stderr,
    )


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
    max_retries: int = 3, delay: int = 3, **kwargs: Any
) -> Dict[str, Any]:
    """
    A wrapper for generate_chat_completion that includes a retry mechanism.
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            result = await generate_chat_completion(**kwargs)
            return result
        except Exception as e:
            last_exception = e
            _debug(
                f"⚠️ LLM call failed on attempt {attempt + 1}/{max_retries}. Retrying in {delay} seconds..."
            )
            await asyncio.sleep(delay)

    _debug(f"❌ LLM call failed after {max_retries} retries.")
    raise last_exception


# Debug Report Dataclasses


@dataclass
class QuickDebugReport:
    """A structured report for debugging the QUICK search process."""

    initial_query: str = ""
    refined_query: str = ""
    urls_found: List[str] = field(default_factory=list)
    urls_crawled: List[str] = field(default_factory=list)
    final_prompt: str = ""
    final_output: str = ""

    def format_report(self) -> str:
        report_parts = [
            "\n\n" + "=" * 25 + " QUICK SEARCH DEBUG REPORT " + "=" * 25,
            f"INITIAL USER QUERY: {self.initial_query}",
            f"REFINED SEARCH QUERY: {self.refined_query}",
            "-" * 80,
            f"URLs Found ({len(self.urls_found)}): {self.urls_found}",
            f"URLs Crawled ({len(self.urls_crawled)}): {self.urls_crawled}",
            f"Final Summarizer Prompt:\n{self.final_prompt[:1000]}...",
            "-" * 80,
            f"Final Output:\n{self.final_output[:500]}...",
            "=" * 80 + "\n",
        ]
        return "\n".join(report_parts)


@dataclass
class CompleteDebugReport:
    """A structured report for debugging the COMPLETE search process."""

    initial_user_query: str = ""
    refined_initial_query: str = ""
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    final_decision: str = ""
    final_payload: str = ""
    final_output: str = ""

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
        report_parts = [
            "\n\n" + "=" * 25 + " COMPLETE SEARCH DEBUG REPORT " + "=" * 25,
            f"INITIAL USER QUERY: {self.initial_user_query}",
            f"REFINED SEARCH QUERY: {self.refined_initial_query}",
            "-" * 80,
        ]
        for iter_data in self.iterations:
            iteration_num = iter_data.get("iteration", "N/A")
            report_parts.append(f"ITERATION {iteration_num}:")

            if iteration_num == 0:
                if iter_data.get("searches"):
                    initial_search = iter_data["searches"][0]
                    query = initial_search.get("query", "N/A")
                    crawled_urls = initial_search.get("crawled_urls", [])
                    report_parts.append(f'  [INITIAL SEARCH] Query: "{query}"')
                    report_parts.append(
                        f"  [INITIAL CRAWL] Crawled URLs: {crawled_urls}"
                    )
            else:
                decision = iter_data.get("continue_decision", "N/A")
                notes = iter_data.get("reasoning_notes", "")
                queries = iter_data.get("generated_queries", [])

                report_parts.append(f"  [DECIDER] Decision to continue: {decision}")
                if notes:
                    report_parts.append(f"  [REASONING NOTES]\n{notes}")
                if queries:
                    report_parts.append(
                        f"  [Q-GEN] Generated Queries for this iteration: {queries}"
                    )

                for search_data in iter_data.get("searches", []):
                    query = search_data.get("query", "N/A")
                    crawled_urls = search_data.get("crawled_urls", [])
                    report_parts.append(f'    - [SEARCH] Executed query: "{query}"')
                    report_parts.append(f"    - [CRAWL] Crawled URLs: {crawled_urls}")

            report_parts.append("-" * 80)

        report_parts.append("FINAL ACTION:")
        report_parts.append(f"  - Final Output Decision: {self.final_decision}")
        if self.final_decision == "SYNTHESIZE":
            report_parts.append(
                f"  - Payload sent to Summarizer Model:\n{self.final_payload[:1000]}..."
            )
        else:
            report_parts.append(
                f"  - Raw context returned to user:\n{self.final_payload[:1000]}..."
            )
        report_parts.append("=" * 80 + "\n")
        return "\n".join(report_parts)


# Valves
class Tools:

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

    def __init__(self) -> None:
        self.valves = self.Valves()
        self._exa: Optional[Exa] = None

    def _exa_client(self) -> Exa:
        if self._exa is None:
            if Exa is None:
                raise RuntimeError("exa_py not installed")
            key = self.valves.exa_api_key or os.getenv("EXA_API_KEY")
            if not key:
                raise RuntimeError("Exa API key missing")
            self._exa = Exa(key)
            _debug("🔑 Exa client initialised")
        return self._exa

    # Main
    async def routed_search(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __request__: Optional[Any] = None,
        __user__: Optional[Dict] = None,
        __messages__: Optional[List[Dict]] = None,
    ) -> dict:
        async def _status(desc: str, done: bool = False) -> None:
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": desc, "done": done}}
                )

        messages = __messages__ or []
        last_user_message = get_last_user_message(messages)
        if not last_user_message:
            return {
                "content": "Could not find a user message to process. Please try again.",
                "show_source": False,
            }

        # Build conversation history snippet for context
        history_messages = messages[-6:-1]
        convo_snippet_parts = []
        for m in history_messages:
            text_content = _get_text_from_message(m.get("content", ""))
            role = m.get("role", "").upper()
            convo_snippet_parts.append(f"{role}: {text_content!r}")
        convo_snippet = "\n".join(convo_snippet_parts)

        # The definitive query for the router now includes history
        router_query = f"Conversation History:\n{convo_snippet}\n\nLatest User Query:\n'{last_user_message}'"

        user_obj = Users.get_user_by_id(__user__["id"]) if __user__ else None
        _debug(f"🟢 Router triggered – full query context:\n{router_query}")
        await _status("Deciding search strategy…")

        router_payload = {
            "model": self.valves.router_model,
            "messages": [
                {"role": "system", "content": SEARCH_STRATEGY_PROMPT},
                {"role": "user", "content": router_query},  # Use the full context query
            ],
            "stream": False,
        }
        try:
            res = await generate_with_retry(
                request=__request__, form_data=router_payload, user=user_obj
            )
            llm_response_text = res["choices"][0]["message"]["content"]
            _debug(f"Search strategy router full response:\n{llm_response_text}")

            decision = ""
            for line in llm_response_text.splitlines():
                if line.lower().strip().startswith("final answer:"):
                    decision = line.split(":", 1)[1].strip().upper()
                    break

            if not decision:
                _debug("⚠️ Could not parse 'Final Answer:'. Defaulting to STANDARD.")
                decision = "STANDARD"

        except Exception as exc:
            _debug(f"⚠️ Router LLM failed after retries: {exc}. Defaulting to STANDARD.")
            decision = "STANDARD"

        _debug(f"🤔 Router decision → {decision}")
        exa = self._exa_client()

        # Mode 1 - Crawl
        if decision == "CRAWL":
            urls = URL_RE.findall(last_user_message)
            if not urls:
                return {
                    "content": "You requested a crawl, but I could not find a URL in your message. Please provide a valid URL.",
                    "show_source": False,
                }

            url_to_crawl = urls[0]
            await _status("Reading content from URL...")
            _debug(f"Executing CRAWL on: {url_to_crawl}")
            try:
                crawled_results = exa.get_contents([url_to_crawl])
                content = (
                    crawled_results.results[0].text
                    if crawled_results.results
                    else "Could not retrieve any text content from the URL."
                )
                await _status("Crawl complete.", done=True)
                return {
                    "content": f"## Content from {url_to_crawl}:\n\n{content}",
                    "show_source": False,
                }
            except Exception as e:
                _debug(f"💥 Crawl failed: {e}")
                return {
                    "content": f"I failed while trying to crawl the URL: {e}",
                    "show_source": False,
                }

        # Mode 2 - Standard
        elif decision == "STANDARD":
            report = QuickDebugReport(initial_query=last_user_message)
            final_result = ""
            context = ""
            try:
                await _status("Formulating search plan...")

                refiner_user_prompt = f"## Conversation History:\n{convo_snippet}\n\n## User's Latest Query:\n'{last_user_message}'"
                refiner_payload = {
                    "model": self.valves.quick_search_model,
                    "messages": [
                        {"role": "system", "content": SQR_PROMPT_TEMPLATE},
                        {"role": "user", "content": refiner_user_prompt},
                    ],
                    "stream": False,
                }

                res = await generate_with_retry(
                    request=__request__, form_data=refiner_payload, user=user_obj
                )
                llm_response_text = res["choices"][0]["message"]["content"].strip()
                _debug(f"SQR full response:\n{llm_response_text}")

                # Extract the last line after the </think> block
                if "</think>" in llm_response_text:
                    # Take the content after the last </think> and strip whitespace
                    refined_query = llm_response_text.split("</think>")[-1].strip()
                else:
                    # Fallback for models that might not follow instructions perfectly
                    _debug(
                        "⚠️ SQR response did not contain a <think> block. Falling back to last line."
                    )
                    refined_query = llm_response_text.splitlines()[-1].strip()
                _debug(f"Refined STANDARD query: {refined_query}")
                report.refined_query = refined_query

                await _status(f'Searching for: "{refined_query}"')
                search_data = exa.search(
                    refined_query,
                    num_results=self.valves.quick_urls_to_search,
                    use_autoprompt=True,
                )
                report.urls_found = [res.url for res in search_data.results]

                crawl_candidates = search_data.results[
                    : self.valves.quick_queries_to_crawl
                ]

                if not crawl_candidates:
                    final_result = "My search found no results to read. Please try a different query."
                else:
                    domains = [
                        urlparse(res.url).netloc.replace("www.", "")
                        for res in crawl_candidates
                    ]
                    await _status(f"Reading from: {', '.join(domains)}")

                    ids_to_crawl = [res.id for res in crawl_candidates]
                    report.urls_crawled = [res.url for res in crawl_candidates]
                    crawled_results = exa.get_contents(ids_to_crawl)

                    await _status(
                        f"Synthesizing answer from {len(crawled_results.results)} sources..."
                    )
                    context = "\n\n".join(
                        [
                            f"## Source: {res.url}\n\n{res.text}"
                            for res in crawled_results.results
                        ]
                    )
                    context = context[: self.valves.quick_max_context_chars]

                    summarizer_user_prompt = f"## Context:\n{context}\n\n## User's Question:\n{last_user_message}"
                    report.final_prompt = f"SYSTEM: {QUICK_SUMMARIZER_PROMPT}\nUSER: {summarizer_user_prompt}"
                    summarizer_payload = {
                        "model": self.valves.quick_search_model,
                        "messages": [
                            {"role": "system", "content": QUICK_SUMMARIZER_PROMPT},
                            {"role": "user", "content": summarizer_user_prompt},
                        ],
                        "stream": False,
                    }
                    final_res = await generate_with_retry(
                        request=__request__, form_data=summarizer_payload, user=user_obj
                    )
                    final_result = final_res["choices"][0]["message"]["content"]
                    await _status("Standard search complete.", done=True)

            except Exception as e:
                _debug(f"💥 STANDARD search path failed with an exception: {e}")
                if context:
                    final_result = f"I found some information but encountered an error while processing it. Here is the raw data I gathered:\n\n{context}"
                else:
                    final_result = f"I failed during the standard search: {e}"
            finally:
                report.final_output = final_result
                _debug(report.format_report())
                return {"content": final_result, "show_source": False}

        # Mode 3 - Complete
        elif decision == "COMPLETE":
            report = CompleteDebugReport(initial_user_query=last_user_message)
            notepad = {}
            final_result = ""
            search_notes = ""
            try:
                await _status("Initiating deep research...")

                refiner_user_prompt = f"## Conversation History:\n{convo_snippet}\n\n## User's Latest Query:\n'{last_user_message}'"
                refiner_payload = {
                    "model": self.valves.complete_agent_model,
                    "messages": [
                        {"role": "system", "content": SQR_PROMPT_TEMPLATE},
                        {"role": "user", "content": refiner_user_prompt},
                    ],
                    "stream": False,
                }

                res = await generate_with_retry(
                    request=__request__, form_data=refiner_payload, user=user_obj
                )
                llm_response_text = res["choices"][0]["message"]["content"].strip()
                _debug(f"SQR full response:\n{llm_response_text}")

                # Extract the last line after the </think> block
                if "</think>" in llm_response_text:
                    # Take the content after the last </think> and strip whitespace
                    refined_query = llm_response_text.split("</think>")[-1].strip()
                else:
                    # Fallback for models that might not follow instructions perfectly
                    _debug(
                        "⚠️ SQR response did not contain a <think> block. Falling back to last line."
                    )
                    refined_query = llm_response_text.splitlines()[-1].strip()
                _debug(f"Refined query: {refined_query}")
                report.refined_initial_query = refined_query

                await _status(f'Initial search for: "{refined_query}"')
                broad_search_data = exa.search(
                    refined_query,
                    num_results=self.valves.complete_urls_to_search_per_query,
                    use_autoprompt=True,
                )
                ids_to_crawl = [
                    res.id
                    for res in broad_search_data.results[
                        : self.valves.complete_queries_to_crawl
                    ]
                ]
                if not ids_to_crawl:
                    raise RuntimeError(
                        "Initial search did not yield any results to analyze."
                    )
                crawled_results = exa.get_contents(ids_to_crawl)
                crawled_urls = [res.url for res in crawled_results.results]
                report.add_iteration(0, "START", "", [])
                report.add_search_to_iteration(0, refined_query, crawled_urls)

                for res in crawled_results.results:
                    if res.url not in notepad:
                        notepad[res.url] = (
                            f"## Content from '{res.title}' ({res.url}):\n{' '.join(res.text.split())}"
                        )
                _debug(f"Notepad initialized with {len(notepad)} sources.")

                for i in range(self.valves.complete_max_search_iterations):
                    iteration_num = i + 1
                    await _status(
                        f"Analyzing findings (Pass {iteration_num}/{self.valves.complete_max_search_iterations})..."
                    )
                    current_context = "\n\n---\n\n".join(notepad.values())

                    if i == 0:
                        _debug("First iteration, forcing continuation.")
                        research_decision = "CONTINUE"
                    else:
                        _debug("Deciding whether to continue research...")
                        decider_user_prompt = f"## User's Question:\n{last_user_message}\n\n## Current Research Context:\n{current_context}\n\nIs every part of the user's question answered in the context? Reply with the single word CONTINUE or FINISH."
                        decider_payload = {
                            "model": self.valves.complete_agent_model,
                            "messages": [
                                {"role": "system", "content": COMPLETE_DECIDER_PROMPT},
                                {"role": "user", "content": decider_user_prompt},
                            ],
                            "stream": False,
                        }
                        res = await generate_with_retry(
                            request=__request__,
                            form_data=decider_payload,
                            user=user_obj,
                        )
                        research_decision = (
                            res["choices"][0]["message"]["content"].strip().upper()
                        )
                        _debug(f"Research decision: {research_decision}")

                    if research_decision == "FINISH":
                        report.add_iteration(
                            iteration_num, research_decision, search_notes, []
                        )
                        _debug("Decision made to finalize the answer.")
                        break

                    await _status("Planning next steps...")
                    notes_user_prompt = f"## User's Original Request:\n{last_user_message}\n\n## Current Research Context:\n{current_context}\n\nProvide your 'Research Analysis & Plan'."
                    notes_payload = {
                        "model": self.valves.complete_agent_model,
                        "messages": [
                            {"role": "system", "content": COMPLETE_NOTES_PROMPT},
                            {"role": "user", "content": notes_user_prompt},
                        ],
                        "stream": False,
                    }
                    res = await generate_with_retry(
                        request=__request__, form_data=notes_payload, user=user_obj
                    )
                    search_notes = res["choices"][0]["message"]["content"]
                    _debug(f"Search Notes:\n{search_notes}")

                    q_gen_sys_prompt = COMPLETE_Q_GEN_PROMPT_TEMPLATE.format(
                        date_str=f"{datetime.now().year} {datetime.now().month} {datetime.now().day}",
                        query_count=self.valves.complete_queries_to_generate,
                    )
                    q_gen_user_prompt = f"## Research Analysis & Plan:\n{search_notes}"
                    q_gen_payload = {
                        "model": self.valves.complete_agent_model,
                        "messages": [
                            {"role": "system", "content": q_gen_sys_prompt},
                            {"role": "user", "content": q_gen_user_prompt},
                        ],
                        "response_format": {"type": "json_object"},
                    }

                    res = await generate_with_retry(
                        request=__request__, form_data=q_gen_payload, user=user_obj
                    )
                    raw_q_gen_response = res["choices"][0]["message"]["content"]
                    clean_response = (
                        raw_q_gen_response.strip()
                        .removeprefix("```json")
                        .removesuffix("```")
                        .strip()
                    )

                    parsed_json = json.loads(clean_response)
                    if isinstance(parsed_json, dict) and "queries" in parsed_json:
                        targeted_queries = parsed_json.get("queries", [])
                    elif isinstance(parsed_json, list):
                        targeted_queries = parsed_json
                    else:
                        targeted_queries = []

                    if targeted_queries:
                        targeted_queries = [
                            str(q) for q in targeted_queries if isinstance(q, str)
                        ]

                    _debug(f"Generated new targeted queries: {targeted_queries}")
                    report.add_iteration(
                        iteration_num, research_decision, search_notes, targeted_queries
                    )

                    if not targeted_queries:
                        continue

                    for t_query in targeted_queries:
                        await _status(f'Following new lead: "{t_query[:50]}..."')
                        search_results = exa.search(
                            t_query,
                            num_results=self.valves.complete_urls_to_search_per_query,
                            use_autoprompt=True,
                        )
                        ids_to_crawl = [
                            res.id
                            for res in search_results.results[
                                : self.valves.complete_queries_to_crawl
                            ]
                        ]
                        if not ids_to_crawl:
                            continue

                        crawled_results = exa.get_contents(ids_to_crawl)
                        crawled_urls = [res.url for res in crawled_results.results]
                        report.add_search_to_iteration(
                            iteration_num, t_query, crawled_urls
                        )
                        for res in crawled_results.results:
                            if res.url not in notepad:
                                notepad[res.url] = (
                                    f"## Content from '{res.title}' ({res.url}):\n{' '.join(res.text.split())}"
                                )
                        _debug(f"Notepad now contains {len(notepad)} sources.")

                await _status(
                    f"Compiling final report from {len(notepad)} gathered sources..."
                )
                final_context = "\n\n---\n\n".join(notepad.values())
                if not final_context:
                    raise RuntimeError(
                        "Unable to gather any information during research."
                    )

                synthesis_decider_user_prompt = f"User's original request: '{last_user_message}'\n\nRespond with a single word: SYNTHESIZE or RETURN_RAW."
                synthesis_decider_payload = {
                    "model": self.valves.complete_agent_model,
                    "messages": [
                        {"role": "system", "content": SYNTHESIS_DECIDER_PROMPT},
                        {"role": "user", "content": synthesis_decider_user_prompt},
                    ],
                    "stream": False,
                }
                res = await generate_with_retry(
                    request=__request__,
                    form_data=synthesis_decider_payload,
                    user=user_obj,
                )
                final_action = res["choices"][0]["message"]["content"].strip().upper()
                _debug(f"Final action decision: {final_action}")
                report.final_decision = final_action

                if final_action == "SYNTHESIZE":
                    await _status("Synthesizing final answer...")
                    summarizer_user_prompt = f"## User's Original Question:\n{last_user_message}\n\n## Agent's Research Notes:\n{search_notes}\n\n## Full Research Context (Your Notepad):\n{final_context}"
                    summarizer_payload = {
                        "model": self.valves.complete_summarizer_model,
                        "messages": [
                            {"role": "system", "content": COMPLETE_SUMMARIZER_PROMPT},
                            {"role": "user", "content": summarizer_user_prompt},
                        ],
                    }
                    final_summary_response = await generate_with_retry(
                        request=__request__, form_data=summarizer_payload, user=user_obj
                    )
                    final_result = final_summary_response["choices"][0]["message"][
                        "content"
                    ]
                    report.final_payload = f"SYSTEM: {COMPLETE_SUMMARIZER_PROMPT}\nUSER: {summarizer_user_prompt}"
                else:  # RETURN_RAW
                    _debug("Returning raw context as requested.")
                    final_result = f"I have gathered the following raw information based on your request:\n\n---\n\n{final_context}"
                    report.final_payload = final_context

                await _status("Research complete.", done=True)

            except Exception as e:
                _debug(f"💥 COMPLETE search path failed with an exception: {e}")
                if notepad:
                    final_context = "\n\n---\n\n".join(notepad.values())
                    final_result = f"The research process encountered an error. Here are the notes I gathered before the issue occurred:\n\n{final_context}"
                else:
                    final_result = f"I failed during the COMPLETE research process: {e}"
            finally:
                report.final_output = final_result
                _debug(report.format_report())
                return {"content": final_result, "show_source": False}

        return {
            "content": f"Router chose '{decision}', but no corresponding action was taken.",
            "show_source": False,
        }


# Final tool definition
class ExaSearch:
    def __init__(self):
        self.tools = Tools()
        self.valves = self.tools.valves

    def __call__(
        self,
        query: str,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
        __request__: Optional[Any] = None,
        __user__: Optional[Dict] = None,
        __messages__: Optional[List[Dict]] = None,
    ) -> dict:
        return asyncio.run(
            self.tools.routed_search(
                query, __event_emitter__, __request__, __user__, __messages__
            )
        )
