# OpenWebUI Auto Tool Selector Suite

[![Version](https://img.shields.io/badge/version-1.2.5-blue.svg)](https://github.com/ShaoRou459/OpenwebUI-Tooling-Setup/releases)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![OpenWebUI](https://img.shields.io/badge/OpenWebUI-compatible-orange.svg)](https://openwebui.com)
[![License](https://img.shields.io/badge/license-GPL-red.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/ShaoRou459/OpenwebUI-Tooling-Setup?style=social)](https://github.com/ShaoRou459/OpenwebUI-Tooling-Setup/stargazers)

Intelligent tool routing and autonomous AI capabilities for OpenWebUI.

[中文 Readme](https://github.com/ShaoRou459/OpenwebUI-Tooling-Setup/blob/main/README_zh.md) | [Quick Start](#installation--setup) | [Configuration Guide](#configuration)

---

## Overview

The OpenWebUI Auto Tool Selector Suite turns models into proactive assistants with autonomous tool use, high-quality search, and code execution. Version 1.2.5 adds structured debug metrics, iterative research, safer concurrency, and better vision support.

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🤖 Autonomous Tool Selection | Routes each query to the best tool (search, code, image, memory) using a helper model |
| 🔍 Research Engine | Three modes: CRAWL (read a URL), STANDARD (quick multi-source), COMPLETE (iterative deep research) |
| 🎨 Image Generation | Optimizes prompts and embeds the generated image back into the chat |
| 💻 Code Execution | Toggle between Jupyter notebook or lightweight Python execution per turn |
| 👁️ Universal Vision | Adds image understanding to non-vision models via automatic descriptions/injection |
| 🛡️ Concurrency Safety | Per-user per-query locking to avoid duplicate searches; cache for repeated queries |
| 🧭 Exa Integration & Fallback | Uses Exa when available; clean fallback to OpenWebUI default web_search |
| 🧪 Structured Debug Metrics | Colorized logs, timing, LLM call metrics, URL stats, and end-of-run summaries |
| ⚡ Real-Time Status | Live progress updates through the OpenWebUI status channel |

---

## Architecture

Two composable parts:

1) Auto Tool Selector (Function) — entrypoint router that analyzes conversation context and selects a single tool or none.
2) Exa Search Router (Tool) — advanced research with CRAWL/STANDARD/COMPLETE, robust retries, and synthesis. Falls back to OpenWebUI native web_search when Exa is not used.

![Architecture Diagram](https://github.com/user-attachments/assets/e79f7658-020f-4804-8d16-e4414ad781e8)

---

## Installation & Setup

### Prerequisites

- Docker access to your OpenWebUI container
- Optional: Exa.ai account and API key for enhanced search

### Step 1: Install Exa client in the container (one-time)

```bash
docker exec -it open-webui bash
pip install exa_py
exit
docker restart open-webui
```

### Step 2: Add the components

1) Auto Tool Selector (Function)
   - Admin Settings → Functions → New Function
   - Paste the contents of `auto_tool_selector.py`
   - Save
   - If you plan to use Jupyter as the code interpreter, download `jupyter_uploader.py`, save it into Jupyter’s home directory as `uploader.py`, and set your OpenWebUI `BASE_URL` and `TOKEN` inside it.

2) Exa Search Router (Tool) — optional but recommended
   - Workspace → Tools → New Tool
   - Paste the contents of `exa_router_search.py`
   - Tool ID must be `exa_router_search`
   - Save

### Step 3: Configure settings in UI

All configuration is done via valves in the UI—no file edits needed.

1) Enable the Function in your model
   - In the model’s Functions list, enable only Auto Tool Selector
   - Do NOT enable the Exa tool directly; the router calls it when needed

2) Set preferences
   - In the valves, configure API keys, model IDs, search depth, debug, and Jupyter usage

---

## Configuration

### Auto Tool Selector Settings

| Setting | Purpose | Recommendation |
|---------|---------|----------------|
| `helper_model` | Decides which tool to use for queries | See model recommendations below |
| `vision_model` | Describes images to enrich non-vision models | See model recommendations below |
| `vision_injection_models` | Non-vision model IDs to enrich with image text | Add your model IDs (comma-separated) |
| `history_char_limit` | Max chars per prior message in history snippet | 500 (default) |
| `use_exa_router_search` | Enable advanced Exa search vs native search | `true` (if Exa tool is installed) |
| `debug_enabled` | Enable detailed debug logging | `false` (enable for troubleshooting) |
| `use_jupyter_code_interpreter` | Jupyter notebook vs basic code execution | `true` (recommended) |

### Exa Search Router Settings *(If Installed)*

| Setting | Purpose | Recommendation |
|---------|---------|----------------|
| `exa_api_key` | Exa.ai API key (or set `EXA_API_KEY` env) | Required for Exa mode |
| `router_model` | Decides CRAWL/STANDARD/COMPLETE | See model recommendations below |
| `quick_search_model` | Refine → crawl → summarize in STANDARD | See model recommendations below |
| `complete_agent_model` | Reasoning and planning in COMPLETE | See model recommendations below |
| `complete_summarizer_model` | Final high‑quality synthesis in COMPLETE | See model recommendations below |
| `quick_urls_to_search` | URLs to fetch in STANDARD | 5 (default) |
| `quick_queries_to_crawl` | Top results to crawl in STANDARD | 3 (default) |
| `quick_max_context_chars` | Context cap for STANDARD summarizer | 8000 (default) |
| `complete_urls_to_search_per_query` | URLs per generated query | 5 (default) |
| `complete_queries_to_crawl` | Top results to crawl per query | 3 (default) |
| `complete_queries_to_generate` | New queries per iteration | 3 (default) |
| `complete_max_search_iterations` | Iterations for COMPLETE | 2 (default) |
| `debug_enabled` | Enable detailed debug logs | `false` (enable for troubleshooting) |
| `show_sources` | Ask UI to display sources list | `false` (set `true` if desired) |

---

## Recommended Models (as of 2024‑10)

Pick per role; any provider is fine if it supports your IDs. If you need fresher 2025 picks, tell me and share links you want reflected.

- Helper/router (low cost, responsive):
  - GPT‑4o‑mini, Claude 3.5 Haiku, Gemini 1.5 Flash, Llama 3.1 8B/70B Instruct (local)
- Vision model (for image description):
  - GPT‑4o, Gemini 1.5 Pro/Flash, Claude 3.5 Sonnet‑Vision (if available)
- STANDARD search model (refine + summarize):
  - GPT‑4o‑mini, Claude 3.5 Haiku, Gemini 1.5 Flash, Llama 3.1 70B (local/server)
- COMPLETE agent model (reasoning/planning):
  - Claude 3.5 Sonnet, GPT‑4o, Gemini 1.5 Pro, Mistral Large 2
- COMPLETE summarizer (final synthesis, longer context helpful):
  - Claude 3.5 Sonnet, GPT‑4o, Gemini 1.5 Pro, Llama 3.1 70B

Local-friendly: Qwen2.5‑72B‑Instruct, Llama 3.1 70B, Mistral Large 2 (via API) are strong picks when budgets or privacy matter.

---

## Usage Examples

### Autonomous Tool Selection
```
User: "What's the latest news about AI developments today?"
→ Automatically routes to web search, finds current articles, synthesizes response

User: "Create a logo for my coffee shop called 'Morning Brew'"
→ Automatically routes to image generation, optimizes prompt, generates and displays image

User: "Analyze this sales data and create a visualization"
→ Automatically routes to code interpreter, processes data, creates charts
```

### Search Modes

- **Crawl Mode**: `"What does this article say? https://example.com/article"`
- **Standard Mode**: `"What are the benefits of renewable energy?"` (default for most queries)
- **Complete Mode**: `"Do a deep research comparison of React vs Vue.js frameworks"` (requires explicit request)

### Vision Enhancement
Non-vision models can now process images when you include them in your messages. The system automatically describes images and provides that context to the model.

---

## Troubleshooting

### Enable Debug Mode
Set `debug_enabled` to `true` in Auto Tool Selector and/or Exa Search Router valves to see detailed, colorized logs with metrics in your Docker container:

```bash
docker logs open-webui -f
```

### Common Issues

**Tool not activating**: Ensure only Auto Tool Selector is enabled for the model. Do not enable the Exa tool directly.

**Search failing**: If using Exa, verify `exa_py` is installed in the container and your Exa API key is set (valve or `EXA_API_KEY`). The system falls back to native `web_search` if Exa is unavailable.


**Concurrent search warning**: “A search is already in progress…” means a duplicate query was detected for this user; wait for completion or retry with a changed query.

---

## Update Log

### 1.2.5 (Current)
- Structured debug metrics (timing, LLM calls, URL stats) with end-of-run summaries
- Iterative COMPLETE research (objectives → reasoning → multi-queries → synthesis)
- Robust LLM response normalization and automatic correction prompts
- Exa search caching, chunked crawling, and safer concurrency (per-user/query locks)
- Vision injection for non-vision models; multi-image analysis with concurrency limits
- Code interpreter toggle (Jupyter vs basic); optional Jupyter file uploader helper
- Clean fallback to OpenWebUI native `web_search` when Exa is disabled
- Settings-only configuration for all behavior (no file edits)

### 1.0
- Initial release with autonomous tool routing
- Basic search, image generation, and code interpretation
- Manual configuration through separate files

---

## FAQ

**Q: Do I need the Exa Search Router tool?**
A: No, it's optional. The Auto Tool Selector will fall back to OpenWebUI's native search if Exa is not available.

**Q: Why use Jupyter over basic code interpreter?**
A: Jupyter provides a full notebook environment with file persistence, better for complex analysis and data work.

**Q: Can I use this with any OpenWebUI model?**
A: Yes, the Auto Tool Selector works with any model. Vision enhancement works best with models that support tool calling.

**Q: How do I know if tools are working?**
A: Enable debug mode and check Docker logs. You'll see detailed information about tool selection and execution.

---

## License & Support

- **Author**: ShaoRou459
- **GitHub**: [OpenwebUI-Tooling-Setup](https://github.com/ShaoRou459/OpenwebUI-Tooling-Setup)
- **Issues**: Report bugs and request features via GitHub Issues
- **Version**: 1.2.5

---

*Transform your AI from reactive to proactive with intelligent tool routing and autonomous capabilities.*
