# OpenWebUI Auto Tool Selector Suite

[![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)](https://github.com/ShaoRou459/OpenwebUI-Tooling-Setup/releases)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)
[![OpenWebUI](https://img.shields.io/badge/OpenWebUI-compatible-orange.svg)](https://openwebui.com)
[![License](https://img.shields.io/badge/license-GPL-red.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/ShaoRou459/OpenwebUI-Tooling-Setup?style=social)](https://github.com/ShaoRou459/OpenwebUI-Tooling-Setup/stargazers)

**Intelligent tool routing and autonomous AI capabilities for OpenWebUI**

📦 **OpenWebUI Marketplace**: [Auto Tool Router](https://openwebui.com/f/sdjfhsud/auto_tool_router) | [Exa Search Router](https://openwebui.com/t/sdjfhsud/exa_router_search)

[中文 Readme](https://github.com/ShaoRou459/OpenwebUI-Tooling-Setup/blob/main/README_zh.md) | [Quick Start](#installation--setup) | [Configuration Guide](#configuration)

---

## Overview

The OpenWebUI Auto Tool Selector Suite transforms your local AI models from passive chat interfaces into intelligent, autonomous assistants. Version 1.1 introduces enhanced debugging, flexible architecture, and improved accessibility for both vision and non-vision models.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **🤖 Autonomous Tool Selection** | Automatically routes user queries to the most appropriate tool without manual intervention |
| **🔍 Multi-Modal Search** | Three search modes: **Crawl** (specific URLs), **Standard** (quick research), **Complete** (deep analysis) |
| **🎨 Intelligent Image Generation** | Auto-optimizes prompts and seamlessly integrates generated images into conversations |
| **💻 Dual Code Execution** | Support for both Jupyter notebooks and basic Python code interpretation |
| **👁️ Universal Vision** | Non-vision models gain image understanding through automatic transcription |
| **🔧 Advanced Debugging** | Comprehensive logging system for troubleshooting and optimization |
| **⚡ Real-Time Status** | Live progress updates throughout tool execution |

---

## Architecture

The suite consists of two main components:

1. **Auto Tool Selector** (Function): Master router that analyzes queries and selects appropriate tools
2. **Exa Search Router** (Tool): Advanced search capabilities with fallback to native OpenWebUI search

![Architecture Diagram](https://github.com/user-attachments/assets/e79f7658-020f-4804-8d16-e4414ad781e8)

---

## Installation & Setup

### Prerequisites

Ensure you have Docker access to your OpenWebUI instance.

### Step 1: Install Dependencies

```bash
docker exec -it open-webui bash
pip install exa_py
exit
docker restart open-webui
```

### Step 2: Add the Components

1. **Install Auto Tool Selector (Function)**:
   - Go to **Admin Settings → Functions → New Function**
   - Copy and paste the contents of `auto_tool_selector.py`
   - Save the function
   - If you plan to use Jupyter Lab as the code interpreter, please download the uploader.py and place it in the home directory of Jupyter.

2. **Install Exa Search Router (Tool)** *(Optional)*:
   - Go to **Workspace → Tools → New Tool**
   - Copy and paste the contents of `exa_router_search.py`
   - **Important**: Set Tool ID to `exa_router_search`
   - Save the tool

### Step 3: Configure Settings

All configuration is now done through the UI settings - no manual file editing required!

1. **Enable the Function**:
   - In your model settings, enable only the **Auto Tool Selector** function
   - Do not enable the Exa Search Router tool directly

2. **Configure Your Preferences**:
   - Access function settings through the model interface
   - Configure API keys, models, and behavior options as needed

---

## Configuration

### Auto Tool Selector Settings

| Setting | Purpose | Recommendation |
|---------|---------|----------------|
| `helper_model` | Decides which tool to use for queries | OpenAI o4-mini, Gemini 2.5 Flash |
| `vision_model` | Analyzes images for non-vision models | OpenAI o3 / o4-mini, Gemini 2.5 Pro |
| `vision_injection_models` | List of non-vision models to enhance | Add your model IDs (comma-separated) |
| `use_exa_router_search` | Enable advanced Exa search vs native search | `true` (if Exa tool is installed) |
| `debug_enabled` | Enable detailed debug logging | `false` (enable for troubleshooting) |
| `use_jupyter_code_interpreter` | Use Jupyter vs basic code execution | `true` (recommended) |

### Exa Search Router Settings *(If Installed)*

| Setting | Purpose | Recommendation |
|---------|---------|----------------|
| `exa_api_key` | **Required**: Your Exa.ai API key | Get yours at [exa.ai](https://exa.ai) |
| `router_model` | Chooses search strategy (Crawl/Standard/Complete) | OpenAI o4-mini or Gemini 2.5 Flash |
| `quick_search_model` | Handles standard search operations | OpenAI o4-mini or Gemini 2.5 Flash |
| `complete_agent_model` | Powers deep research analysis | OpenAI o3/o3-pro, Gemini 2.5 Pro, Claude 3.7 Sonnet |
| `complete_summarizer_model` | Creates final comprehensive summaries | Gemini 2.5 Pro, OpenAI o4-mini |
| `debug_enabled` | Enable search operation debugging | `false` (enable for troubleshooting) |

---

## Recommended Models (as of 2025‑09)

Pick per role; choose any provider you already use. Links reference official model pages.

- Helper/router (fast, low cost):
  - OpenAI o4-mini — small, cost‑efficient reasoning with strong vision
  - Google Gemini 2.5 Flash — controllable “thinking” budget, great latency
- Vision model (image/visual reasoning for non‑vision chats):
  - OpenAI o3 or o4-mini — tool‑using multimodal reasoning
  - Google Gemini 2.5 Pro — long‑context, native multimodality
  - Meta Llama 4 Maverick/Scout (open‑weight) — strong multimodal open models
- COMPLETE agent (deep research/planning/coding/math):
  - OpenAI o3 (or o3‑pro) — frontier reasoning and tool use
  - Google Gemini 2.5 Pro — long‑context “thinking” model
  - Anthropic Claude 3.7 Sonnet — hybrid reasoning with extended thinking
  - DeepSeek R1‑0528 (open‑weight) — strong math/coding reasoning
- COMPLETE summarizer (final synthesis):
  - Gemini 2.5 Pro (very long context)
  - OpenAI o4‑mini (fast, economical)
  - Llama 4 Scout (open‑weight)

Open‑/self‑host friendly picks: Meta Llama 4 Maverick/Scout and DeepSeek R1‑0528 (MIT‑licensed weights) are excellent when you need local or budget‑conscious deployments.

Sources:
- OpenAI o3 / o4‑mini (Apr 16, 2025) and o3‑pro (Jun 10, 2025): https://openai.com/index/introducing-o3-and-o4-mini/ and system card https://openai.com/index/o3-o4-mini-system-card
- Google Gemini 2.5 Pro (GA Jun 17, 2025): https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro and report (Jun 16, 2025): https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf
- Anthropic Claude 3.7 Sonnet (Feb 24, 2025): https://www.anthropic.com/news/claude-3-7-sonnet
- Meta Llama 4 (Apr 5, 2025): https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- DeepSeek R1‑0528 (May 28, 2025): https://api-docs.deepseek.com/news/news250528

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
Set `debug_enabled` to `true` in your function/tool settings to see detailed logs in your Docker container:

```bash
docker logs open-webui -f
```

### Common Issues

**Tool not activating**: Check that only the Auto Tool Selector function is enabled in model settings, not the individual tools.

**Search failing**: If using Exa search, verify your API key is set correctly. The system will fall back to native search if Exa is unavailable.

**Vision not working**: Ensure `vision_model` is set and your model ID is listed in `vision_injection_models`.

---

## Update Log

### Version 1.1 (Current)
- **New**: Enhanced debugging system with color-coded logging
- **New**: Vision model integration for non-vision models
- **New**: Modular search architecture with native OpenWebUI fallback
- **New**: Choice between Jupyter and basic code interpreters
- **New**: Settings-based configuration (no more manual file management)
- **Improved**: More robust error handling and retry mechanisms
- **Improved**: Better status updates and user feedback

### Version 1.0
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
- **Version**: 1.1.0

---

*Transform your AI from reactive to proactive with intelligent tool routing and autonomous capabilities.*
