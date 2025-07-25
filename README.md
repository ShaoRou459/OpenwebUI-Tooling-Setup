# OpenWebUI Tooling Suite Documentation  
[中文 Readme](https://github.com/ShaoRou459/OpenwebUI-Tooling-Setup/blob/main/README_zh.md) | Skip to [Installation & Usage](#4-installation--usage)

---

## 1. Background & Goals  
Standard OpenWebUI models are pretty “passive” by default:  
- You have to manually pick and click tools every time  
- Search results often fall short  

**This project gives your local model “autopilot”** by adding a two-layer intelligent router that automatically decides whether to:  
- Search the web  
- Generate images  
- Run code  

…all while matching the online, drawing, and code-execution prowess of ChatGPT/Gemini. Future versions will hook in even more tools.

---

## 2. Features Overview  

| Feature                | Description                                                                                                           |
|------------------------|-----------------------------------------------------------------------------------------------------------------------|
| **Autonomous Routing** | Every user prompt is intercepted by the main router and automatically routed to the right tool—no manual clicks.     |
| **Three Search Modes** | Powered by Exa.ai:  
• **Crawl**: read/process a specified URL directly  
• **Quick**: summarize top results in ~5 s  
• **Complete**: deep, multi-step research + final synthesis |
| **Intelligent Imaging**| Auto-optimize prompts for higher-quality pictures, then feed the generated image back into the conversation.         |
| **Code Interpreter**   | Native Python interpreter + Jupyter support, with convenient file-download links.                                      |
| **Real-Time Status**   | Live progress updates at every step, so you’re never left hanging.                                                    |

<video src="https://private-user-images.githubusercontent.com/212266166/468440945-696cb316-c160-4210-a0dc-f87a04be1647.mp4" controls muted loop style="max-width:100%;"></video>

---

## 3. Configuration Guide: Valves & Model Selection  
All behavior is tuned via **Valves**, so you can balance speed, cost, and “smarts.”

### 3.1 Auto Tool Selector (Master Router) Valves  

| Valve                     | Purpose                                            | Recommendation                 |
|---------------------------|----------------------------------------------------|--------------------------------|
| `helper_model`            | Decide whether to call a tool or just chat         | GPT-4.1-mini / Llama3-8B       |
| `vision_model`            | Analyze images and describe them                   | GPT-4.1 / Gemini 2.0 Flash     |
| `vision_injection_models` | Models that consume image annotations (non-vision) | Add model id from OWUI use comma (,) to separate       |
| `history_char_limit`      | Max chars from history to send (saves tokens)      | default                        |

### 3.2 Exa Search Router Valves  

| Valve                      | Purpose                                                      | Recommendation                                |
|----------------------------|--------------------------------------------------------------|-----------------------------------------------|
| `exa_api_key`              | **Required**: Your Exa.ai API key                            | Apply at https://exa.ai                       |
| `router_model`             | Chooses **Crawl/Quick/Complete** internally                   | GPT-4.1-mini                                  |
| `quick_search_model`       | Used for **Quick** (refine + summarize)                       | GPT-4.1-mini / Gemini 2.0 Flash               |
| `complete_agent_model`     | “Brain” for **Complete** (planning, querying, analyzing)      | GPT-4.1 / Gemini 2.0 Flash / Claude 4 Sonnet  |
| `complete_summarizer_model`| Final synthesis in **Complete** path                          | Gemini 2.5 Flash / Llama 4 Maverick           |

#### Search-Depth Parameters (optional, sensible defaults)  
- **Quick**:  
    - `quick_urls_to_search`  
    - `quick_queries_to_crawl`  
    - `quick_max_context_chars`  
- **Complete**:  
    - `complete_urls_to_search_per_query`  
    - `complete_queries_to_crawl`  
    - `complete_queries_to_generate`  
    - `complete_max_search_iterations`  

---

## 4. Installation & Usage  

### Step 0: Choose Your Router File  
- **Native interpreter** → `auto_tool_selector_default_code_interpter.py`  
- **Jupyter backend** → `auto_tool_selector_jupyter_ci.py`  

### Step 1 (Jupyter Only): Enable File Downloads  
Place `uploader.py` in the **root folder** where your Jupyter server launches. Without it, the AI can’t return download links.

### Step 2: Install Dependencies  
Run once on the Docker host:

    docker exec -it open-webui bash
    pip install exa_py
    exit
    docker restart open-webui

### Step 3: Configure in OpenWebUI Admin  
1. **Add the Master Router (Function):**  
   Admin Settings → Functions → New Function → paste your chosen Auto Tool Selector code.  
2. **Add the Search Router (Tool):**  
   Workspace → Tools → New Tool → paste the Exa Search Router code  
   → **Tool ID must be** `exa_router_search`.  
3. **Set Your API Key:**  
   In the Exa Search Router’s Valves, fill in `exa_api_key`.

### Step 4: Enable the Suite  
In **only** the target model’s settings, enable **just** the `Auto Tool Selector` function.  
_Do not_ enable `Exa Search Router` directly—this would bypass the master router.

---

## 5. FAQ  
**Q1: Why install `exa_py` manually?**  
OpenWebUI won’t auto-install libraries. The “Requirements” comment in the tool code is for documentation only—you must pip-install inside the container.  

**Q2: What’s the difference between a Function and a Tool?**  
- **Function:** Runs _before_ the model reply to intercept + reroute (our Auto Tool Selector).  
- **Tool:** Called _during_ generation to perform a task (our Exa Search Router).  

**Q3: Why didn’t Complete search trigger on my complex question?**  
Complete mode is resource-heavy. It only activates when your prompt explicitly asks for “deep research” or similar. Otherwise it defaults to Quick.

---

## 6. Architecture Diagram  
![Architecture Diagram](https://github.com/user-attachments/assets/e79f7658-020f-4804-8d16-e4414ad781e8)

