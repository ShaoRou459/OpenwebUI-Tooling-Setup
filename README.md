# OpenWebUI Tooling Suite Documentation

[中文Readme](https://github.com/ShaoRou459/OpenwebUI-Tooling-Setup/blob/main/README_zh.md)

## 1. Backstory and Purpose
Standard OpenWebUI models are passive, requiring manual tool selection and often providing weak search results. This project solves that by introducing an intelligent, two-layer router that automates tool use for web search, image generation, and code execution + allows for other tools to be added for future expansion. It gives your local models the autonomous capabilities of leading AI services like ChatGPT and Gemini, making them proactive and far more powerful.

Auto Tool Router inspired and based on Auto-Tool v2 by mhwhgm.

### Quick Navigation
*   [Features](#2-overview)
*   [Valves and Model Configuration](#3-configuration-guide-valves--model-selection)
*   [Installation & Setup](#4-setup--usage)
*   [Frequently Asked Questions](#5-faq)


## 2. Overview
This suite works using a master router (`Auto Tool Selector`) that intercepts every user prompt. It analyzes the request (including any images) and decides on the best action:

*   **Chat:** If it's a simple query, it's passed directly to the main model for a fast answer.
*   **Delegate:** If it's a complex task, it's forwarded to a specialized tool for image generation, code execution, or advanced web research via the `Exa Search Router`.

#### Summary of Key Features

*   **Autonomous Tool Delegation:** Intelligently analyzes user intent to automatically call the right tool for the job—be it the code interpreter, image generator, or the advanced web search agent. No manual selection required.

*   **Multi-Strategy Web Search:** Powered by Exa.ai, the search tool dynamically selects one of three powerful modes:
    *   **Crawl:** Directly reads and processes a specific URL.
    *   **Quick:** Delivers fast, factual answers by summarizing top sources, often in under 5 seconds.
    *   **Complete:** Executes a multi-step, agentic research process that synthesizes information from a wide range of sources for deep, comprehensive reports.

*   **Enhanced Image Generation:** Automatically improves user prompts to generate higher-quality, more relevant visuals. The system also feeds visual context back to the chat model, enabling seamless compatibility with tools like GPT-Image-1.

*   **Jupyter-Ready Code Interpreter:** Offers deep integration with **Jupyter** for complex data analysis and convenient file returns, while still fully supporting the default OpenWebUI code interpreter.

*   **Transparent Operation:** Provides clear, real-time status updates so you're never left guessing what the tool is doing.

## Demo:

https://github.com/user-attachments/assets/696cb316-c160-4210-a0dc-f87a04be1647

---
## 3. Configuration Guide: Valves & Model Selection
This suite is designed to be highly configurable through a system called `Valves`. Each tool has its own set of valves that allow you to fine-tune its behavior, balancing cost, speed, and intelligence.

### Auto Tool Selector `Valves`
* **`helper_model`**:
    * **Purpose:** Specifies the language model used for the primary routing decisions (e.g., choosing between search, image generation, or none).
    * **Recommendation:** A fast and efficient model is ideal. **`GPT-4.1-mini`** or **`Llama3-8B`** are excellent choices.
* **`vision_model`**:
    * **Purpose:** Specifies the model used to analyze images and generate text descriptions.
    * **Recommendation:** A high-quality and fast multi-modal model is required. **`GPT 4.1`** or **`Gemini 2.5 Flash`** are recommended.
* **`history_char_limit`**:
    * **Purpose:** Sets the maximum number of characters for each message included in the conversation history snippet sent to the router. This helps keep the context concise and within token limits.

### Exa Search Router `Valves`
* **`exa_api_key`**:
    * **Purpose:** Your API key for the Exa search service. This is required for the tool to function.
* **`router_model`**:
    * **Purpose:** The model used to make the internal `CRAWL`, `STANDARD`, or `COMPLETE` decision.
    * **Recommendation:** Similar to the master router, a fast and cheap model like **`GPT-4.1-mini`** is best.
* **`quick_search_model`**:
    * **Purpose:** The model used for all steps in the `STANDARD` path (refining the query, summarizing the results).
    * **Recommendation:** A balanced model is good for this path. **`GPT-4.1-mini`** is a strong contender.
* **`complete_agent_model`**:
    * **Purpose:** The "smart" model for all agentic steps in the `COMPLETE` path (refining, deciding, planning, and generating new queries).
    * **Recommendation:** This role requires high intelligence. A powerful model like **`GPT-4.1`**, **`Gemini 2.0 Flash`**, or **`Claude 4 Sonnet`** is highly recommended.
* **`complete_summarizer_model`**:
    * **Purpose:** A dedicated, high-quality model for the final synthesis step of the `COMPLETE` path.
    * **Recommendation:** This model needs high intelligence and a large context window. **`Gemini 2.5 Flash`** or **`Llama 4 Maverick`** are excellent choices.

#### Search Depth Parameters
* **`quick_urls_to_search`**: Number of URLs to find in a `STANDARD` search.
* **`quick_queries_to_crawl`**: Number of top URLs to actually read from the found results in a `STANDARD` search.
* **`quick_max_context_chars`**: Maximum characters of combined text from all sources to send to the summarizer in a `STANDARD` search.
* **`complete_urls_to_search_per_query`**: Number of URLs to find for _each sub-query_ in a `COMPLETE` search.
* **`complete_queries_to_crawl`**: Number of top URLs to read for _each sub-query_ in a `COMPLETE` search.
* **`complete_queries_to_generate`**: Number of new sub-queries for the agent to generate in each research loop.
* **`complete_max_search_iterations`**: The maximum number of "think, search, analyze" loops the agent can perform in a `COMPLETE` search.
---
## 4. Setup & Usage

#### **Step 1 (Jupyter Users ONLY): Enable File Returns**
**⚠️ CRITICAL: This step is mandatory if you use a Jupyter backend for code interpretation.**
To allow the AI to return generated files (like charts, CSVs, etc.) as download links, you **must** place the `uploader.py` script in the **root directory of your Jupyter environment**. This is the main folder where your Jupyter server starts. The AI cannot send files back to you without this script in the correct location.

#### **Step 2: Install Required Library**
The `exa_py` library must be installed inside the OpenWebUI Docker container. This is a one-time setup.
Run the following commands on the machine hosting your Docker instance:
1.  **Access the container's terminal:**
    ```bash
    docker exec -it open-webui bash
    ```
2.  **Install the library using pip:**
    ```bash
    pip install exa_py
    ```
3.  **Exit the container:**
    ```bash
    exit
    ```
4.  **Restart the container to apply changes:**
    ```bash
    docker restart open-webui
    ```

#### **Step 3: Add and Configure the Routers**
1.  **Choose the Correct Master Router File:**
    *   For the **default** OpenWebUI code interpreter, use `auto_tool_selector_default_code_interpter.py`.
    *   For a **Jupyter** backend, use `auto_tool_selector_jupyter_ci.py`.

2.  **Add the Tools in the Admin Panel:**
    *   **Create the Function (Master Router):** Navigate to `Admin Settings > Functions > New Function`. Paste the code from your chosen `Auto Tool Selector` file here.
    *   **Create the Tool (Search Router):** Go to `Workspace > Tools > New Tool`. Paste the code for the `Exa Search Router` here. Give it the exact ID `exa_router_search`.

3.  **Set the API Key:**
    *   In the settings for the `Exa Search Router` tool, find the `Valves` section and enter your Exa API key into the `exa_api_key` field.

#### **Step 4: Activate the System**
*   To activate the suite, go to the settings for the model you want to enhance and enable **only** the `Auto Tool Selector` function.
*   **Important:** The master router calls the other tools automatically. Do **not** enable the `Exa Search Router` directly on your model, as this will bypass the master router.
---
## 5. FAQ
**Why do I need to install `exa_py` manually?**
OpenWebUI's tool interface does not automatically install Python libraries. The `Requirements` line in the tool's code is for documentation only. You must manually install the library inside the Docker container to make it available to the Python runtime.

**What's the difference between a "Function" and a "Tool" in the setup?**
* **Function (Filter):** This runs _before_ the main chat model to intercept and reroute the user's prompt. Our `Auto Tool Selector` is a Function.
* **Tool:** This is a standard tool the model can call _after_ receiving the prompt. Our `Exa Search Router` is a Tool called by the master router.

## 6. Diagram
<img width="6663" height="4095" alt="Drawing 2025-07-14 17 48 55 excalidraw" src="https://github.com/user-attachments/assets/e79f7658-020f-4804-8d16-e4414ad781e8" />

**Why doesn't the `COMPLETE` search run for my complex questions?**
The `COMPLETE` search mode is resource-intensive and is triggered only when your prompt explicitly contains a phrase like "deep research." For all other queries, the system uses the faster `STANDARD` search to balance performance and thoroughness.
