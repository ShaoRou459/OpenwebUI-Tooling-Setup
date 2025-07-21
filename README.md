# OpenWebUI Tooling Suite Documentation

## 1. Backstory and Purpose

This project was born from a desire to address a core limitation in the standard OpenWebUI experience: the lack of intelligent, autonomous tool use. Unlike services like ChatGPT or Gemini that can seamlessly call upon a powerful suite of tools, models in OpenWebUI often feel passive, requiring manual intervention for even basic tasks. Furthermore, the default search tool is often ineffective, providing raw data dumps instead of synthesized answers.

This suite aims to solve these issues by introducing an intelligent router that automatically delegates tasks to one of three fundamental tools: web search, image generation, and a code interpreter. More importantly, it doesn't just call these tools—it fundamentally enhances their capabilities to bridge the gap with leading AI services.

---

## 2. Overview

This suite is built around a two-part system that works in tandem: the **Auto Tool Selector** and the **Exa Search Router**. The Auto Tool Selector acts as the central "brain" or master router, analyzing every user prompt to make an initial decision. The Exa Search Router is a highly specialized research agent that the master router can call upon when it determines a web search is necessary.

The core philosophy of this suite is best understood by following the flow of a user's request. When a user submits a prompt, it is first intercepted by the **Auto Tool Selector**. This router analyzes the prompt's text and any included images to determine the user's intent. If it decides a specialized task is required (like searching the web or generating an image), it forwards the request to the appropriate tool. If no special tool is needed, the prompt is passed directly to the main chat model. This ensures that simple questions get fast answers, while complex tasks are handled by the most capable agent for the job.

#### Summary of Key Features

* **Intelligent, Multi-Layered Routing:** The system uses a "chain-of-thought" process at multiple levels, first to select the right tool, and then again within the search tool to select the right research strategy.
* **Dynamic, Multi-Modal Understanding:** Natively handles both text and image-based queries, automatically getting context from images to inform its decisions.
* **Multi-Strategy Research Agent:** The web search tool isn't one-size-fits-all. It dynamically chooses between several modes based on the user's needs: `CRAWL` for reading a specific webpage, `STANDARD` for fast, factual lookups, or `COMPLETE` for deep, iterative research dives.
* **Enhanced Image Generation:** The image tool goes beyond simple generation by using a helper model to refine user ideas into more effective prompts, ensuring higher-quality and more relevant visual outputs.
* **Jupyter-Ready Code Interpreter:** Features a code interpreter designed to integrate seamlessly with a Jupyter Notebook environment for complex data analysis and code execution, while still functioning for simpler tasks without it.
* **Robust & Resilient Execution:** Built-in retry mechanisms and graceful failure logic ensure the system is reliable and provides the best possible output even when external services have temporary issues.
* **Transparent User Experience:** Provides clear, dynamic status updates to the user, showing them exactly what the tool is doing at every step of the process.

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
* **`vision_injection_models`**:
    * **Purpose:** A list of non-vision model IDs that should receive the AI-generated image description. This gives "sight" to powerful text-only models.
    * **Example:** `["Llama3-70B", "Claude-3-Opus"]`
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
* **`complete_urls_to_search_per_query`**: Number of URLs to find for *each sub-query* in a `COMPLETE` search.
* **`complete_queries_to_crawl`**: Number of top URLs to read for *each sub-query* in a `COMPLETE` search.
* **`complete_queries_to_generate`**: Number of new sub-queries for the agent to generate in each research loop.
* **`complete_max_search_iterations`**: The maximum number of "think, search, analyze" loops the agent can perform in a `COMPLETE` search.

---

## 4. Setup & Usage

#### Library Requirements

* **`exa_py`**: The official Python client for the Exa search API.

#### Installation

The `exa_py` library must be manually installed inside the OpenWebUI Docker container. This is a one-time setup. If you ever update or recreate your OpenWebUI container, you will need to repeat these steps.

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

#### Configuration

1.  **Add the Tools in the Admin Panel:**
    * Navigate to your OpenWebUI admin panel (Admin Settings > Functions).
    * **Create the Function:** Click "New Function”. Paste the complete code for the `Auto Tool Selector` here.
    * **Create the Tool:** Go to Workspace > Tools, click "New Tool". Paste the complete code for the `Exa Search Router` here. Give it the ID `exa_router_search`.

2.  **Set the API Key:**
    * In the settings for the `Exa Search Router` tool you just created, find the `Valves` section.
    * Enter your Exa API key into the `exa_api_key` field.

3.  **Enable the System:**
    * To activate the suite, you only need to enable the `Auto Tool Selector`. Go to Models, select the model you wish to enhance, and enable the `Auto Tool Selector` function.
    * **Important:** The master router will handle calling the search tool automatically, so you do not need to enable the `Exa Search Router` directly on the model.

4.  **(Optional) Tweak Numerical Valves:**
    * In the `Exa Search Router` settings, you can adjust the "Search Depth Parameters" to control how many sources the tool reads and how many research loops it performs. The default values are balanced, but you can increase them for more thorough research or decrease them for faster results.

---

## 5. FAQ

**Why do I need to install `exa_py` manually?**
OpenWebUI's tool interface does not automatically install Python libraries listed in a tool's description. The `Requirements` line is for documentation purposes. You must manually install the library inside the Docker container to make it available to the Python runtime environment.

**What's the difference between a "Function" and a "Tool" in the setup?**

* **Function (Filter):** This is a special type of tool that runs *before* the main chat model. Our `Auto Tool Selector` is a Function because it needs to intercept and route the user's prompt.
* **Tool:** This is a standard tool that the model can decide to call *after* it has received the prompt. Our `Exa Search Router` is a Tool because it is called upon by the `Auto Tool Selector`.

**Why doesn't the `COMPLETE` search run for my complex questions?**
The `COMPLETE` search mode is intentionally designed to be used sparingly, as it is resource-intensive. It will only be activated if the user's prompt explicitly contains the phrase "deep research" or a similar, direct instruction. For all other complex queries, the system will default to the more efficient `STANDARD` search.
